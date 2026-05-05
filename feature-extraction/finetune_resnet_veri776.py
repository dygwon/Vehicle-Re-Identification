"""
finetune_resnet_veri776.py

Fine-tune a ResNet backbone on VeRi-776 for vehicle re-identification.
Training objective: ID cross-entropy + batch-hard triplet loss.

Usage:
    python finetune_resnet_veri776.py --data-root /path/to/VeRi

Dependencies: torch, torchvision, pillow, numpy.
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

VERI_FILENAME_RE = re.compile(r"^(\d+)_c(\d+)_")


def parse_veri_filename(fn: str):
    m = VERI_FILENAME_RE.match(fn)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VeRi776(Dataset):
    """
    VeRi-776 vehicle re-id dataset.

    Layout under `root`:
        image_train/   training images
        image_query/   eval queries
        image_test/    eval gallery

    __getitem__ returns 3-tuples: (img, pid, camid)
    """

    SPLIT_DIRS = {
        "train": "image_train",
        "query": "image_query",
        "gallery": "image_test",
    }

    def __init__(self, root, split="train", transform=None, pid2label=None):
        assert split in self.SPLIT_DIRS
        self.root = Path(root)
        self.split = split
        self.transform = transform

        img_dir = self.root / self.SPLIT_DIRS[split]
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {img_dir}")

        samples, pids = [], set()
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parsed = parse_veri_filename(fn)
            if parsed is None:
                continue
            pid, camid = parsed
            samples.append([str(img_dir / fn), pid, camid])
            pids.add(pid)

        if split == "train":
            if pid2label is None:
                pid2label = {pid: i for i, pid in enumerate(sorted(pids))}
            for s in samples:
                s[1] = pid2label[s[1]]

        self.samples = [tuple(s) for s in samples]
        self.pid2label = pid2label
        self.num_classes = len(pid2label) if pid2label is not None else len(pids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid, camid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


# ---------------------------------------------------------------------------
# PK batch sampler -- each batch has P identities, K images per identity
# ---------------------------------------------------------------------------


class PKBatchSampler(Sampler):
    def __init__(self, samples, P=16, K=4, num_iters=None):
        self.P, self.K = P, K
        self.batch_size = P * K
        self.pid_to_idxs = defaultdict(list)
        for i, sample in enumerate(samples):
            pid = sample[1]
            self.pid_to_idxs[pid].append(i)
        self.pids = list(self.pid_to_idxs.keys())
        self.num_iters = num_iters or (len(samples) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_iters):
            chosen = random.sample(self.pids, self.P)
            batch = []
            for pid in chosen:
                idxs = self.pid_to_idxs[pid]
                if len(idxs) >= self.K:
                    batch.extend(random.sample(idxs, self.K))
                else:
                    batch.extend(random.choices(idxs, k=self.K))
            yield batch

    def __len__(self):
        return self.num_iters


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, eps=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            t = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            t = (1 - self.eps) * t + self.eps / self.num_classes
        return (-t * log_probs).sum(1).mean()


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.ranking = nn.MarginRankingLoss(margin=margin)

    def forward(self, features, targets):
        dist = torch.cdist(features, features, p=2)
        same = targets.unsqueeze(1) == targets.unsqueeze(0)
        d_ap = (dist - (~same).float() * 1e9).max(dim=1).values
        d_an = (dist + same.float() * 1e9).min(dim=1).values
        return self.ranking(d_an, d_ap, torch.ones_like(d_an))


# ---------------------------------------------------------------------------
# Model: ResNet backbone + ID classifier
# ---------------------------------------------------------------------------

RESNET_FACTORY = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
}


class ResNetReID(nn.Module):
    """
    Standard ReID model: ResNet backbone with pooled feature output and a
    linear ID classifier.

    Train mode -> (id_logits, features)
    Eval  mode -> features
    """

    def __init__(
        self, num_classes, backbone="resnet50", pretrained=True, last_stride_one=True
    ):
        super().__init__()
        if backbone not in RESNET_FACTORY:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 'IMAGENET1K_V2' has noticeably better features for resnet50; the
        # smaller variants only ship V1.
        if pretrained:
            weights = "IMAGENET1K_V2" if backbone == "resnet50" else "IMAGENET1K_V1"
        else:
            weights = None
        net = RESNET_FACTORY[backbone](weights=weights)

        feat_dim = net.fc.in_features
        net.fc = nn.Identity()  # drop ImageNet classifier; we add our own

        # Bag-of-Tricks: setting the last block's stride to 1 keeps more
        # spatial resolution before the global average pool, which gives
        # ~1-2 mAP on ReID benchmarks essentially for free. The stride-2
        # conv lives in a different place depending on block type:
        #   BasicBlock (resnet18/34): stride is on conv1.
        #   Bottleneck (resnet50):    stride is on conv2.
        if last_stride_one:
            block = net.layer4[0]
            if hasattr(block, "conv3"):  # Bottleneck has conv1/conv2/conv3
                block.conv2.stride = (1, 1)
            else:  # BasicBlock has conv1/conv2
                block.conv1.stride = (1, 1)
            block.downsample[0].stride = (1, 1)

        self.backbone = net
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # (B, feat_dim) after GAP
        if not self.training:
            return features
        return self.classifier(features), features


# ---------------------------------------------------------------------------
# Evaluation: CMC and mAP, excluding same-id same-cam matches
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, pids, camids = [], [], []
    for imgs, pid, camid in loader:
        imgs = imgs.to(device, non_blocking=True)
        f = model(imgs)
        f = F.normalize(f, dim=1)
        feats.append(f.cpu())
        pids.append(pid)
        camids.append(camid)
    return torch.cat(feats), torch.cat(pids), torch.cat(camids)


def evaluate(qf, qp, qc, gf, gp, gc, max_rank=50):
    distmat = (1 - qf @ gf.t()).numpy()  # cosine distance, features L2-normed
    qp, qc, gp, gc = qp.numpy(), qc.numpy(), gp.numpy(), gc.numpy()

    indices = np.argsort(distmat, axis=1)
    matches = (gp[indices] == qp[:, None]).astype(np.int32)

    cmcs, aps = [], []
    for i in range(distmat.shape[0]):
        order = indices[i]
        keep = ~((gp[order] == qp[i]) & (gc[order] == qc[i]))
        m = matches[i][keep]
        if m.sum() == 0:
            continue
        cmc = np.minimum(m.cumsum(), 1)
        cmcs.append(cmc[:max_rank])
        ap = ((m.cumsum() / np.arange(1, len(m) + 1)) * m).sum() / m.sum()
        aps.append(ap)

    cmc = np.stack(cmcs).mean(axis=0)
    return cmc, float(np.mean(aps))


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def get_transforms(img_h, img_w):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Mild color jitter helps the model generalize across cameras with
            # different color reproduction.
            transforms.ColorJitter(
                brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05
            ),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, value="random"),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, test_tf


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Optimizer / scheduler factory
# ---------------------------------------------------------------------------


def make_optimizer_and_scheduler(model, lr, weight_decay, epochs, warmup=10):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True
    )

    def lr_at(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to VeRi-776 root")
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument(
        "--model-name", default="resnet50", choices=list(RESNET_FACTORY.keys())
    )
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--img-w", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--pretrained-weights",
        default=None,
        help="Path to manually-downloaded ImageNet weights "
        "(torchvision .pth file). If omitted, "
        "torchvision will fetch from download.pytorch.org "
        "on first run.",
    )
    parser.add_argument("--P", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Static input shapes -- let cuDNN autotune the conv kernels.
    torch.backends.cudnn.benchmark = True
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    train_tf, test_tf = get_transforms(args.img_h, args.img_w)
    pin = device.type == "cuda"

    train_ds = VeRi776(args.data_root, "train", transform=train_tf)
    query_ds = VeRi776(args.data_root, "query", transform=test_tf)
    gallery_ds = VeRi776(args.data_root, "gallery", transform=test_tf)
    print(f"Train:   {len(train_ds):>6} imgs / {train_ds.num_classes} ids")
    print(f"Query:   {len(query_ds):>6} imgs")
    print(f"Gallery: {len(gallery_ds):>6} imgs")

    train_loader = DataLoader(
        train_ds,
        batch_sampler=PKBatchSampler(train_ds.samples, P=args.P, K=args.K),
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    eval_kwargs = dict(
        batch_size=128, num_workers=args.num_workers, pin_memory=pin, shuffle=False
    )
    query_loader = DataLoader(query_ds, **eval_kwargs)
    gallery_loader = DataLoader(gallery_ds, **eval_kwargs)

    model = ResNetReID(
        num_classes=train_ds.num_classes,
        backbone=args.model_name,
        pretrained=(args.pretrained_weights is None),
    ).to(device)
    if args.pretrained_weights is not None:
        state = torch.load(args.pretrained_weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Drop the ImageNet classifier; our num_classes won't match.
        state = {
            k: v
            for k, v in state.items()
            if not k.startswith("fc.") and "classifier" not in k
        }
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_weights}")
        print(f"  missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    ce_loss = CrossEntropyLabelSmooth(train_ds.num_classes).to(device)
    tri_loss = BatchHardTripletLoss(margin=args.margin).to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, args.lr, args.weight_decay, args.epochs
    )

    best_path = save_dir / f"{args.model_name}_veri776_best.pt"
    log_path = save_dir / f"{args.model_name}_veri776_log.jsonl"
    # Truncate any existing log from a prior run so we get a clean history.
    log_path.write_text("")
    best_map = 0.0

    for epoch in range(args.epochs):
        model.train()
        running = defaultdict(float)
        t0 = time.time()

        for imgs, pids, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)

            id_logits, features = model(imgs)
            l_ce = ce_loss(id_logits, pids)
            l_tri = tri_loss(features, pids)
            loss = l_ce + l_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            running["ce"] += l_ce.item() * bs
            running["tri"] += l_tri.item() * bs
            running["total"] += loss.item() * bs
            running["n"] += bs

        scheduler.step()
        n = running["n"]
        epoch_metrics = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "ce": running["ce"] / n,
            "tri": running["tri"] / n,
            "total": running["total"] / n,
            "time_sec": time.time() - t0,
        }
        print(
            f"Epoch {epoch + 1:>3}/{args.epochs}  "
            f"lr={epoch_metrics['lr']:.2e}  "
            f"ce={epoch_metrics['ce']:.4f}  "
            f"tri={epoch_metrics['tri']:.4f}  "
            f"total={epoch_metrics['total']:.4f}  "
            f"time={epoch_metrics['time_sec']:.1f}s"
        )

        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            qf, qp, qc = extract_features(model, query_loader, device)
            gf, gp, gc = extract_features(model, gallery_loader, device)
            cmc, mAP = evaluate(qf, qp, qc, gf, gp, gc)
            print(
                f"  -> mAP={mAP * 100:.2f}  "
                f"R1={cmc[0] * 100:.2f}  "
                f"R5={cmc[4] * 100:.2f}  "
                f"R10={cmc[9] * 100:.2f}"
            )
            epoch_metrics.update(
                {
                    "mAP": mAP,
                    "rank1": float(cmc[0]),
                    "rank5": float(cmc[4]),
                    "rank10": float(cmc[9]),
                }
            )

            if mAP > best_map:
                best_map = mAP
                torch.save(
                    {
                        "model_name": args.model_name,
                        "state_dict": model.state_dict(),
                        "mAP": mAP,
                        "rank1": float(cmc[0]),
                        "epoch": epoch + 1,
                    },
                    best_path,
                )
                print(f"  -> saved new best (mAP={mAP * 100:.2f})")

        # Append this epoch's metrics to the JSONL log. Flush + fsync so the
        # log survives a crash or Ctrl-C cleanly.
        with open(log_path, "a") as f:
            f.write(json.dumps(epoch_metrics) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # Produce a deployment-ready, backbone-only checkpoint from the best
    # mAP weights.
    if best_path.is_file():
        ckpt = torch.load(best_path, map_location="cpu")
        full_state = ckpt["state_dict"]
        backbone_state = {
            k.replace("backbone.", "", 1): v
            for k, v in full_state.items()
            if k.startswith("backbone.")
        }
        deploy_path = save_dir / f"{args.model_name}_veri776_deploy.pt"
        torch.save(
            {
                "model_name": args.model_name,
                "state_dict": backbone_state,
                "img_size": (args.img_h, args.img_w),
                "feat_dim": model.feat_dim,
                "num_classes": train_ds.num_classes,
                "pid2label": train_ds.pid2label,
                "mAP": ckpt.get("mAP"),
                "rank1": ckpt.get("rank1"),
                "epoch": ckpt.get("epoch"),
            },
            deploy_path,
        )
        print(
            f"Saved deployment-ready backbone to {deploy_path} "
            f"(mAP={ckpt.get('mAP', 0) * 100:.2f})"
        )


if __name__ == "__main__":
    main()
