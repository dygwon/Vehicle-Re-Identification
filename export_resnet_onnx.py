import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np

# ---- adjust to match your finetune ----
CHECKPOINT   = "checkpoints/resnet34_veri776_deploy.pt"
ONNX_PATH    = "resnet34_veri776.onnx"
NUM_CLASSES  = 576          # VeRi-776 train identities
INPUT_H      = 256
INPUT_W      = 256
EXPORT_FEATS = True        # True -> output 512-d embeddings (drop fc)
OPSET        = 17
# ---------------------------------------

# Build the architecture, then load weights
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

ckpt = torch.load(CHECKPOINT, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
# strip common prefixes (DDP / Lightning)
state_dict = {k.replace("module.", "").replace("model.", "", 1): v
              for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"missing={len(missing)} unexpected={len(unexpected)}")

# For re-ID deployment you usually want pre-fc embeddings.
if EXPORT_FEATS:
    class FeatExtractor(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.layers = nn.Sequential(m.layer1, m.layer2, m.layer3, m.layer4)
            self.pool = m.avgpool
        def forward(self, x):
            x = self.stem(x)
            x = self.layers(x)
            x = self.pool(x).flatten(1)
            return torch.nn.functional.normalize(x, dim=1)  # cosine-ready
    model = FeatExtractor(model)

model.eval()

dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=OPSET,
    do_constant_folding=True,
    dynamo=False,           # use legacy TorchScript exporter; avoids the
                            # version_converter axes-attribute assertion
                            # in the new dynamo-based optimizer
)

# Validate the export end-to-end
onnx.checker.check_model(onnx.load(ONNX_PATH))

# On HPC nodes with SLURM cpusets/cgroups, ORT's default thread-affinity
# pinning fails with EINVAL on cores outside the allowed mask. Setting
# thread counts explicitly bypasses the affinity step.
so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
sess = ort.InferenceSession(ONNX_PATH, sess_options=so,
                            providers=["CPUExecutionProvider"])

with torch.no_grad():
    torch_out = model(dummy).numpy()
ort_out = sess.run(None, {"input": dummy.numpy()})[0]
print("max abs diff:", np.abs(torch_out - ort_out).max())
print(f"wrote {ONNX_PATH}")
