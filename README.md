# Vehicle Re-Identification

## Overview

Models (full finetuned and onnx exports) can be found on huggingface under the [Vehicle Re-Identification](https://huggingface.co/collections/dgwon/vehicle-re-identification) collection. Model engines built for the target device can be found in this repository under the `engines/` directory.

## Environment Setup

```bash
$ conda create -n reid python=3.10 -y
$ conda activate reid
$ pip install python-dotenv ultralytics fiftyone torch torchvision pillow numpy onnx onnxscript
```

## Data

### Object Detection

**coco-2017**

This dataset will be downloaded when running object detection training notebook.

Subsetted to four classes: cars, motorcycle, bus, and truck

### Feature Extraction

**VeRi-776**

A zip file for this dataset can be found at [dgwon/resnet-34-veri776-onnx](https://huggingface.co/dgwon/resnet-34-veri776-onnx)

## Models

All model training was completed on one RTX A6000.

### Object Detection

**yolov8s**

- training: `detection/train-object-detection.ipynb` (run this notebook)

### Feature Extraction

**resnet-34**

- training: `python feature-extraction/finetune_resnet_veri776.py --data-root /PATH/TO/VERI/`
- sample calibration data: `python feature-extraction/sample_calib_resnet.py`
- export to onnx: `python export_resnet_onnx.py`
- evaluate target build: `python feature-extraction/eval_resnet_engine.py`

## Target

**Jetson Orin Nano**

JetPack 6.2

### Object Detection

Build the engine.

```bash
/usr/src/tensorrt/bin/trtexec \
	--onnx=/home/dan/.cache/huggingface/hub/models--dgwon--yolov8s-coco2017-vehicle-detection-onnx/snapshots/a8559d5265603f25fe5c599b9f0ce2f51f0adf74/best.onnx \
	--saveEngine=engines/best_detect_fp16.engine \
	--fp16
```

Benchmarking

```bash
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=engines/best_detect_fp16.engine \
    --iterations=100 \
    --warmUp=500 \
    --avgRuns=100
```

### Feature Extraction

Build the engine.

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/dan/.cache/huggingface/hub/models--dgwon--resnet-34-veri776-onnx/snapshots/e75b1b1cd60cbe850d7577bcf2de88d23154aaa3/resnet34_veri776.onnx \
  --saveEngine=engines/resnet34_veri776_int8.engine \
  --int8 --fp16 \
  --calib=/home/dan/.cache/huggingface/hub/models--dgwon--resnet-34-veri776-onnx/snapshots/e75b1b1cd60cbe850d7577bcf2de88d23154aaa3/calib.cache \
  --minShapes=input:1x3x256x256 --optShapes=input:8x3x256x256 --maxShapes=input:16x3x256x256 \
  --memPoolSize=workspace:2048 --builderOptimizationLevel=4
```

Benchmarking

```bash
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=engines/resnet34_veri776_int8.engine \
    --shapes=input:1x3x256x256 \
    --warmUp=500 \
    --iterations=1000 \
    --avgRuns=100 \
    --useCudaGraph \
    --useSpinWait
```
