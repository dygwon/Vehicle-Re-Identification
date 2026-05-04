# Vehicle Re-Identification

## Overview

Models can be found on huggingface under the [Vehicle Re-Identification](https://huggingface.co/collections/dgwon/vehicle-re-identification) collection.

## Data

### Object Detection

**coco-2017**

Subsetted to four classes: cars, trucks, motorcycles, 

### Feature Extraction

VeRi-776

## Models

### Object Detection

yolov8s

### Feature Extraction

resnet-34

## Target

**Jetson Orin Nano**

JetPack 6.2

### Object Detection

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=/home/dan/.cache/huggingface/hub/models--dgwon--yolov8s-coco2017-vehicle-detection-onnx/snapshots/a8559d5265603f25fe5c599b9f0ce2f51f0adf74/best.onnx \
    --saveEngine=best_detect_int8.engine \
    --int8 --fp16 \
    --calib=/home/dan/.cache/huggingface/hub/models--dgwon--yolov8s-coco2017-vehicle-detection-onnx/snapshots/a8559d5265603f25fe5c599b9f0ce2f51f0adf74/calib_detect.cache
```

### Feature Extraction

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=/home/dan/.cache/huggingface/hub/models--dgwon--resnet-34-veri776-onnx/snapshots/e75b1b1cd60cbe850d7577bcf2de88d23154aaa3/resnet34_veri776.onnx \
    --saveEngine=resnet34_veri776_int8.engine \
    --int8 --fp16 \
    --calib=/home/dan/.cache/huggingface/hub/models--dgwon--resnet-34-veri776-onnx/snapshots/e75b1b1cd60cbe850d7577bcf2de88d23154aaa3/calib.cache \
    --minShapes=input:1x3x256x256 \
    --optShapes=input:8x3x256x256 \
    --maxShapes=input:16x3x256x256 \
    --memPoolSize=workspace:2048 \
    --builderOptimizationLevel=4
```
