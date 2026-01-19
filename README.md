# FastestDetNext
Various improvements have been made to make FastestDet even lighter and faster.
# This fork `custom` branch aims to improve the official implementation. A huge thank you to the authors of FastestDet.

|Inference image<br>96x96|→|Original image<br>640x425|
|:-:|:-:|:-:|
|<img width="96" height="96" alt="image" src="https://github.com/user-attachments/assets/99f2d9cf-cde9-4a51-962d-86e6b591ba93" />|→|<img width="640" height="425" alt="image" src="https://github.com/user-attachments/assets/24881eea-c940-4530-a5ca-5c53d36785af" />|

1. Optimizer: SGD/AdamW
2. Scheduler: Step/Cosine
3. Supports Distributed Data Parallel (DDP) training
4. EMA
5. Distillation
6. Resume
7. Toggle pyramid-levels: `P1 only`, `P2 only`, `P3 only`, `P1 + P2`, `P1 + P3`, `P2 + P3`, `P1 + P2 + P3`. For very low resolution models, including P2 and P3 in the architecture significantly reduces accuracy.
8. Added stride adjustment function using `--stride-half` and `--stride-quarter` flags. This changes the stride to 1/2 or 1/4, which significantly improves accuracy at the expense of increased computational cost.
9. Backbone ShuffleNetv2 improvements: `Shuffle-free shufflenet`
10. SE (Squeeze-and-Excitation) block/eSE (efficient Squeeze-and-Excitation) block.
11. Skip-Residual block.
12. Added `--use-spp-separate-1x1` to use separate 1x1 conversion for each SPP branch, which is important to minimize quantization error.
13. Free expansion of the number of channels by `--stage-out-channels`.
14. Freely expansion of the number of stages by `--stage-repeats`.
15. Added `--multi-label-robust-mode`, which switches the class head to multi-label mode (sigmoid + BCEWithLogits, multi-hot target in the same cell) when specified.
16. Params and FLOPs Profiler: `profile_onnx.py`.
    ```bash
    uv run python profile_onnx.py model.onnx

    Params: 72,861 (72.861 Kparams)
    FLOPs:  10,908,672 (0.011 GFLOPs)
    ```
17. Replace pycocotools with [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval)

---

***2022.7.14:Optimize loss, adopt IOU aware based on smooth L1, and the AP is significantly increased by 0.7***
# :zap:FastestDet:zap:
[![DOI](https://zenodo.org/badge/508635170.svg)](https://zenodo.org/badge/latestdoi/508635170)
![image](https://img.shields.io/github/license/dog-qiuqiu/FastestDet)
![image](https://img.shields.io/github/stars/dog-qiuqiu/FastestDet?style=flat)
![image](https://github.com/dog-qiuqiu/FastestDet/blob/main/data/data.png)
* ***Faster! Stronger! Simpler!***
* ***It has better performance and simpler feature map post-processing than Yolo-fastest***
* ***The performance is 10% higher than Yolo-fastest***
* ***The coco evaluation index increased by 1.2% compared with the map0.5 of Yolo-fastestv2***
* ***Algorithm intro: https://zhuanlan.zhihu.com/p/536500269  QQ group: 1062122604***
# Evaluating indicator/Benchmark
Network|mAPval 0.5|mAPval 0.5:0.95|Resolution|Run Time(4xCore)|Run Time(1xCore)|Params(M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[yolov5s](https://github.com/ultralytics/yolov5)|56.8%|37.4%|640X640|395.31ms|1139.16ms|7.2M
[yolov6n](https://github.com/meituan/YOLOv6)|-|30.8%|416X416|109.24ms|445.44ms|4.3M
[yolox-nano](https://github.com/Megvii-BaseDetection/YOLOX)|-|25.8%|416X416|76.31ms|191.16ms|0.91M
[nanodet_m](https://github.com/RangiLyu/nanodet)|-|20.6%|320X320|49.24ms|160.35ms|0.95M
[yolo-fastestv1.1](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|24.40%|-|320X320|26.60ms|75.74ms|0.35M
[yolo-fastestv2](https://github.com/dog-qiuqiu/Yolo-FastestV2/tree/main/modelzoo)|24.10%|-|352X352|23.8ms|68.9ms|0.25M
FastestDet|25.3%|13.0%|352X352|23.51ms|70.62ms|0.24M
* ***Test platform Radxa Rock3A RK3568 ARM Cortex-A55 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
* ***CPU lock frequency 2.0GHz***
# Improvement
* Anchor-Free
* Single scale detector head
* Cross grid multiple candidate targets
* Dynamic positive and negative sample allocation
# Multi-platform benchmark
Equipment|Computing backend|System|Framework|Run time(Single core)|Run time(Multi core)
:---:|:---:|:---:|:---:|:---:|:---:
Radxa rock3a|RK3568(arm-cpu)|Linux(aarch64)|ncnn|70.62ms|23.51ms
Radxa rock3a|RK3568(NPU)|Linux(aarch64)|rknn|28ms|-
Qualcomm|Snapdragon 835(arm-cpu)|Android(aarch64)|ncnn|32.34ms|16.24ms
Intel|i7-8700(X86-cpu)|Linux(amd64)|ncnn|4.51ms|4.33ms
# How to use
## Dependent installation
```bash
git clone https://github.com/PINTO0309/FastestDetNext.git && cd FastestDetNext
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```
## Test
* Picture test
```bash
uv run python test.py \
--yaml configs/coco.yaml \
--weight weights/weight_AP05:0.253207_280-epoch.pth \
--img data/3.jpg
```
<div align=center>
<img src="https://github.com/dog-qiuqiu/FastestDet/blob/main/result.png" />
</div>

## How to train
### Building data sets(The dataset is constructed in the same way as darknet yolo)
* The format of the data set is the same as that of Darknet Yolo, Each image corresponds to a .txt label file. The label format is also based on Darknet Yolo's data set label format: "category cx cy wh", where category is the category subscript, cx, cy are the coordinates of the center point of the normalized label box, and w, h are the normalized label box The width and height, .txt label file content example as follows:
  ```
  11 0.344192634561 0.611 0.416430594901 0.262
  14 0.509915014164 0.51 0.974504249292 0.972
  ```
* The image and its corresponding label file have the same name and are stored in the same directory. The data file structure is as follows:
  ```
  dataset
  ├── train
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  └── val
      ├── 000043.jpg
      ├── 000043.txt
      ├── 000057.jpg
      ├── 000057.txt
      ├── 000070.jpg
      └── 000070.txt
  ```
* Generate a dataset path .txt file, the example content is as follows：

  train.txt
  ```
  dataset/train/000001.jpg
  dataset/train/000002.jpg
  dataset/train/000003.jpg
  ```
  val.txt
  ```
  dataset/val/000070.jpg
  dataset/val/000043.jpg
  dataset/val/000057.jpg
  ```
* Generate the .names category label file, the sample content is as follows:

  category.names
  ```
  person
  bicycle
  car
  motorbike
  ...
  ```
* The directory structure of the finally constructed training data set is as follows:
  ```
  dataset
  ├── category.names        # .names category label file
  ├── train                 # train dataset
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  ├── train.txt              # train dataset path .txt file
  ├── val                    # val dataset
  │   ├── 000043.jpg
  │   ├── 000043.txt
  │   ├── 000057.jpg
  │   ├── 000057.txt
  │   ├── 000070.jpg
  │   └── 000070.txt
  └── val.txt                # val dataset path .txt file

  ```
### Build the training .yaml configuration file
* Reference./configs/coco.yaml
  ```
  DATASET:
    TRAIN: "/home/qiuqiu/Desktop/coco2017/train2017.txt"  # Train dataset path .txt file
    VAL: "/home/qiuqiu/Desktop/coco2017/val2017.txt"      # Val dataset path .txt file
    NAMES: "dataset/coco128/coco.names"                   # .names category label file
  MODEL:
    INPUT_WIDTH: 352                                      # The width of the model input image
    INPUT_HEIGHT: 352                                     # The height of the model input image
  TRAIN:
    LR: 0.001                                             # Train learn rate
    WARMUP: true                                          # Trun on warm up
    BATCH_SIZE: 64                                        # Batch size
    END_EPOCH: 350                                        # Train epichs
    MILESTIONES:                                          # Declining learning rate steps
      - 150
      - 250
      - 300
  ```
* Number of classes is derived from `TRAIN.CLASSES` (if set) or the number of lines in `DATASET.NAMES`.
* `--classes` overrides `TRAIN.CLASSES` and updates the derived class count.

### Stride details
* Backbone downsampling: `first_conv` stride 2, `maxpool` stride 2, and the first block of each stage (`stage2/3/4`) uses stride 2.
* Feature map strides (relative to input):
  * `P1` (stage2 output): stride 8
  * `P2` (stage3 output): stride 16
  * `P3` (stage4 output): stride 32

### Stride adjustment
* Backbone stride presets (requires re-training; not compatible with existing checkpoints):
  * `--stride-half` → P1/P2/P3 = 4/8/16
  * `--stride-quarter` → P1/P2/P3 = 2/4/8
* If neither flag is set, the default strides are used: P1/P2/P3 = 8/16/32.

### `--pyramid-levels`
* Comma-separated list of pyramid levels to fuse: `P1,P2,P3` (default) or any subset (e.g. `P1,P3`, `P2`).
* When fusing multiple levels, features are aligned to `P2` resolution: `P1` is downsampled (avg pool stride 2), `P3` is upsampled (x2).
* When using a single level, its native stride is used (P1=8, P2=16, P3=32).

### Output tensor meaning
* Output shape is `float32[B, 1 + 4 + NC, H, W]`.
  * `B`: batch size
  * `NC`: number of classes (derived from `TRAIN.CLASSES` or `DATASET.NAMES`)
  * `H, W`: feature map size (depends on input size and selected stride)
* Channel layout per spatial location:
  * `0`: objectness (sigmoid)
  * `1-4`: box regression (tx, ty, tw, th)
  * `5..(4+NC)`: class probabilities (softmax)

### Score calculation
* Final score per box is computed as:
  * `score = (objectness ** 0.6) * (max_class_prob ** 0.4)`

### Train
- Perform training tasks
  ```bash
  uv run python train.py \
  --yaml configs/coco.yaml
  ```

  <details><summary>Click to expand</summary>

  ```bash
  SIZE=64x64
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 20
  
  SIZE=60x80
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 20
  
  SIZE=96x96
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 20
  
  SIZE=90x120
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 20
  
  SIZE=120x160
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 32 \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 20
  ```

  </details>

  <details><summary>Click to expand</summary>

  ```bash
  SIZE=64x64
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 20
  
  SIZE=60x80
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 20
  
  SIZE=96x96
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 20
  
  SIZE=90x120
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 20
  
  SIZE=120x160
  uv run python train.py \
  --exp-name exp_x3_00_x1_00_${SIZE}_lr0.01000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 32 \
  --lr 0.01000 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 3.00 \
  --stage-repeats 1.00 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 20
  ```

  </details>

  <details><summary>Click to expand</summary>

  ```bash
  SIZE=64x64
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 50
  
  SIZE=60x80
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 50
  
  SIZE=96x96
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 50
  
  SIZE=90x120
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 50
  
  SIZE=120x160
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 50
  
  ###################
  
  SIZE=64x64
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50
  
  SIZE=60x80
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50
  
  SIZE=96x96
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50
  
  SIZE=90x120
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50
  
  SIZE=120x160
  python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00100 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50
  ```

  </details>

  <details><summary>Click to expand</summary>

  ```bash
  SIZE=64x64
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  SIZE=60x80
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  SIZE=96x96
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  SIZE=90x120
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  SIZE=120x160
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_half_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-half \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  ###################
  
  SIZE=64x64
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  SIZE=60x80
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  SIZE=96x96
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  SIZE=90x120
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  
  SIZE=120x160
  uv run python train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.001000_P1_quarter_sppsep_09cls_mlrm \
  --yaml configs/uhd09ft.yaml \
  --batch-size 60 \
  --num-workers 60 \
  --lr 0.00010 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --use-spp-separate-1x1 \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --stride-quarter \
  --val-interval 50 \
  --teacher-weight runs/exp_x50_00_x1_25_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/last_0100_0.799208.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5
  ```

  </details>

### DDP (torchrun) with distributed evaluation
* Run multi-GPU training and distributed evaluation with `torchrun`:
  ```bash
  SIZE=640x640
  torchrun --nproc_per_node=8 train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.00100_skipred_noema_noamp_P1_09cls_mlrm \
  --yaml configs/uhd09.yaml \
  --batch-size 8 \
  --num-workers 8 \
  --lr 0.00800 \
  --epoch 300 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --val-interval 10

  SIZE=640x640
  torchrun --nproc_per_node=8 train.py \
  --exp-name exp_x50_00_x1_25_${SIZE}_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill \
  --weight runs/exp_x50_00_x1_25_128x128_lr0.00100_skipred_noema_noamp_P1_09cls_mlrm/best_0295_0.421461.pth \
  --yaml configs/uhd09ft.yaml \
  --batch-size 5 \
  --num-workers 5 \
  --lr 0.00080 \
  --epoch 100 \
  --img-size ${SIZE} \
  --opencv_inter_nearest \
  --stage-out-channels 50.00 \
  --stage-repeats 1.25 \
  --use-skip-residual \
  --pyramid-levels P1 \
  --multi-label-robust-mode \
  --teacher-weight runs/exp_x50_00_x1_25_128x128_lr0.00100_skipred_noema_noamp_P1_09cls_mlrm/best_0295_0.421461.pth \
  --distill-weight-max 1.0 \
  --distill-temperature 1.5 \
  --val-interval 20
  ```
* `BATCH_SIZE` in the yaml is per GPU, so the global batch size is `BATCH_SIZE * nproc_per_node`.
* `--nproc_per_node` sets the number of GPU processes to launch on the node (typically the number of GPUs to use).
* Checkpoints, TensorBoard logs, ONNX export, and preview images are written by rank0 only.
* Generally, it is recommended to set the learning rate to `--nproc_per_node` times the default value.

### Evaluation
* Calculate map evaluation
  ```bash
  uv run python eval.py \
  --yaml configs/uhd09.yaml \
  --weight runs/distill/exp_x3_00_x1_00_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/best_0100_0.486152.pth \
  --img-size 640x640 \
  --batch-size 1

  uv run python eval.py \
  --yaml configs/uhd09.yaml \
  --weight runs/distill/exp_x3_00_x1_00_640x640_lr0.00010_skipred_noema_noamp_P1_09cls_mlrm_distill/best_0100_0.486152.pth \
  --img-size 320x320 \
  --batch-size 1
  ```
* COCO2017 evaluation
  ```
  creating index...
  index created!
  creating index...
  index created!
  Running per image evaluation...
  Evaluate annotation type *bbox*
  DONE (t=30.85s).
  Accumulating evaluation results...
  DONE (t=4.97s).
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.130
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.253
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.119
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.237
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.142
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.208
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.043
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.236
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.372
  ```
# Deploy
## 1. Export ONNX
```bash
STAGE_OUT_CHANNELS=3_00
STAGE_REPEATS=1_00
IMAGE_SIZE=96x96
RESIZE_MODE=opencv_inter_nearest
NUM_CLASSES=09
CKPT=runs/exp_x3_00_x1_00_64x64_lr0.01000_P1_half_sppsep_09cls_mlrm/best_0300_0.151091.pth
uv run python export_onnx.py \
--weight ${CKPT} \
--onnx-out fastestdetnext_x${STAGE_OUT_CHANNELS}_x${STAGE_REPEATS}_${IMAGE_SIZE}_${RESIZE_MODE}_cls${NUM_CLASSES}.onnx \
--opset 17
```
## 2. ESP-DL Quantization - PTQ (Post-Training Quantization)
### 2-1. Inference speed priority mode
```bash
uv run python quantize_onnx_model_for_esp32.py \
--list-path dataset/wholebody34/train.txt \
--onnx-model fastestdetnext_x3_00_x1_00_64x64_opencv_inter_nearest_cls09.onnx \
--espdl-model fastestdetnext_x3_00_x1_00_64x64_opencv_inter_nearest_cls09.espdl
```
### 2-2. Precision Priority Mode
```bash
uv run python quantize_onnx_model_for_esp32.py \
--list-path dataset/wholebody34/train.txt \
--onnx-model fastestdetnext_x3_00_x1_00_64x64_opencv_inter_nearest_cls09.onnx \
--espdl-model fastestdetnext_x3_00_x1_00_64x64_opencv_inter_nearest_cls09.espdl \
--int16-op-pattern /SPP/output3/output3.0/Conv
```

# Citation
* If you find this project useful in your research, please consider cite:
  ```
  @misc{=FastestDet,
    title={FastestDet: Ultra lightweight anchor-free real-time object detection algorithm.},
    author={xuehao.ma},
    howpublished = {\url{https://github.com/dog-qiuqiu/FastestDet}},
    year={2022}
  }
  ```
# Reference
* https://github.com/Tencent/ncnn
