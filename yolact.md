> # **_YOLACT:Real-time Instance Segmentation_**
***
基于一个简单的全卷积网络，在`MS COCO`数据集上单卡（`TitanXP`）实现`29.8map`，速度达到`33fps`。相比于目前现有的`state-of-the-art`的方法，速度明显快。

## 动机

目前现有的`state-of-the-art`的目标分割网络，诸如`Mask RCNN`和`FCIS`等是基于`Faster R-CNN`和`R-FCN`。然而，这些方法关注的是性能而不在乎速度。与之前目标检测时期`two-stage`和`one-stage`两条道一起走的情况不同，目标分割方向，并没有人太多研究关注类似于`one-stage`一样的`trade-off`速度和性能。因此，yolact的工作主要是填补这一研究方向的空白，就如`SSD`和`YOLO`填补目标检测那样。

## 问题

- `two-stage`的目标分割方法，通过`feature localization`产生features，然后送进mask predictor分支，这一套下来实质是固有串联，难以加速。

## 方案

- （1）生成不依赖于预测框的non-local `prototype masks`，`prototype masks`是关于整个image的masks.
- （2）针对生成的一系列固定数量的`prototype masks`，预测一组线性组合参数，通过线性参数是得`prototype masks`合成最终的`per instance`

***
