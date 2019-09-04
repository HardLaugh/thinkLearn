> # **_YOLACT:Real-time Instance Segmentation_**

基于一个简单的全卷积网络，在`MS COCO`数据集上单卡（`TitanXP`）实现`29.8map`，速度达到`33fps`。相比于目前现有的`state-of-the-art`的方法，速度明显快。

## `Motivation`

目前现有的`state-of-the-art`的目标分割网络，诸如`Mask RCNN`和`FCIS`等是基于`Faster R-CNN`和`R-FCN`。然而，这些方法关注的是性能而不在乎速度。与之前目标检测时期`two-stage`和`one-stage`两条道一起走的情况不同，目标分割方向，并没有人太多研究关注类似于`one-stage`一样的trade-off速度和性能。
