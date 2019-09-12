> # Detection-Based

- detecting：借助state-of-the-art的目标检测器框架
- mask head：在目标检测框架基础上，附加mask branch
***
# **_YOLACT:Real-time Instance Segmentation_**
***
[原文链接](https://arxiv.org/abs/1904.02689) : https://arxiv.org/abs/1904.02689  
基于一个简单的全卷积网络，在`MS COCO`数据集上单卡（`TitanXP`）实现`29.8map`，速度达到`33fps`。相比于目前现有的`state-of-the-art`的方法，速度明显快。

## 动机

目前现有的`state-of-the-art`的目标分割网络，诸如`Mask RCNN`和`FCIS`等是基于`Faster R-CNN`和`R-FCN`。然而，这些方法关注的是性能而不在乎速度。与之前目标检测时期`two-stage`和`one-stage`两条道一起走的情况不同，目标分割方向，并没有人太多研究关注类似于`one-stage`一样的`trade-off`速度和性能。因此，yolact的工作主要是填补这一研究方向的空白，就如`SSD`和`YOLO`填补目标检测那样。

## 问题

- `two-stage`的目标分割方法，通过`feature localization`产生features，然后送进mask predictor分支，这一套下来实质是固有串联，难以加速。

## 方案

- （1）生成不依赖于预测框的non-local `prototype masks`，`prototype masks`是关于整个image的masks.
- （2）针对生成的一系列固定数量的`prototype masks`，预测一组线性组合参数，通过线性参数是得`prototype masks`合成最终的`per instance`

***
## 网络结构
![model](./img/model.PNG)

> ### backbone

- `image_size` : `[550, 550]`
- `ResNet`：`[C1, C2, C3, C4, C5]`
- `feature scale`：`[275, 138, 69, 35, 18]`
- `feature stride`：`[2, 4, 8, 16, 32]`
```

```

> ### necks
- `FPN`
- `lateral` ：`[C3, c4, c5] ---> [P3, P4, P5] ---> [P3, P4, P5] + [P6, P7] `

从P5层基础上附加额外尺度的特征层`[P6, P7]`

- 预测尺度`pred_scale` ：`[24, 48, 96, 192, 384]`
- 预测长宽比`aspect_ratios`：`[1, 1/2, 2]`
- 统一的feature channel：`256`
```
  self.lat_layers  = nn.Conv2d(x, 256, kernel_size=1)
  self.downsample_layers = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
```

> ### Protonet head
![protonet picture](./img/protonet.PNG)  
- base结构: FCN, 全卷积层
- 输入为`P3`，也即backbone FPN features中最深分辨率最高的特征图
- 最后一层输出`k channel`，形成`k`个`prototypes masks` 
- 最后一层protonet的输出需要通过ReLu限制输出的大小  
>> pseudocode  
```
  # cfg :  [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
  nn.Conv2d(x, 256, 3, padding=1) * 3
  InterpolateModule(scale_factor=-2, mode='bilinear', align_corners=False)
  nn.Conv2d(x, 256, 3, padding=1)
  nn.Conv2d(x, 32, 1)
```

> ### branch head
对于prediction head branch，出了预测`classes`，`boxes`，增加一分支用于预测`prototype masks`的`mask coefficients`，也即是针对每一个预测框，网络共产生`4+c+k`个输出,和reinaNet网络相比，yolact使用的prediction head更为`轻量`，因为各个预测任务分支共享了大部分的卷积特征
![HEAD Architecture](./img/head_architecture.PNG)

- mask coefficient head  
假设mask head的预测层的`feature size` 为`H * W * k`，要预测`n`个实例目标，也即是`n * k`，那么可以则有
![矩阵乘法](./img/矩阵乘法.PNG)

> - `tanh`：需要注意的是，为了保证非线性，同时考虑到各个`prototypes masks`特征图可以相加减，最后一层的激活函数考虑为`tanh`，这样可以产生更为稳定的特征 `prototypes`

## LOSSES

- classes：hard negative mining(ohem)
- localization loss: smooth L1
- mask loss: 计算`gt`和`合成masks`图像的像素级别的`BCE(binary cross entropy)`，并通过除以`gt bounding box`的面积归一化

## Fast NMS
Fast NMS 一种能提高速度，却对模型最后性能影响微乎其微的NMS
- `traditional NMS`：对于每一个类别，按照置信度从大到小排序，然后，对于其中的每一个检测，丢掉置信度比其低，但IoU却大于一定阈值的检测。对于30fps而言，花费大于5fps时间的NMS也是一大阻碍。
- `Fast NMS`：整体思路是通过矩阵实现。`traditional NMS`实现上是串行实现的，为了能够并行化，只需要允许已经被removed掉的检测液参与抑制，这在`traditional NMS`是不存在的。
> - 首先，针对类别c，对于每个类别，按置信度从大到小选出n个detections，然后通过向量化的python技巧，可以求出IoU矩阵，因此，获得了IoU矩阵的维度为`c * n * n`；
> - 把IoU矩阵的下三角包括对角线置为0；
> - 求column-wise最大值k，设定t为阈值，满足(k < t)的检测都保留
## Semantic Segmentation Loss
出了上述结构产生的loss，为了进一步促使网络学习到更为丰富的feature，原作者附加了额外的学习任务
- Semantic Segmentation Loss：在`FPN`结构的P3层上扯出一个语义分割分支。由于语义分割的mask是从instance masks生成的，意味着网络并没有严格的学习语义分割
- 实现上需要注意的是：在P3层的基础上，attach一个`kernel=1`的卷积层，卷积出一个不包含背景类别的`c`个channel的`feature Map`。由于单个像素不一定有唯一的类别，故而使用sigmoid作为激活层(`binary_cross_entropy_with_logits`)。

## 文章相关

`FCN`通常用于像素级分类，图像中属于同一类别的像素归为一个类别，这意味着具有平移不变性，然而实例分割需要平移变换，类似于FCIS，MaskRcnn，Mask Scoring R-CNN尝试显示的提供平移变换特征：
- `FCIS`：位置敏感特征图
![FCIS](./img/FCIS.PNG)
- `MaskRcnn`：在roipooling之后的第二阶段附加`mask branch`，因此，自然而然的有了实例位置信息，不需要实例的位置定位。
- `yolact`：yolact的最后阶段，通过预测的box框crop得出最后的`final mask`，这是网络唯一一处体现了平移变换特征，但作者发现yolact的方法，即使没有这最后一步的crop的步骤，对于medium和large的目标，一样能work。如下图所示：
![yolact](./img/yolact.PNG)
- `Mask Quality`：large objects的masks质量高于`mask rcnn`，因为masks的生成直接从原图crop出来，不需要经过一系列repool和对齐变换。
- `帧间稳定性`：即使训练时候是静态图，但相比二阶段目标分割网络，yolact的帧间稳定性更好，原因应该是yolact是one stage网络，而two-stage是高度依赖于一阶段的region proposals。









