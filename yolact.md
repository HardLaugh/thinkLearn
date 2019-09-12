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
对于prediction head branch，出了预测`classes`，`boxes`，增加一分支用于预测`prototype masks`的`mask coefficients`，也即是针对每一个预测框，网络共产生`4+c+k`个输出
