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
