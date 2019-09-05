## 网络结构

> ### backbone

- `ResNet`：`[C1, C2, C3, C4, C5]`
- `image_size` : `[550, 550]`
- 
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

- base结构: FCN, 全卷积层  

![protonet picture](./img/protonet.PNG)
code  
```
  nn.Conv2d(x, 256, 3, padding=1) * 3
  InterpolateModule(scale_factor=-2, mode='bilinear', align_corners=False)
  nn.Conv2d(x, 256, 3, padding=1)
  nn.Conv2d(x, 32, 1)
```

> ### branch head
- class head

- box head 

- mask coefficient head
