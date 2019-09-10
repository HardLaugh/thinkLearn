# anchor_head.py

> ## AnchorHead: Anchor-based head

- anchor_scales：[8, 16, 32]，anchor的尺度，和anchor_strides相关，主要是为了形成和strides线性相关的anchor尺度
- anchor_strides：[4, 8, 16, 32, 64]，featMap的步长，也即是相对于原图image_size的采样比例
- anchor_ratios：[0.5, 1.0, 2.0]，anchor的长宽比
- anchor_base_sizes = None，如果为None，则自动生成，自动生成的规则是anchor_scales * anchor_strides，也即是(32, 64, 128, 256, 512)，分别对应每一层featMap的anchorsize

以two_stage为例子，相关cfg如下
```
  rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
```
rpn_head 继承base类AnchorHead的_init_()初始化,而AnchorHead初始化的事情和包含功能如下：

- 针对每一个anchor_base(featMap对应）生成一个AnchorGenerator生成器  
AnchorGenerator的主要思想是用gen_base_anchors生成featMap的单个grid包含的`base_anchors`(其实就是最左上角的grid cell的anchors作为基础anchors)，最后在`grid_anchors`中shift `base_anchors`到所有grid cell，具体有如下几个类函数：  
1. gen_base_anchors：以上述cfg为例，先根据stide定义中心坐标
```
  w = h = stride
  # center坐标
  x_ctr = 0.5 * (w - 1)
  y_ctr = 0.5 * (h - 1)
```
接下来，计算所有可能的anchors的[ws, hs]，通过broadcast机制很容易得到，如下所示, 假设`w_ratios = [a,b,c]`, `h_ratios =[d,e,f]`，`scales = [s]`
```
  ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
  hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
```
扩展`w_ratios`为`[3, 1]`, `scales`为`[1, 1]`，则`w_ratios[:, None] * self.scales[None, :] = [3, 1]，分别为[[a*s,], [b*s,], [c*s,]]`，接着reshape为[3,]，最后通过下述代码，形成`base_anchors`，也即是根据中心坐标计算出anchors的左上角和右下角坐标
```
base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
```







