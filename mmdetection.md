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
2. meshgrid: row_major=True, 假设输入`w_ratios = [0,1,2]`, `h_ratios =[0,1,2]`，则输出
```
  xx = [0, 1, 2, 0, 1, 2, 0, 1, 2]
  yy = [0, 0, 0, 1, 1, 1, 2, 2, 2]
```
根据设置的row_major=True，直观上来说就是一个grid map的按行优先reshape成的list，也即是内存空间上按行优先保存数据（[y, x]），分别为 [0, 0], [0, 1], [0, 2].....[2, 2]  

3. grid_anchors: 利用meshgrid的机制，可以生成把`base_anchors`的坐标shift到各个grid cell的偏移矩阵, 首先有
```
feat_h, feat_w = featmap_size
shift_x = torch.arange(0, feat_w, device=device) * stride
shift_y = torch.arange(0, feat_h, device=device) * stride
```
如上所示，假设，特征图的尺寸为`3*3`，stride=4，则shift_x和shift_y分别为，
```
shift_x = [0, 4, 8]
shift_y = [0, 4, 8]
```
用meshgrid可以生成：
```
shift_xx = [0, 4, 8,    0, 4, 8,    0, 4, 8]
shift_yy = [0, 0, 0,    4, 4, 4,    8, 8, 8]
shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
shifts = [
[0, 0, 0, 0],
[4, 0, 4, 0],
...
]
```
按行优先来说，featMap的第一个grid cell, 坐标的`[y=0, x=0]`anchor shift为`[0, 0]`也即是`base anchors`不移动，接着是grid坐标为`[y= 0, x=1]`，shifts的偏移为`shift_x = 4, shift_y = 0`，也即是`base anchors`的`[xmin, ymin, xmax, ymax]`分别加上`[4, 0, 4, 0]`,直观上来说就是把`base anchors`移动到第二个grid cell单元位置上。  
最后按照broadcast的机制，如下：
```
all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
```
all_anchors在内存上的存放顺序为按行存放，先存放grid坐标为`[y= 0, x=0]`的所有`base anchors`，然后存放坐标`[y= 0, x=1]`的所有`base anchors`..........到最后存放坐标`[y=2, x=2]`的所有anchors。

# bbox_head
bbox_head抽象出来两个head，分别是bbox_head和Anchorhead
bbox_head 是针对two_stage的第二阶段建立的，也就是没有anchor了，而anchorhead是建立在比如rpn，ssh，retina，yolo等经过fpn后的featMap上


# train information flow

## data_loader coco 数据集
> datasets: __getitem__ 返回一个data results 形式如下：
```
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
```
pre_results如下：
```
results = dict(img_info=img_info, ann_info=ann_info)
results['img_prefix'] = self.img_prefix
results['seg_prefix'] = self.seg_prefix
results['proposal_file'] = self.proposal_file
results['bbox_fields'] = []
results['mask_fields'] = []
```
进入pipeline：
- `LoadImageFromFile`：从`preresult`中读取`img_prefix`和`img_info`中的`filename`组合成最终文件的绝对路径`filename`，然后形成如下`dict key`：
```
results['filename'] = filename
results['img'] = img
results['img_shape'] = img.shape
results['ori_shape'] = img.shape
```
- `LoadAnnotations`：根据设置的`with_{bboxes, label, mask, seg}`，从`anno_info`返回各种标注信息
```
results['gt_bboxes'] 
results['gt_labels']
results['gt_masks']
results['gt_semantic_seg']
results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
results['mask_fields'].append('gt_masks')
```
- `DefaultFormatBundle`:
```
- img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
- proposals: (1)to tensor, (2)to DataContainer
- gt_bboxes: (1)to tensor, (2)to DataContainer
- gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
- gt_labels: (1)to tensor, (2)to DataContainer
- gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
- gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                   (3)to DataContainer (stack=True)
```

- `Collect`: 把信息收集起来format化，返回data包含两个部分  
```
dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
```
1. 一个是keys下的东西, 分别是data['img'], data['gt_bboxes'], data['gt_labels']
2. 一个是meta_keys, 保存是图片的相关特征信息
```
meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')):
```
目前为止上述的所有都是针对单张图片，也即是data表示一张图片的相关信息，data总共包含上面的keys设置的关键词和`img_meta`关键词


## dataset 和groupsampler

- `groupsampler`把`image_size`的aspect ratio 大于1的分成一组， 小于1的成一组，共两个group组，然后每个group的图片数量补足成能整除`sampers_per_gpu`的数量，之后，按组shuffle，也即是每个batch的图片要么都是来自第一组，要么都是来自第二组。完成这步关键的参数和函数有: dataset.flag









