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
- `LoadAnnotations`：根据设置的`with_{bboxes, label, mask, seg}`，从`anno_info`返回

