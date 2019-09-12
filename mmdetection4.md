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
