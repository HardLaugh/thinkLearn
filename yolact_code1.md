> ### 第二步: 快速上手介绍
- 猪数据集信息构建：按照第一步可以生成一个instances_Animal_train2019.json的文件
```
PIG_CLASSES = ('pig',)
Pig2019_train_dataset = dataset_base.copy({
    'name': 'PIG 2019 Train',
    'class_names': PIG_CLASSES,

    'train_info': './data/pig/annotations/instances_Animal_train2019.json',
    'train_images': './data/pig/images/',


    # 'valid_info': './data/pig/annotations/instances_Animal_train2019.json',
    # 'valid_images': './data/pig/images/',

    'valid_info': './data/pig/annotations/instances_Animal_test2019.json',
    'valid_images': './data/pig/images/',

    'label_map': None

})
```
以上面代码为例，基于dataset_base继承出Pig2019_train_dataset，给出自己的类别信息(`PIG_CLASSES`)，相关的路径：cocoJson路径和存放图片的路径。label_map置为None，表示label_id按PIG_CLASSES给出的顺序排列。

- 训练模型信息构建：
```
yolact_resnet50_pig_config = yolact_base_config.copy({
    'name': 'yolact_resnet50_pig',

    # Dataset stuff
    'dataset': Pig2019_train_dataset,
    'num_classes': len(Pig2019_train_dataset.class_names) + 1,

    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(1, 4)),
        
        'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug
    }),
})
```
以上面代码为例，从config中找到相关的自己想要训练的模型base，继承出上面的`yolact_resnet50_pig_config`模型配置信息

- train.py 文件：需要注意的几个参数：1. 受限于docker的sharedmemory，num_workers必须为0；2. config = yolact_resnet50_pig_config；3. 模型有DataParallel的代码，注意你运行环境的可行性。
运行train.py 基本上就可以跑起来了，最多就是需要调试一下图片路径的信息
