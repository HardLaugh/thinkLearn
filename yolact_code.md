## source code 相关
作者的源码总体上和`mmdetection`类似
> ### 第一步：转换成coco数据格式
- `coco data fromat`：coco数据格式为`JSON`，总体上分为几个大字段，主要以分割和目标检测为主，分别是：
```
{
   "info": info,
   "licenses": [license],
   "categories": [category],
   "images": [image],
   "annotations": [annotation]
}
```
info, licenses, categories都不是必须的东西，但避免以后忘记这个cocodata是做什么用的，最好加一些相关信息，诸如什么时间，做什么类型训练用的数据
- 相关的coco api介绍：  
> - 
接下来，以大华的养猪数据标注格式说明转换过程：  
- 单个image对应一个`"images"`字段list一个元素，image下的每个annotation对应`"annotations"`字段list的一个元素，不同的`annotation_id`但通过唯一的`image_id`指向同一张图片
```
  # 单个image_info
  create_image_info = {
    "id": image_id,
    "file_name": file_name,
    "width": image_size[0],
    "height": image_size[1],
}
```
```
  # 单个annotation_info
  create_annotation_info = {
    "id": annotation_id,
    "image_id": image_id,
    "category_id": category_info["id"], # label_id
    "iscrowd": is_crowd,
    "area": area.tolist(),
    "bbox": bounding_box.tolist(),
    "segmentation": polygon,
    "width": image_size[1],
    "height": image_size[0],
} 
```
- pseudocode
```
  coco_output = {"info": None, "licenses":None, "catergories":None, "images":[], "annotations":[]}
  image_id = 1
  annotation_id = 1
  for per_image_path, per_anns_path:
      image_info = create_image_info
      coco_output["images"].append(image_info)
      
      for per_anno in per_anns_info:
          annotation_info = create_annotation_info
          annotation_id += 1
          coco_output["annotations"].append(annotation_info)
          
      image_id += 1
```
