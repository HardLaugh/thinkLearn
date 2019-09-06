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
