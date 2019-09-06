和reinaNet网络相比，yolact使用的prediction head更为`轻量`，因为各个预测任务分支共享了大部分的卷积特征
![HEAD Architecture](./img/head_architecture.PNG)

- mask coefficient head  
假设mask head的预测层的`feature size` 为`H * W * k`，要预测`n`个实例目标，也即是`n * k`，那么可以则有
![矩阵乘法](./img/矩阵乘法.PNG)

> - `tanh`：需要注意的是，为了保证非线性，同时考虑到各个`prototypes masks`特征图可以相加减，最后一层的激活函数考虑为`tanh`，这样可以产生更为稳定的特征 `prototypes`

## LOSSES

- classes：hard negative mining(ohem)
- localization loss: smooth L1
- mask loss: 计算`gt`和`合成masks`图像的像素级别的`BCE(binary cross entropy)`，并通过除以`gt bounding box`的面积归一化

## Fast NMS
Fast NMS 一种能提高速度，却对模型最后性能影响微乎其微的NMS
- `general NMS`：对于每一个类别，按照置信度从大到小排序，然后，对于其中的每一个检测，丢掉置信度比其低，但IoU却大于一定阈值的检测。
- `Fast NMS`：

## 文章相关

`FCN`通常用于像素级分类，图像中属于同一类别的像素归为一个类别，这意味着具有平移不变性，然而实例分割需要平移变换，类似于FCIS，MaskRcnn，Mask Scoring R-CNN尝试显示的提供平移变换特征：
- `FCIS`：位置敏感特征图
![FCIS](./img/FCIS.PNG)
- `MaskRcnn`：在roipooling之后的第二阶段附加`mask branch`，因此，自然而然的有了实例位置信息，不需要实例的位置定位。
- `yolact`：yolact的最后阶段，通过预测的box框crop得出最后的`final mask`，这是网络唯一一处体现了平移变换特征，但作者发现yolact的方法，即使没有这最后一步的crop的步骤，对于medium和large的目标，一样能work。如下图所示：
![yolact](./img/yolact.PNG)








