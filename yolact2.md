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
- `traditional NMS`：对于每一个类别，按照置信度从大到小排序，然后，对于其中的每一个检测，丢掉置信度比其低，但IoU却大于一定阈值的检测。对于30fps而言，花费大于5fps时间的NMS也是一大阻碍。
- `Fast NMS`：整体思路是通过矩阵实现。`traditional NMS`实现上是串行实现的，为了能够并行化，只需要允许已经被removed掉的检测液参与抑制，这在`traditional NMS`是不存在的。
> - 首先，针对类别c，对于每个类别，按置信度从大到小选出n个detections，然后通过向量化的python技巧，可以求出IoU矩阵，因此，获得了IoU矩阵的维度为`c * n * n`；
> - 把IoU矩阵的下三角包括对角线置为0；
> - 求column-wise最大值k，设定t为阈值，满足(k < t)的检测都保留








