和reinaNet网络相比，yolact使用的prediction head更为`轻量`，因为各个预测任务分支共享了大部分的卷积特征
![HEAD Architecture](./img/head_architecture.PNG)

- mask coefficient head  
假设mask head的预测层的`feature size` 为`H * W * k`，要预测`n`个实例目标，也即是`n * k`，那么可以则有
![矩阵乘法](./img/矩阵乘法)

> - 需要注意的是，为了保证非线性，同时考虑到各个`prototypes masks`特征图可以相加减，最后一层的激活函数考虑为`tanh`，这样可以产生更为稳定的特征 `prototypes masks`





