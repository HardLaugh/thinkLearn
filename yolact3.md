## 文章相关

`FCN`通常用于像素级分类，图像中属于同一类别的像素归为一个类别，这意味着具有平移不变性，然而实例分割需要平移变换，类似于FCIS，MaskRcnn，Mask Scoring R-CNN尝试显示的提供平移变换特征：
- `FCIS`：位置敏感特征图
![FCIS](./img/FCIS.PNG)
- `MaskRcnn`：在roipooling之后的第二阶段附加`mask branch`，因此，自然而然的有了实例位置信息，不需要实例的位置定位。
- `yolact`：yolact的最后阶段，通过预测的box框crop得出最后的`final mask`，这是网络唯一一处体现了平移变换特征，但作者发现yolact的方法，即使没有这最后一步的crop的步骤，对于medium和large的目标，一样能work。如下图所示：
![yolact](./img/yolact.PNG)


## coding 相关

### localization 相关回归公式
主要是中心坐标的回归公式
> - 第一种：RCNN系列和SSD，中心坐标的回归公式如下，主要回归的是gt框和anchors的offset。对于中心坐标而言，按这种公式回归，并没有对中心坐标的范围做限制，因而训练中的离群点会使得模型训练波动，特别是前期。为此，fast rcnn系列提出了`smooth L1`对loss的计算进行了限制；
> - 第二种：Yolov2 regressor：yolo系列的框回归公式，是基于grid网格划分特征图，而中心点坐标都是相对于与所属grid的位置进行回归。  
直接回归坐标，yolov2给过一个解释：模型不稳定，特别是早期，故而v1到v3都沿用了这一方式，由于采用这种方式训练回归的参数被限制到了`[0, 1]`，因而yolo系列的模型训练只采用了`mean squared error(MSE)`。
> - yolact开源代码里面集成了上面两种方式，默认为采用第一种方式，基于什么考量暂未知，估计只是试试。
