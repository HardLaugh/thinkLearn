- yolact的缺点：
> - `localization failure`：图像中某个地方包含多个目标(比如：large目标包含small目标)
> - `leakage`：`final mask`的生成是在`prototypes masks`组合合成后利用预测的box框crop出来的，这其中并没有对crop region区域外的noise进行处理。当框预测准确的时候，`final mask`的效果较好，然而框预测不准确的时候，就会发生crop region区域外noise识别为mask，造成所谓的`leakage`。

## coding 相关

### localization 相关回归公式
主要是中心坐标的回归公式
> - 第一种：RCNN系列和SSD，中心坐标的回归公式如下，主要回归的是gt框和anchors的offset。对于中心坐标而言，按这种公式回归，并没有对中心坐标的范围做限制，因而训练中的离群点会使得模型训练波动，特别是前期。为此，fast rcnn系列提出了`smooth L1`对loss的计算进行了限制；
> - 第二种：Yolov2 regressor：yolo系列的框回归公式，是基于grid网格划分特征图，而中心点坐标都是相对于与所属grid的位置进行回归。  
直接回归坐标，yolov2给过一个解释：模型不稳定，特别是早期，故而v1到v3都沿用了这一方式，由于采用这种方式训练回归的参数被限制到了`[0, 1]`，因而yolo系列的模型训练只采用了`mean squared error(MSE)`。
> - yolact开源代码里面集成了上面两种方式，默认为采用第一种方式。
