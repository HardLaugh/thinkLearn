- yolact的缺点：
> - `localization failure`：图像中某个地方包含多个目标(比如：large目标包含small目标)
> - `leakage`：`final mask`的生成是在`prototypes masks`组合合成后利用预测的box框crop出来的，这其中并没有对crop region区域外的noise进行处理。当框预测准确的时候，`final mask`的效果较好，然而框预测不准确的时候，就会发生crop region区域外noise识别为mask，造成所谓的`leakage`。
