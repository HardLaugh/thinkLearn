# PAnet instance segmentation

- 神经网络中，feature information的传播方式路径极其重要(PAnet)
- low level features对于large instance identification特别有帮助，但从lowlevel 到topmost features之间的路径太长
- information fusion，FCN和FC 两中feature views的特征融合
- fpn的不断堆叠？采用类似resnet方式堆叠fpn module
