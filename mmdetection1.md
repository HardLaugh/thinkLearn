2. meshgrid: row_major=True, 假设输入`w_ratios = [0,1,2]`, `h_ratios =[0,1,2]`，则输出
```
  xx = [0, 1, 2, 0, 1, 2, 0, 1, 2]
  yy = [0, 0, 0, 1, 1, 1, 2, 2, 2]
```
根据设置的row_major=True，直观上来说就是一个grid map的按行优先reshape成的list，也即是内存空间上按行优先保存数据（[y, x]），分别为 [0, 0], [0, 1], [0, 2].....[2, 2]  

3. grid_anchors: 利用meshgrid的机制，可以生成把`base_anchors`的坐标shift到各个grid cell的偏移矩阵, 首先有
```
feat_h, feat_w = featmap_size
shift_x = torch.arange(0, feat_w, device=device) * stride
shift_y = torch.arange(0, feat_h, device=device) * stride
```
如上所示，假设，特征图的尺寸为`3*3`，stride=4，则shift_x和shift_y分别为，
```
shift_x = [0, 4, 8]
shift_y = [0, 4, 8]
```
用meshgrid可以生成：
```
shift_xx = [0, 4, 8,    0, 4, 8,    0, 4, 8]
shift_yy = [0, 0, 0,    4, 4, 4,    8, 8, 8]
shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
shifts = [
[0, 0, 0, 0],
[4, 0, 4, 0],
...
]
```
按行优先来说，featMap的第一个grid cell, 坐标的`[y=0, x=0]`anchor shift为`[0, 0]`也即是`base anchors`不移动，接着是grid坐标为`[y= 0, x=1]`，shifts的偏移为`shift_x = 4, shift_y = 0`，也即是`base anchors`的`[xmin, ymin, xmax, ymax]`分别加上`[4, 0, 4, 0]`,直观上来说就是把`base anchors`移动到第二个grid cell单元位置上。  
最后按照broadcast的机制，如下：
```
all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
```
all_anchors在内存上的存放顺序为按行存放，先存放grid坐标为`[y= 0, x=0]`的所有`base anchors`，然后存放坐标`[y= 0, x=1]`的所有`base anchors`..........到最后存放坐标`[y=2, x=2]`的所有anchors。

## 未完待续
