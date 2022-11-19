# Week 05

### Task
1.canny detail实现
2.canny 实现
3.透视变换实现

### Result
#### 1. canny detail实现
(1)高斯平滑
<img width=400 src='./rst/Gaussian img.png'>

(2)用sobel检测水平,垂直,对角边缘
<img width=600 src='./rst/Sobel img.png'>

(3)非极大值抑制
<img widht=300 src='./rst/NMS img.png'>

(4)双阈值检测
<img width=250 src='./rst/Canny img.png'>


#### 2. canny 实现

<img width=250 src='./rst/Canny img_auto.png'>

#### 3.透视变换

* 变换矩阵
<img width=400 src='./rst/PerspectiveTransformMatrix.png'><br>

* 原图

<img width=200 src='./PerspectiveTransform_src.jpg'><br>

* 变换结果

<img width=200 src='./rst/PerspectiveTransofrm_dst.png'>
