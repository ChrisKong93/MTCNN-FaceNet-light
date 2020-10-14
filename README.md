# MTCNN-FaceNet-light

MTCNN-FaceNet-light with c++

只用opencv实现MTCNN和FaceNet

本项目主要是用来学习算法底层的原理，没有用cuda以及其他方式加速，所有速度很慢，对实时性要求很高的朋友就不需要在我这里浪费时间了，如果你是想研究mtcnn和facenet的底层实现，那可以看看我的项目

运行环境：

Windows下 Clion VS OpenCV

Opencv配置环境变量

```
OpenCV_DIR D:/opencv/build/
```

VS无需特殊配置

Clion将解释器设置成VS的

项目导入Clion，reload CMakeLists.txt，run

2019-12-06 修改多处bug，卷积初始化后直接进行卷积

2019-12-24 添加BN层

需要模型文件的可以给我发邮件