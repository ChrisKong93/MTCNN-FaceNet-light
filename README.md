# MTCNN-FaceNet-light

MTCNN-FaceNet-light with c++

只用opencv实现MTCNN和FaceNet

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