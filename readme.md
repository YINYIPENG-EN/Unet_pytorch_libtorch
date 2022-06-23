# 经典图像分割网络：Unet 支持libtorch部署推理



**说明：**

支持python与Libtorch C++推理

python版本支持支持对于单类别检测，C++暂不支持

python板支持视频检测，C++暂不支持(仅图像)

增加网络可视化工具

增加pth转onnx格式

增加pth转pt格式



**环境**

windows 10

pytorch:1.7.0(低版本应该也可以)

libtorch 1.7 Debug版

cuda 10.2

VS 2017

英伟达 1650 4G



数据集制作工具：labelme

```
pip install labelme==3.16.7
```



# 训练 

然后进入json_to_dataset.py，修改classes，加入自己的类，注意！不要把_background_这个类删掉！！

运行以后，程序会将原始图copy到datasets/JPEGImags下，然后生成的png标签文件生成在datasets/SegmentationClass文件下。接下来就是这两个文件复制到VOCdevkit/VOC2007/中。

接下来是运行VOCdevkit/voc2unet.py，将会在ImageSets/Segmentation/下生成txt文件。

接下来就可以运行train.py进行训练了，这里需要主要更改 NUM_CLASSES 。

训练的权重会保存在logs下。



# 预测

说明：本项目可以对所有类进行检测并分割，同时也支持**单独某个类**进行分割。

网络采用VGG16为backbone。在终端输入命令：



可以对图像进行预测：

```
python demo.py --predict --image
```



如果你想和原图进行叠加，在命令行输入：

```
python demo.py --predict --image --blend
```



视频预测：

```
python demo.py --predict --video --video_path 0
```



 预测几个类时，用逗号','隔开：

```
python demo.py --predict --image --classes_list 15,7
```



# libtorch 推理

libtorch环境配置和一些遇到的问题可以参考我另一篇文章，这里不再说：

[https://blog.csdn.net/z240626191s/article/details/124341346]

进入tools文件，在pth2pt.py中修改权重路径，num_classes，还有输入大小(默认512).运行以后会保存.pt权重文件

将pt权重文件放在你想放的地方，我这里是放在了与我exe执行程序同级目录下。

打开通过VS 2017打开Libtorch_unet/Unet/Unet.sln，注意修改以下地方：(VS 配置libtorch看上面链接)

在main.cpp中最上面**修改两个宏定义**，一个是网络输入大小，一个是num_classes根据自己的需要修改。

COLOR Classes是我写的一个结构体，每个类对应的颜色，如果你自己的数据集小于21个类，那你不用修改，只需要记住哪个类对应哪个颜色即可。**如果是大于21个类，需要自己在定义颜色**。

在main.cpp torch::jit::load()修改自己的pt权重路径(如果你没和exe放一个目录中，建议填写绝对路径)，当然，如果你希望通过传参的方式也可以，自己修改下即可。

argv[1]是图像路径(执行exe时可以传入)。

然后将项目重新生成，用cmd执行Unet.exe 接着输入图像路径，如下：

```
Unet.exe street.jpg
```



```
*****************************************
**        libtorch Unet图像分割项目    **
**          支持GPU和CPU推理           **
** 生成项目后执行exe并输入图像路径即可   **
**           作者：yinyipeng           **
**           联系方式：                **
**      微信：y24065939s               **
**      邮箱：15930920977@163.com      **
*****************************************

The model load success!
The cuda is available
cuda
put model into the cuda
The output shape is: [1, 21, 512, 512]
seq_img shape is [512, 512, 3]
```



**一些注意事项**

在libtorch推理中需要用到的一些代码，比如Mat转tensor,tensor转Mat等。

**Mat转tensor**

input是经过resize和转RGB的输入图像，转的shape(1,512,512,3)

```
torch::Tensor tensor_image = torch::from_blob(input.data, { 1,input.rows, input.cols,3 }, torch::kByte);
```



**推理：**

在实际验证中，如果在送入模型之前用tensor_image.to(device)即将张量放入cuda，在下面cuda推理中会报关于内存的错误，但在cpu下不会，感觉是libtorch的一个bug吧，但如果在forward函数中将tensor_image放入cuda就可以正常推理。这点需要注意。

```
output = module.forward({tensor_image.to(device)}).toTensor(); //The shape is [batch_size, num_classes, 512,512]
```



**C++中张量的切片：**

指的是对最后一个维度的第0维度进行操作

```
seg_img.index({ "...", 0 })
```



**CUDA FLAOT32-->CUDA UINT8转CPU UINT8(GPU->CPU数据转换)**

在cuda 32 float转cuda UINT 8再转cpu uint8时(因为最后需要CPU进行推理计算数据)，也发现了一个问题，如果你在cuda上转uint8，然后用to(torch::kCPU)后，发现最终显示结果全黑，没有结果，但打印seg_img是有值的，后来打印了一下res这个矩阵，发现里面像素值全为0，且值为**cpu float 32,但我要的是uint8，明明我前面转过了。**即没有tensor数据没有拷贝到Mat中，解决方法是先将cuda放在cpu上，在转uint8，而不是在cuda上转uint8后再迁移到cpu。

```
//在放入CPU的时候，必须要转uint8型，否则后面无法将tensor拷贝至Mat
seg_img = seg_img.to(torch::kCPU).to(torch::kUInt8); 
```



**tensor转Mat**

```
cv::Mat res(cv::Size(input_shape, input_shape), CV_8UC3,seg_img.data_ptr());
```



其他细节：



权重链接：

链接：https://pan.baidu.com/s/1RHYiG1nph1XZ8poxNtepXg 
提取码：yypn