import colorsys
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed
from PIL import Image
from torch import nn
import cv2
from nets.unet import Unet as unet
import matplotlib.pyplot as plt


# np.set_printoptions(threshold=np.inf)

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor', 'void']

# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
# --------------------------------------------#
class Unet(object):
    _defaults = {
        "model_image_size": (512, 512, 3),
        # --------------------------------#
        #   blend参数用于控制是否
        #   让识别结果和原图混合
        # --------------------------------#
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, opt, **kwargs):
        self.__dict__.update(self._defaults)
        self.opt = opt
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        self.net = unet(num_classes=self.opt.num_classes, in_channels=self.model_image_size[-1]).eval()

        state_dict = torch.load(self.opt.model_path)
        self.net.load_state_dict(state_dict)

        if self.opt.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('{} model loaded.'.format(self.opt.model_path))

        if self.opt.num_classes <= 21:  # 给每一类附色
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                          for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, nw, nh

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # ---------------------------------------------------------#
        image = image.convert('RGB')
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]


        # ---------------------------------------------------#
        #   进行不失真的resize，添加灰条，进行图像归一化
        # ---------------------------------------------------#
        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        images = [np.array(image) / 255]  # 归一化处理
        images = np.transpose(images, (0, 3, 1, 2))  # 将(batch_size, w, h, channels)->(batch_size,channels, w,h)

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor)  # numpy-->Tensor
            if self.opt.cuda:
                images = images.cuda()

            pr = self.net(images)[0]  # [0]指的是取出该批次，此刻shape为(num_classes,input_w,input_h)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类 pr.permute(1,2,0)是将Tensor转为(input_w,input_h,channels),dim=-1实际就是channels这个维度
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)  # 放在cpu上进行预测
            # print(pr)
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
                 int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]


        # img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # mask = pr[:, :] == 7
        # pre = pr * mask
        # pre = np.uint8(pre)
        # contours, hierarchy = cv2.findContours(pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img, contours, -1, (255,255,255),3)
        # cv2.imshow("pre", img)
        # if cv2.waitKey(0) & 0xff == ord('q'):
        #     cv2.destroyAllWindows()

        # ------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #   如果只画轮廓线，可以将填充颜色部分注释掉
        # ------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))  # w,h,3
        if self.opt.classes_list:
            any_classes = list(map(int, self.opt.classes_list.split(',')))
            for c in any_classes:
                mask = pr[:, :] == c
                pre = pr[:, :]*mask
                contours, hierarchy = cv2.findContours(np.uint8(pre), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')
                Contours_image = cv2.drawContours(seg_img, contours, -1, (255, 255, 255), 1)

        else:
            for c in range(self.opt.num_classes):  # 每个像素值对应不同的类，元素为0的就是背景类

                mask = pr[:, :] == c
                pre = pr[:, :] * mask
                contours, hierarchy = cv2.findContours(np.uint8(pre), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')
                Contours_image = cv2.drawContours(seg_img, contours, -1, (255, 255, 255), 3)

        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))  # (640, 480)
        # image.save("fenge.jpg")#可保存分割后的图，注意没有叠加到原图上
        # ------------------------------------------------#
        #   将新图片和原图片混合
        # ------------------------------------------------#
        if self.opt.blend:
            image = Image.blend(old_img, image, 0.7)

        return image

    def get_FPS(self, image, test_interval):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        images = [np.array(image) / 255]
        images = np.transpose(images, (0, 3, 1, 2))

        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor)
            if self.opt.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
                 int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
                     int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
