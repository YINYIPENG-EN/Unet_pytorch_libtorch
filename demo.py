import argparse
from tools.predict import Predict
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, default='vgg16', help='model')
    parse.add_argument('--model_path', type=str, default='./model_data/unet_voc.pth', help='weights path')
    parse.add_argument('--num_classes', type=int, default=21, help='num_classes,include background class')
    parse.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parse.add_argument('--predict', action='store_true', default=False, help='model predict')
    parse.add_argument('--classes_list', type=str, default='', help='predict any classes,input number instead of classes name,eg:15,2')
    parse.add_argument('--blend', action='store_true', default=False, help='Seg image overlaps with original image')
    parse.add_argument('--image', action='store_true', default=False, help='image predict')
    parse.add_argument('--video', action='store_true', default=False, help='video predict')
    parse.add_argument('--video_path', type=str, default='0', help='video path')
    parse.add_argument('--output', type=str, default='', help='output path')
    parse.add_argument('--fps', action='store_true', default=False, help='FPS test')

    '''
    参数说明：
    model：主干网络，暂时只支持VGG16
    model_path:权重路径
    num_classes:类别数量(含背景)，默认21
    cuda:是否用GPU推理
    predict 预测模式
    image:图像预测
    video:视频预测
    video_path:视频路径，默认0
    output:输出路径
    fps:测试FPS
    blend:分割图是否和原图叠加
    classes_list:预测某些类，如果是多个类，用','隔开，例如：15，7
    '''
    opt = parse.parse_args()
    if opt.predict:
        Predict(opt)
