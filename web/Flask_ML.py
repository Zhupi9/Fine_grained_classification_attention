#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os,base64
from datetime import timedelta

import torch
import torchvision
import numpy as np
from PIL import Image
from flask import Flask, redirect, render_template, jsonify, request
from werkzeug.utils import secure_filename


class BCNN(torch.nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features = torch.nn.Sequential(*list(self.features.children())
        [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512 ** 2, 200)

        # Freeze all previous layers.
        for param in self.features.parameters():
            param.requires_grad = False
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant(self.fc.bias.data, val=0)

    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 28 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28 ** 2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 200)
        return X


#加载训练模型
model = BCNN()
model = torch.nn.DataParallel(model).cuda()  # 因为训练时使用多个GPU，导致state_dict中key多了一个module
pthfile = torch.load('../model/vgg_16_epoch_38.pth')  # 加载训练好的模型
model.load_state_dict(pthfile)
#获取类别的编号与其对应的名称
class_list=open('../data/cub200/raw/CUB_200_2011/classes.txt',"r").readlines()

def get_name(order):
    line=class_list[order]
    str=line.split(' ')[1]
    return str[4:]

def transform_image(image_byte):  # 转换输入的图像
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.CenterCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])
    image = Image.open(io.BytesIO(image_byte))
    return test_transforms(image).unsqueeze(0)


def predict(image):
    '''
    image = open(
        '../data/cub200/raw/CUB_200_2011/images/010.Red_winged_Blackbird/Red_Winged_Blackbird_0005_5636.jpg',
        'rb').read()
    '''
    transformed_image = transform_image(image)
    outputs = model(transformed_image)
    _, category = torch.max(outputs.data, 1)

    return ('%d' % category[0])

app = Flask(__name__)

app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/upload', methods=['POST', 'GET'])
def upload():

    if request.method == 'POST':
        f = request.files['file']

        base_path=os.path.dirname(__file__)
        save_path=os.path.join(base_path,'static/images/',secure_filename(f.filename))
        test_path=os.path.join('../static/images/',secure_filename(f.filename))
        f.save(save_path)
        image=open(save_path,'rb').read()
        cate_int=predict(image)
        return jsonify({'img_path':test_path,'cate':get_name(int(cate_int))})
        # return render_template('Flask_ML_OK.html',category=cate_int)


@app.route('/')
def show():
    return render_template('Flask_ML.html')

if __name__ == '__main__':
    app.run()
