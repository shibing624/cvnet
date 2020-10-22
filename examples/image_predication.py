# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from imageai.Prediction import ImagePrediction

import os

pwd_path = os.getcwd()

print(pwd_path)

predict = ImagePrediction()
predict.setModelTypeAsResNet()
predict.setModelPath(os.path.join(pwd_path, '../data', 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
predict.loadModel()

predictions, probs = predict.predictImage(os.path.join(pwd_path, '../data', 'red_car.png'), result_count=5)
for p, prob in zip(predictions, probs):
    print(p + ':' + str(prob))
