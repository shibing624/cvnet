# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath('../data/resnet50_coco_best_v2.0.1.h5')
detector.loadModel()

detections, objects_path = detector.detectObjectsFromImage('../data/lap.png', output_image_path="image3new.jpg",
                                                           extract_detected_objects=True)

for i, each_path in zip(detections, objects_path):
    print(i['name'] + ':' + str(i['percentage_probability']))
    print("Object's image saved in " + each_path)
    print("--------------------------------")
