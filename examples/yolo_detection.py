# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath('../data/yolo-tiny.h5')
detector.loadModel()

detections, objects_path = detector.detectObjectsFromImage('../data/lap.png', output_image_path="lap_yolo_tiny.jpg",
                                                           extract_detected_objects=True)

for i, each_path in zip(detections, objects_path):
    print(i['name'] + ':' + str(i['percentage_probability']))
    print("Object's image saved in " + each_path)
    print("--------------------------------")
