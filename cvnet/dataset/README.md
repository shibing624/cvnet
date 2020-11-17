### Data Format for Object Detection

MS COCO data:


2014 version:

```
mkdir data
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./coco2014/
wget http://images.cocodataset.org/zips/train2014.zip -P ./coco2014/
wget http://images.cocodataset.org/zips/val2014.zip -P ./coco2014/

unzip ./coco2014/captions_train-val2014.zip -d ./coco2014/
unzip ./coco2014/train2014.zip -d ./coco2014/
unzip ./coco2014/val2014.zip -d ./coco2014/

```

2017 version:
```
wget -c http://images.cocodataset.org/zips/train2017.zip -P ./coco2017/
wget -c http://images.cocodataset.org/zips/val2017.zip -P ./coco2017/
wget -c http://images.cocodataset.org/zips/test2017.zip -P ./coco2017/
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./coco2017/
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip -P ./coco2017/
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip -P ./coco2017/
unzip ./coco2017/train2017.zip -d ./coco2017/
unzip ./coco2017/val2017.zip -d ./coco2017/
unzip ./coco2017/test2017.zip -d ./coco2017/
unzip ./coco2017/annotations_trainval2017.zip -d ./coco2017/
unzip ./coco2017/stuff_annotations_trainval2017.zip -d ./coco2017/
unzip ./coco2017/image_info_test2017.zip -d ./coco2017/

```
