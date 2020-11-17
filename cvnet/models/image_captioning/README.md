### Preprocessing

```shell
python bulid_vocab.py
python resize.py

```
eg:
```
python build_vocab.py --caption_path /home/users/xuming06/workspace/cv/DataSet/COCO/annotations/captions_train2017.json
python resize.py --image_dir /home/users/xuming06/workspace/cv/DataSet/COCO/train2017 --output_dir ./data/resized2014/
```

### Train

```shell
python train.py

```
eg:

```
python train.py --image_dir ./data/resize2017 --caption_path /home/users/xuming06/workspace/cv/DataSet/COCO/annotations/captions_train2017.json
```

### Test

```shell
python sample.py --image="../../dataset/png/animal.png"

```