# OCR

## 什么是OCR
OCR，光学字符识别（Optical Character Recognition），指对图像中的文字进行识别，并获取文本结果。常见于拍照检测、文档识别、证照票据识别、车牌识别、
自然场景文本定位识别等，相关技术在数字时代得到广泛应用。


### 应用场景

- 文本拍照识别
- 证照票据识别
- 车牌识别
- 拍照搜题
- 算术题拍照检查
- 招牌识别
- 内容审核


### 技术难点

| 难点 | 说明 | 案例 |
| :------- | :--------- | :--------- |
| 复杂版式 | 表格、目录文本 | <img src="../../../docs/ocr/table_words.png" width="250" /> |
| 扭曲变形 | 书本翻页、纸张褶皱 | <img src="../../../docs/ocr/niuqu.png" width="250" /> |
| 笔迹干扰、手写、涂改 | 答卷 | <img src="../../../docs/ocr/hand_write.png" width="250" /> |
| 不均匀光照、反光、弱光 | 侧面光照、阴影 | <img src="../../../docs/ocr/fanguang.png" width="250" /> |
| 失焦、运动模糊、摩尔纹 | 运动照片 | <img src="../../../docs/ocr/moer.png" width="250" /> |
| 复杂背景 | 彩色背景 | <img src="../../../docs/ocr/complex_bg.png" width="250" /> |
| 多字体、多语言混排 | 课本封面 | <img src="../../../docs/ocr/fonts_words.png" width="250" /> |
| 角度、弯曲、变形 | logo、图章 | <img src="../../../docs/ocr/angles.png" width="250" /> |



## OCR实现流程

一般流程是：输入 -> 文本检测 -> 文本识别 -> 输出

- 文本检测：解决的问题是哪里有文字，文字的范围有多少
- 文本识别：对定位好的文字区域进行识别，主要解决的问题是每个文字是什么，将图像中的文字区域转化为字符信息

### 模型

#### 文本检测算法
- DB([paper](https://arxiv.org/abs/1911.08947))（推荐）
- EAST([paper](https://arxiv.org/abs/1704.03155))
- SAST([paper](https://arxiv.org/abs/1908.05498))

#### 文本识别算法
- CRNN([paper](https://arxiv.org/abs/1507.05717))（推荐）
- Rosetta([paper](https://arxiv.org/abs/1910.05085))
- STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))
- RARE([paper](https://arxiv.org/abs/1603.03915v1))
- SRN([paper](https://arxiv.org/abs/2003.12294))

## 讨论

#### 基于深度学习的文字检测方法有哪几种？各有什么优缺点？
常用的基于深度学习的文字检测方法一般可以分为基于回归的、基于分割的两大类，当然还有一些将两者进行结合的方法。
1. 基于回归的方法分为box回归和像素值回归。a. 采用box回归的方法主要有CTPN、Textbox系列和EAST，这类算法对规则形状文本检测效果较好，
但无法准确检测不规则形状文本。 b. 像素值回归的方法主要有CRAFT和SA-Text，这类算法能够检测弯曲文本且对小文本效果优秀但是实时性能不够。
2. 基于分割的算法，如PSENet，这类算法不受文本形状的限制，对各种形状的文本都能取得较好的效果，但是往往后处理比较复杂，导致耗时严重。
目前也有一些算法专门针对这个问题进行改进，如DB，将二值化进行近似，使其可导，融入训练，从而获取更准确的边界，大大降低了后处理的耗时。

#### 对于中文行文本识别，CTC和Attention哪种更优？
1. 从效果上来看，通用OCR场景CTC的识别效果优于Attention，因为带识别的字典中的字符比较多，常用中文汉字三千字以上，
如果训练样本不足的情况下，对于这些字符的序列关系挖掘比较困难。中文场景下Attention模型的优势无法体现。而且Attention适合短语句识别，
对长句子识别比较差。
2. 从训练和预测速度上，Attention的串行解码结构限制了预测速度，而CTC网络结构更高效，预测速度上更有优势。




