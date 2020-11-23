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

- 复杂版式（如：表格文本、目录文本、横版文本、竖版文本）
[table](../../../docs/ocr/table_words.png)
- 扭曲变形（如：书本翻页、纸张褶皱）
[book](../../../docs/ocr/niuqu.png)
- 笔迹干扰、手写、涂改（如：答卷）
[handle_write](../../../docs/ocr/hand_write.png)
- 不均匀光照、反光、弱光
[fanguang](../../../docs/ocr/fanguang.png)
- 失焦、运动模糊、摩尔纹
[moer](../../../docs/ocr/moer.png)
- 复杂背景
[bg](../../../docs/ocr/complex_bg.png)
- 多字体、多语言混排
[fonts](../../../docs/ocr/fonts_words.png)
- 角度、弯曲、变形
[angle](../../../docs/ocr/angle.png)


## OCR实现流程

CRNN
- CTC


- 通用文本识别


- 车牌检测和识别


