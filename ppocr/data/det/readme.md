## detection 说明
支持多种文本检测训练算法：
* EAST
* DB
* SAST

## 检测训练数据集标注格式

标注的文件格式如下：
```bash
"图像文件名 \t json.dumps编码的图像编码信息 "
ch4_test_images/img_61.jpg  [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```

json.dumps编码前的图像标注信息是包含多个字典的 list，字典中的 point 表示文本框的四个点的坐标 (x, y)，从左上角的点开始顺时针排列。transcription 表示当前文本框的文字，当其内容为 '###' 时，表示该文本框无效，在训练时会跳过。

如果想在其他数据集上训练，可以按照上述形式构建标注文件。

## 断电训练
如果训练程序中断，希望加载训练中断的模型从而恢复训练，可以通过指定 Global.checkpoints 指定要加载的模型路径：
```bash
python tools/train.py -c configs/det/det_mv3_db.yml \
        -o Global.checkpoints=./your/trained/model
```
注意：Global.checkpoints 的优先级高于 Global.pretrain_weights 的优先级，即同时指定两个参数时，优先加载 Global.checkpoints 指定的模型，如果 Global.checkpoints 指定的模型路径有误，会加载 Global.pretrain_weights 指定的模型。

## 指标评估
PaddleOCR 计算三个 OCR 检测相关的指标，分别是： Precision、Recall、Hmean。

运行如下代码，根据配置文件 det_db_mv3.yml 中 save_res_path 指定的测试集检测结果文件，计算评估指标。

评估时设置后处理参数 box_thresh=0.6，unclip_ratio=1.5，使用不同数据集、不同模型训练，可调整这两个参数进行优化

```bash
python tools/eval.py -c configs/det/det_mv3_db.yml  \
        -o Global.checkpoints="{path/to/weights}/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```

训练中模型参数默认保存在 Global.save_model_dir 目录下。在评估指标时，需要设置 Global.checkpoints 指向保存的参数文件。

```bash
python tools/eval.py -c configs/det/det_mv3_db.yml  \
        -o Global.checkpoints="./output/det_db/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```

注：`box_thresh`、`unclip_ratio`是 DB 后处理所需要的参数，在评估 EAST 模型时不需要设置

## 测试检测效果

**测试单张图像的检测效果：**
```bash
python tools/infer_det.py -c configs/det/det_mv3_db.yml \
        -o TestReader.infer_img="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy"
```

**测试DB模型时，调整后处理阈值:**
```bash
python tools/infer_det.py -c configs/det/det_mv3_db.yml \
        -o TestReader.infer_img="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```

**测试文件夹下所有图像的检测效果:**
```bash
python tools/infer_det.py -c configs/det/det_mv3_db.yml \
        -o TestReader.infer_img="./doc/imgs_en/" Global.checkpoints="./output/det_db/best_accuracy"
```

