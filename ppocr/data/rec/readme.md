## recognition 说明
支持多种文本识别训练算法：
* Rosetta
* CRNN
* STAR-Net
* RARE
* SRN

### 一、数据准备
PaddleOCR 支持两种数据格式: `lmdb` 用于训练公开数据，调试算法; `通用数据` 训练自己的数据:

**设置数据集：**

训练数据的默认存储路径是 PaddleOCR/train_data, 如果磁盘上已有数据集，只需创建软链接至数据集目录：
```bash
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/dataset
```

**训练自己的数据集：**

1. 首先将训练图片放入同一个文件夹（train_images），并用一个txt文件（rec_gt_train.txt）记录图片路径和标签。将图片路径和图片标签用 `\t` 分割。

    ```bash
    " 图像文件名                 图像标注信息 "
    train_data/train_0001.jpg   简单可依赖
    train_data/train_0002.jpg   用科技让复杂的世界更简单
    ```

2. 同训练集类似，测试集也需要提供一个包含所有图片的文件夹（test）和一个 rec_gt_test.txt

3. 字典：最后需要提供一个字典（{word_dict_name}.txt），使模型在训练时，可以将所有出现的字符映射为字典的索引。因此字典需要包含所有希望被正确识别的字符，{word_dict_name}.txt需要写成如下格式，并以 utf-8 编码格式保存：

    ```bash
    l
    d
    a
    d
    r
    n
    ```
    word_dict.txt 每行有一个单字，将字符与数字索引映射在一起，“and” 将被映射成 [2 5 1]

4. 自定义字典：如需自定义 dic 文件，在 configs/rec/rec_icdar15_train.yml 中添加 character_dict_path 字段, 指向对应的字典路径。 并将 character_type 设置为 ch。

5. 添加空格类别：如果希望支持识别 ”空格” 类别, 将 yml 文件中的 use_space_char 字段设置为 true。
    
    注意：use_space_char 仅在 character_type=ch 时生效


### 二、启动训练
1. 首先下载 pretrain model，可以下载训练好的模型在 icdar2015 数据上进行 finetune

2. 开始训练:

    如果安装的是 cpu 版本，将配置文件中的 use_gpu 字段修改为 false

    ```bash
    # 设置PYTHONPATH路径
    export PYTHONPATH=$PYTHONPATH:.
    # GPU训练 支持单卡，多卡训练，通过CUDA_VISIBLE_DEVICES指定卡号
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    # 训练icdar15英文数据
    python tools/train.py -c configs/rec/rec_icdar15_train.yml
    ```

3. 数据增强：具体见 img_tools.py

4. 训练：

    PaddleOCR 支持训练和评估交替进行, 可以在 configs/rec/rec_icdar15_train.yml 中修改 eval_batch_step 设置评估频率，默认每 500 个 iter 评估一次。评估过程中默认将最佳 acc 模型，保存为 output/rec_CRNN/best_accuracy 。

    如果验证集很大，测试将会比较耗时，可以减少评估次数，或训练完再进行评估。

    注意：预测/评估时的配置文件请务必与训练一致。

5. 评估：

    评估数据集可以通过 configs/rec/rec_icdar15_reader.yml 修改 EvalReader 中的 label_file_path 设置。

    注意：评估时必须确保配置文件中 infer_img 字段为空

    ```bash
    export CUDA_VISIBLE_DEVICES=0
    # GPU 评估， Global.checkpoints 为待测权重
    python tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy
    ```

6. 预测：

    默认预测图片存储在 infer_img 里，通过 -o Global.checkpoints 指定权重：
    ```bash
    # 预测英文结果
    python tools/infer_rec.py -c configs/rec/rec_icdar15_train.yml \
            -o Global.checkpoints={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
    ```

    ```bash
    # 预测中文结果
    python tools/infer_rec.py -c configs/rec/rec_chinese_lite_train.yml \
            -o Global.checkpoints={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/ch/word_1.jpg
    ```