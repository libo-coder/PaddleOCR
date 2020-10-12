# coding=utf-8
"""
数据的预处理：在模型搭建和训练之前，第一步就是对数据进行必要的预处理，将数据处理成适合于 PaddleOCR 的格式
    PaddleOCR 格式： “图像名称 标注” 的列表     样例： train_images/img_0.jpg 福
    1. 切分训练数据为训练集和验证集，方便模型训练，切分比例为 95:5
    2. 缩减字符列表：
        2.1 全角统一为半角
        2.2 英文字符统一为小写
        2.3 中文字符统一为简体
        2.4 忽略所有空格和符号
    3. 经过缩减的字符列表能从原来的 4000 多个字符降低到 3808 个字符
    4. 对字符列表进行缩减有助于模型更加快速的收敛

@author: libo
"""
import os
import random
from langconv import Converter

word_list = []
datas = []

converter = Converter('zh-hans')

def is_chinese(uchar):
    """ 判断一个 unicode 是否是汉字 """
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_english(uchar):
    """ 判断一个 unicode 是否是英文 """
    if uchar >= u'\u0061' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_number(uchar):
    """ 判断一个 unicode 是否是半角数字 """
    if uchar >= u'\u0030' and uchar <= u'\u007a':
        return True
    else:
        return False


def Q2B(uchar):
    """ 单个字符 全角转半角 """
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:      # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


# 读取标注数据
with open('/home/room/data1/libo/paddle_project/dataset/rec_data/train.list', 'r', encoding='UTF-8') as f:
    for line in f:
        # print(line)         # 45 \t 48 \t img_0.jpg \t 福
        name, label = line[:-1].split('\t')[-2:]
        # print('name: %s \nlabel: %s' %(name, label))
        label = label.replace(' ', '')
        label = converter.convert(label)
        label.lower()
        new_label = []
        for word in label:
            word = Q2B(word)
            if is_chinese(word) or is_number(word) or is_english(word):
                new_label.append(word)
                if word not in word_list:
                    word_list.append(word)
        if new_label:
            datas.append('%s\t%s\n' % (os.path.join('train_images', name), ''.join(new_label)))

word_list.sort()

# 生成词表
with open('/home/room/data1/libo/paddle_project/dataset/rec_data/vocab.txt', 'w', encoding='UTF-8') as f:
    for word in word_list:
        f.write(word + '\n')

random.shuffle(datas)
split_num = int(len(datas) * 0.95)

# 分割数据为训练和验证集
with open('/home/room/data1/libo/paddle_project/dataset/rec_data/train.txt', 'w', encoding='UTF-8') as f:
    for line in datas[:split_num]:
        # print('line: ', line)
        f.write(line)

with open('/home/room/data1/libo/paddle_project/dataset/rec_data/val.txt', 'w', encoding='UTF-8') as f:
    for line in datas[split_num:]:
        f.write(line)

