# coding=utf-8
"""
模型预测，指标评估  PaddleOCR计算三个检测相关的指标，分别是: Precision, Recall, Hmean
评估时设置后处理参数 box_thresh, unclip_ratio，使用不同的数据集、不同模型训练，可调整这两个参数进行优化
运行指令：（检测）
    python tools/eval.py -c configs/det/det_mv3_db.yml \
            -o Global.checkpoints="{path/to/weights}/best_accuracy" PostProcess.box_thresh=0.6  PostProcess.unclip_ratio=1.5
    
    训练中模型参数默认保存在 Global.save_model_dir 目录下，在评估指标时，需要设置 Global.checkpoints 指向保存的参数文件
    注： box_thresh、unclip_ratio 是 DB 后处理所需要的参数，在评估 EAST 模型时不需要设置

运行指令：（识别）
    1. 验证评估数据集可以通过 configs/rec/rec_icdar15_train.yml 修改 EvalReader 中的 label_file_path 设置
    2. 注意验证评估时必须确保配置文件中 img_infer 字段为空（在验证评估数据集的时候）
    3. python tools/eval.py -c configs/rec/rec_icdar15_train.yml 
              -o Global.checkpoints={path/to/weights}/best_accuracy

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

import program
from paddle import fluid
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.data.reader_main import reader_main
from ppocr.utils.save_load import init_model
from eval_utils.eval_det_utils import eval_det_run
from eval_utils.eval_rec_utils import test_rec_benchmark
from eval_utils.eval_rec_utils import eval_rec_run


def main():
    startup_prog, eval_program, place, config, train_alg_type = program.preprocess()
    eval_build_outputs = program.build(
        config, eval_program, startup_prog, mode='test')
    eval_fetch_name_list = eval_build_outputs[1]
    eval_fetch_varname_list = eval_build_outputs[2]
    eval_program = eval_program.clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    init_model(config, eval_program, exe)

    if train_alg_type == 'det':
        eval_reader = reader_main(config=config, mode="eval")
        eval_info_dict = {'program':eval_program,\
            'reader':eval_reader,\
            'fetch_name_list':eval_fetch_name_list,\
            'fetch_varname_list':eval_fetch_varname_list}
        metrics = eval_det_run(exe, config, eval_info_dict, "eval")
        logger.info("Eval result: {}".format(metrics))
    else:
        reader_type = config['Global']['reader_yml']
        if "benchmark" not in reader_type:
            eval_reader = reader_main(config=config, mode="eval")
            eval_info_dict = {'program': eval_program, \
                              'reader': eval_reader, \
                              'fetch_name_list': eval_fetch_name_list, \
                              'fetch_varname_list': eval_fetch_varname_list}
            metrics = eval_rec_run(exe, config, eval_info_dict, "eval")
            logger.info("Eval result: {}".format(metrics))
        else:
            eval_info_dict = {'program':eval_program,\
                'fetch_name_list':eval_fetch_name_list,\
                'fetch_varname_list':eval_fetch_varname_list}
            test_rec_benchmark(exe, config, eval_info_dict)


if __name__ == '__main__':
    main()
