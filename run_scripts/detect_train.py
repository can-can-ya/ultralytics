# -*- coding: utf-8 -*-

"""
1. 脚本名称自动解析 task 和 mode，例如：
   detect_train.py -> task=detect, mode=train
2. PROJECT 自动设置为：
   /home/goujiaxiang/code_can/ultralytics/ultralytics/runs/{SCRIPT_NAME}
3. NAME 自动设置为：
   yyyy-mm-dd-hh-mm-ss_模型名_数据集名
4. 支持自定义单 GPU / 多 GPU 联合训练
5. 显式设置随机种子，方便实验复现
6. 可直接配置学习率、优化器和学习率调度策略
7. 通过 ultralytics.cfg.entrypoint() 进入官方 CLI 流程，
   方便后续 PyCharm 调试 Ultralytics 源码
"""

from datetime import datetime
from pathlib import Path

from ultralytics.cfg import entrypoint

if __package__:
    from ._run_logging import log_to_file
else:
    from _run_logging import log_to_file


# ============================================================
#                       可配置参数
# ============================================================

# ------------------------------------------------------------
# 模型与数据集
# ------------------------------------------------------------

MODEL = "/home/goujiaxiang/pretrained_can/yolov8m.pt" # 预训练权重
DATA = "/home/goujiaxiang/data_can/sus_illegal_const_729/dataset.yaml" # 数据集配置文件


# ------------------------------------------------------------
# 基础训练参数
# ------------------------------------------------------------

EPOCHS = 100 # 训练轮数
IMAGE_SIZE = 960 # 输入图片尺寸
BATCH_SIZE = 32 # 所有 GPU 加起来的 Batch Size
WORKERS = 8 # DataLoader worker 数量
PATIENCE = 30 # Early Stopping patience
FREEZE = 10 # 冻结前 10 个模块，主要保留预训练 Backbone 特征，只重点训练后面的 Neck/Head
WEIGHT_DECAY = 0.0005 # 正则化
GPU_DEVICES = [0, 1] # 单 GPU：GPU_DEVICES = [0]；CPU：GPU_DEVICES = "cpu"
SEED = 42 # 随机种子
DETERMINISTIC = True # 是否启用确定性算法
OPTIMIZER = "SGD" # 优化器常见选择："SGD"，"Adam"，"AdamW"
LR0 = 0.001 # 初始学习率，SGD 常见：0.01；Adam / AdamW 常见：0.001。微调小数据集时可以小一点
LRF = 0.01 # 最终学习率比例，最终学习率约为：LR0 * LRF
COS_LR = False # 是否使用 Cosine Learning Rate Scheduler
WARMUP_EPOCHS = 3.0 # Warmup 轮数
SAVE = True # 是否保存模型
AMP = True # 是否使用 AMP 混合精度训练


# ============================================================
#                   自动生成参数
# ============================================================

SCRIPT_PATH = Path(__file__).resolve() # 当前脚本路径
SCRIPT_NAME = SCRIPT_PATH.stem # 当前脚本文件名
PROJECT = f"/home/goujiaxiang/code_can/ultralytics/ultralytics/runs/{SCRIPT_NAME}" # 运行结果保存路径
script_parts = SCRIPT_NAME.split("_", maxsplit=1)
if len(script_parts) != 2:
    raise ValueError(
        f"脚本名称格式错误：{SCRIPT_PATH.name}\n"
        "脚本名称必须满足：task_mode.py\n"
        "例如：detect_train.py"
    )
TASK = script_parts[0] # 任务类型
MODE = script_parts[1] # 任务模式
MODEL_NAME = Path(MODEL).stem # 自动解析模型名称
DATASET_NAME = Path(DATA).parent.name # 自动解析数据集名称
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # 当前时间
NAME = f"{CURRENT_TIME}_{MODEL_NAME}_{DATASET_NAME}" # 自动生成实验名称


# ------------------------------------------------------------
# GPU 参数转换
# ------------------------------------------------------------

if isinstance(GPU_DEVICES, (list, tuple)):
    DEVICE = ",".join(str(device) for device in GPU_DEVICES)
else:
    DEVICE = str(GPU_DEVICES)


# ============================================================
#                       主函数
# ============================================================

def main():
    # 与 exist_ok=True 配合，保证日志和训练结果使用同一目录。
    with log_to_file(Path(PROJECT) / NAME):
        _main()


def _main():

    print("=" * 80)
    print("🚀 Ultralytics")
    print("=" * 80)

    print(f"📄 脚本文件：{SCRIPT_PATH.name}")

    print(f"🎯 TASK：{TASK}")
    print(f"⚙️ MODE：{MODE}")

    print("-" * 80)

    print(f"🤖 MODEL：{MODEL}")
    print(f"📦 DATA：{DATA}")

    print("-" * 80)

    print(f"📁 PROJECT：{PROJECT}")
    print(f"🧪 NAME：{NAME}")

    print("-" * 80)

    print(f"🖥️ GPU：{DEVICE}")

    print(f"🎲 SEED：{SEED}")
    print(f"🔒 DETERMINISTIC：{DETERMINISTIC}")

    print("-" * 80)

    print(f"📉 OPTIMIZER：{OPTIMIZER}")
    print(f"📉 LR0：{LR0}")
    print(f"📉 LRF：{LRF}")
    print(f"📉 COS_LR：{COS_LR}")
    print(f"📉 WARMUP_EPOCHS：{WARMUP_EPOCHS}")

    print("=" * 80)

    args = [
        TASK,
        MODE,
        f"model={MODEL}",
        f"data={DATA}",
        f"epochs={EPOCHS}",
        f"imgsz={IMAGE_SIZE}",
        f"batch={BATCH_SIZE}",
        f"device={DEVICE}",
        f"workers={WORKERS}",
        f"project={PROJECT}",
        f"name={NAME}",
        "exist_ok=True", # 日志会预先创建目录，禁止结果目录自动递增
        f"patience={PATIENCE}",
        f"freeze={FREEZE}",
        f"weight_decay={WEIGHT_DECAY}",
        f"seed={SEED}",
        f"deterministic={DETERMINISTIC}",
        f"optimizer={OPTIMIZER}",
        f"lr0={LR0}",
        f"lrf={LRF}",
        f"cos_lr={COS_LR}",
        f"warmup_epochs={WARMUP_EPOCHS}",
        f"save={SAVE}",
        f"amp={AMP}",
    ]

    command = " ".join(args)

    print("\n📌 实际执行参数：")
    print(f"yolo {command}")

    print("\n" + "=" * 80)
    print("🏃 开始训练")
    print("=" * 80)

    entrypoint(command)


if __name__ == "__main__":
    main()
