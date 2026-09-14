# -*- coding: utf-8 -*-

"""
特点：
1. 脚本名称自动解析 task 和 mode，例如：
   detect_predict.py -> task=detect, mode=predict
2. PROJECT 自动设置为：
   /home/goujiaxiang/code_can/ultralytics/ultralytics/runs/{SCRIPT_NAME}
3. NAME 自动设置为：
   yyyy-mm-dd-hh-mm-ss_(模型路径中的训练时间, temp)-模型文件名_SOURCE名称
4. 支持 CPU、单 GPU；当指定多张 GPU 时，对 SOURCE 目录中的图片/视频按文件进行并行分片预测
5. 可自定义总 Batch Size、WORKERS、置信度、IoU、视频步长等参数
6. SOURCE 支持图片、视频以及图片/视频混合目录
7. 可选地对 .pt 模型使用自定义类别显示名称（英文或中文）
8. 通过 ultralytics.cfg.entrypoint() 进入官方 CLI 流程，
   方便后续 PyCharm 调试 Ultralytics 源码

注意：
- Ultralytics 原生 predict 并不会像 train 一样通过 device=0,1 自动启动 DDP。
  本脚本的“多 GPU”采用文件级并行：把 SOURCE 目录中的文件分给多张 GPU 分别预测。
- 单个视频文件无法通过这种方式拆给多张 GPU；此时请使用单 GPU，
  或提前把视频切分成多个视频文件后放入同一 SOURCE 目录。
"""

from datetime import datetime
from pathlib import Path
import multiprocessing as mp
import re
import shutil
import tempfile

from ultralytics.cfg import entrypoint

if __package__:
    from ._run_logging import log_to_file
else:
    from _run_logging import log_to_file


# ============================================================
#                       可配置参数
# ============================================================

# ------------------------------------------------------------
# 模型与待预测数据
# ------------------------------------------------------------

MODEL = (
    "/home/goujiaxiang/code_can/ultralytics/ultralytics/runs/detect_train/"
    "2026-09-15-01-24-37_yolov8m_sus_illegal_const_729/"
    "weights/best.pt"
)  # 待预测模型，可使用 best.pt、last.pt 等

SOURCE = (
    "/home/goujiaxiang/data_can/sus_illegal_const_test"
)  # 可为图片、视频，或包含图片/视频的目录


# ------------------------------------------------------------
# 基础预测参数
# ------------------------------------------------------------

IMAGE_SIZE = 960  # 推理输入尺寸，建议与训练尺寸保持一致
BATCH_SIZE = 32  # 所有 GPU 加起来的 Batch Size
WORKERS = 8  # worker 数量；标准 predict 场景下实际作用取决于数据源/加载器实现

CONF = 0.25  # 置信度阈值
IOU = 0.7  # NMS IoU 阈值
MAX_DET = 300  # 每张图片最多保留的检测目标数量
VID_STRIDE = 1  # 视频抽帧步长：1=每帧都预测


# ------------------------------------------------------------
# GPU / CPU 配置
# ------------------------------------------------------------

# 单 GPU：
# GPU_DEVICES = [0]
#
# 多 GPU：
# GPU_DEVICES = [0, 1]
#
# CPU：
# GPU_DEVICES = "cpu"
GPU_DEVICES = [0, 1]


# ------------------------------------------------------------
# 预测结果保存配置
# ------------------------------------------------------------

SAVE = True  # 保存画框后的图片/视频
SAVE_TXT = True  # 保存 YOLO 格式预测标签
SAVE_CONF = True  # SAVE_TXT=True 时同时保存置信度
SAVE_CROP = False  # 是否保存检测目标裁剪图

SHOW_LABELS = True  # 是否显示类别名称
SHOW_CONF = True  # 是否显示置信度


# ------------------------------------------------------------
# 自定义类别显示名称
# ------------------------------------------------------------

# None：
#   使用模型权重中自带的类别名称
#
# 英文示例：
# CLASS_NAMES = {
#     0: "sus_illegal_const",
# }
#
# 中文示例：
# CLASS_NAMES = {
#     0: "疑似违建",
# }
#
# 注意：
# 1. 当前实现仅支持 .pt 模型；
# 2. 只生成临时模型副本，不修改原始 MODEL；
# 3. 类别数量必须与模型一致；
# 4. 中文显示依赖服务器上的中文字体。
CLASS_NAMES = None


# ============================================================
#                   自动生成参数
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()  # 当前脚本路径
SCRIPT_NAME = SCRIPT_PATH.stem  # 当前脚本文件名（不含 .py）

PROJECT = (
    f"/home/goujiaxiang/code_can/ultralytics/ultralytics/runs/{SCRIPT_NAME}"
)

script_parts = SCRIPT_NAME.split("_", maxsplit=1)
if len(script_parts) != 2:
    raise ValueError(
        f"脚本名称格式错误：{SCRIPT_PATH.name}\n"
        "脚本名称必须满足：task_mode.py\n"
        "例如：detect_predict.py"
    )

TASK = script_parts[0]  # 任务类型，例如 detect
MODE = script_parts[1]  # 任务模式，例如 predict

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

MODEL_TIME_MATCHES = re.findall(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}",
    str(MODEL),
)
MODEL_TIME = MODEL_TIME_MATCHES[-1] if MODEL_TIME_MATCHES else "temp"

MODEL_NAME = Path(MODEL).stem  # 例如 best、last

source_path = Path(SOURCE)
SOURCE_NAME = (
    source_path.name
    if source_path.is_dir()
    else source_path.stem
)

NAME = f"{CURRENT_TIME}_{MODEL_TIME}-{MODEL_NAME}_{SOURCE_NAME}"


# ============================================================
#                       工具函数
# ============================================================

IMAGE_SUFFIXES = {
    ".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png",
    ".tif", ".tiff", ".webp", ".pfm", ".heic",
}

VIDEO_SUFFIXES = {
    ".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov",
    ".mp4", ".mpeg", ".mpg", ".ts", ".wmv", ".webm",
}


def build_command(
    model: str,
    source: str,
    device: str,
    batch_size: int,
    name: str,
) -> str:
    """构造 Ultralytics CLI 参数字符串。"""

    args = [
        TASK,
        MODE,
        f"model={model}",
        f"source={source}",
        f"imgsz={IMAGE_SIZE}",
        f"batch={batch_size}",
        f"device={device}",
        f"workers={WORKERS}",
        f"conf={CONF}",
        f"iou={IOU}",
        f"max_det={MAX_DET}",
        f"vid_stride={VID_STRIDE}",
        f"project={PROJECT}",
        f"name={name}",
        "exist_ok=True",  # 日志会预先创建目录，禁止结果目录自动递增
        f"save={SAVE}",
        f"save_txt={SAVE_TXT}",
        f"save_conf={SAVE_CONF}",
        f"save_crop={SAVE_CROP}",
        f"show_labels={SHOW_LABELS}",
        f"show_conf={SHOW_CONF}",
    ]

    return " ".join(args)


def prepare_model_with_custom_names(temp_root: Path) -> str:
    """
    CLASS_NAMES 不为空时，生成临时 .pt 权重副本，
    仅修改类别显示名称，不修改模型参数。
    """
    if CLASS_NAMES is None:
        return MODEL

    model_path = Path(MODEL)

    if model_path.suffix.lower() != ".pt":
        raise ValueError(
            "CLASS_NAMES 自定义显示名称当前仅支持 .pt 模型。"
        )

    import torch

    ckpt = torch.load(str(model_path), map_location="cpu")

    model_objects = []

    if isinstance(ckpt, dict):
        for key in ("model", "ema"):
            obj = ckpt.get(key)
            if obj is not None and hasattr(obj, "names"):
                model_objects.append(obj)
    elif hasattr(ckpt, "names"):
        model_objects.append(ckpt)

    if not model_objects:
        raise RuntimeError(
            "无法从当前 .pt 权重中找到模型类别 names。"
        )

    class_count = len(model_objects[0].names)
    expected_ids = set(range(class_count))

    if set(CLASS_NAMES.keys()) != expected_ids:
        raise ValueError(
            f"CLASS_NAMES 必须完整覆盖 0~{class_count - 1}。\n"
            f"当前配置：{CLASS_NAMES}"
        )

    for obj in model_objects:
        obj.names = dict(CLASS_NAMES)

    temp_model = temp_root / f"{model_path.stem}_custom_names.pt"
    torch.save(ckpt, temp_model)

    return str(temp_model)


def collect_source_files(source_dir: Path):
    """收集 SOURCE 目录中的图片和视频文件。"""
    return [
        p
        for p in sorted(source_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in (IMAGE_SUFFIXES | VIDEO_SUFFIXES)
    ]


def create_worker_source_dir(files, worker_dir: Path):
    """为多 GPU worker 创建临时数据目录。"""

    worker_dir.mkdir(parents=True, exist_ok=True)

    for index, src in enumerate(files, start=1):
        link_name = f"{index:06d}_{src.name}"
        link_path = worker_dir / link_name

        try:
            link_path.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, link_path)


def run_predict_worker(
    model: str,
    source: str,
    gpu_id: int,
    batch_size: int,
    worker_name: str,
):
    """单个 GPU 子进程执行预测，日志与该 GPU 的结果放在一起。"""
    # worker 自己记录 Python 日志，不使用需要等待 EOF 的底层日志管道。
    with log_to_file(Path(PROJECT) / worker_name, capture_subprocess=False):
        _run_predict_worker(model, source, gpu_id, batch_size, worker_name)


def _run_predict_worker(model, source, gpu_id, batch_size, worker_name):
    command = build_command(
        model=model,
        source=source,
        device=str(gpu_id),
        batch_size=batch_size,
        name=worker_name,
    )

    print("\n" + "=" * 80)
    print(f"🚀 GPU {gpu_id} 开始预测")
    print("=" * 80)
    print(f"📌 实际执行参数：\nyolo {command}")
    print("=" * 80)

    entrypoint(command)


# ============================================================
#                       主函数
# ============================================================

def main():
    # 多 GPU 的每个 worker 独立保存日志，主进程无需捕获子进程文件描述符。
    with log_to_file(Path(PROJECT) / NAME, capture_subprocess=False):
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
    print(f"📦 SOURCE：{SOURCE}")

    print("-" * 80)

    print(f"📁 PROJECT：{PROJECT}")
    print(f"🧪 NAME：{NAME}")

    print("-" * 80)

    print(f"📐 IMAGE_SIZE：{IMAGE_SIZE}")
    print(f"📦 BATCH_SIZE（总）：{BATCH_SIZE}")
    print(f"👷 WORKERS：{WORKERS}")
    print(f"🎯 CONF：{CONF}")
    print(f"🔗 IOU：{IOU}")

    print("-" * 80)

    print(f"🖥️ GPU / CPU：{GPU_DEVICES}")
    print(f"🏷️ CLASS_NAMES：{CLASS_NAMES}")

    print("=" * 80)

    source_path = Path(SOURCE)

    if not source_path.exists():
        raise FileNotFoundError(f"SOURCE 不存在：{SOURCE}")

    # 推理通常不需要随机数种子：
    # 模型处于 eval/inference 状态，正常预测没有训练阶段的随机更新过程。

    with tempfile.TemporaryDirectory(
        prefix="ultralytics_predict_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        predict_model = prepare_model_with_custom_names(
            temp_root
        )

        # ----------------------------------------------------
        # CPU 预测
        # ----------------------------------------------------
        if isinstance(GPU_DEVICES, str):

            if GPU_DEVICES.lower() != "cpu":
                raise ValueError(
                    'GPU_DEVICES 为字符串时目前只支持 "cpu"。'
                )

            command = build_command(
                model=predict_model,
                source=SOURCE,
                device="cpu",
                batch_size=BATCH_SIZE,
                name=NAME,
            )

            print("\n📌 实际执行参数：")
            print(f"yolo {command}")

            print("\n" + "=" * 80)
            print("🏃 开始 CPU 预测")
            print("=" * 80)

            entrypoint(command)
            return

        # ----------------------------------------------------
        # GPU 参数检查
        # ----------------------------------------------------
        if (
            not isinstance(GPU_DEVICES, (list, tuple))
            or not GPU_DEVICES
        ):
            raise ValueError(
                'GPU_DEVICES 应为 [0]、[0, 1] 等列表，或者 "cpu"。'
            )

        gpu_devices = [int(x) for x in GPU_DEVICES]

        # ----------------------------------------------------
        # 单 GPU 预测
        # ----------------------------------------------------
        if len(gpu_devices) == 1:

            device = str(gpu_devices[0])

            command = build_command(
                model=predict_model,
                source=SOURCE,
                device=device,
                batch_size=BATCH_SIZE,
                name=NAME,
            )

            print("\n📌 实际执行参数：")
            print(f"yolo {command}")

            print("\n" + "=" * 80)
            print(f"🏃 开始 GPU {device} 预测")
            print("=" * 80)

            entrypoint(command)
            return

        # ----------------------------------------------------
        # 多 GPU 文件级并行预测
        # ----------------------------------------------------

        if not source_path.is_dir():
            raise ValueError(
                "多 GPU 并行预测要求 SOURCE 为目录。\n"
                "如果 SOURCE 是单张图片或单个视频，请使用单 GPU；\n"
                "若要多 GPU 处理视频，请先把视频切成多个文件后放入同一目录。"
            )

        if BATCH_SIZE < len(gpu_devices):
            raise ValueError(
                f"BATCH_SIZE={BATCH_SIZE} 小于 GPU 数量="
                f"{len(gpu_devices)}。"
            )

        if BATCH_SIZE % len(gpu_devices) != 0:
            raise ValueError(
                f"BATCH_SIZE={BATCH_SIZE} 必须能被 GPU 数量="
                f"{len(gpu_devices)} 整除。"
            )

        files = collect_source_files(source_path)

        if not files:
            raise RuntimeError(
                f"SOURCE 目录中没有找到支持的图片或视频：{SOURCE}"
            )

        per_gpu_batch = BATCH_SIZE // len(gpu_devices)

        file_groups = [
            files[i::len(gpu_devices)]
            for i in range(len(gpu_devices))
        ]

        active_workers = []

        print("\n📊 多 GPU 文件分配：")
        for gpu_id, group in zip(
            gpu_devices,
            file_groups,
        ):
            print(
                f"   GPU {gpu_id}: {len(group)} 个文件，"
                f"Batch Size={per_gpu_batch}"
            )

        print(
            "\n⚠️ 多 GPU predict 采用文件级并行，"
            "结果会写入 NAME/gpu_X 子目录。"
        )

        print("\n" + "=" * 80)
        print("📌 多 GPU 最终实际执行参数")
        print("=" * 80)

        for gpu_id, group in zip(
            gpu_devices,
            file_groups,
        ):

            if not group:
                continue

            worker_source_dir = (
                temp_root / "sources" / f"gpu_{gpu_id}"
            )

            create_worker_source_dir(
                group,
                worker_source_dir,
            )

            worker_name = f"{NAME}/gpu_{gpu_id}"

            command = build_command(
                model=predict_model,
                source=str(worker_source_dir),
                device=str(gpu_id),
                batch_size=per_gpu_batch,
                name=worker_name,
            )

            print(f"\n🖥️ GPU {gpu_id}：")
            print(f"yolo {command}")

            process = mp.Process(
                target=run_predict_worker,
                args=(
                    predict_model,
                    str(worker_source_dir),
                    gpu_id,
                    per_gpu_batch,
                    worker_name,
                ),
            )

            process.start()

            active_workers.append(
                (gpu_id, process)
            )

        failed = []

        for gpu_id, process in active_workers:

            process.join()

            if process.exitcode != 0:
                failed.append(
                    (gpu_id, process.exitcode)
                )

        if failed:
            raise RuntimeError(
                f"部分 GPU 预测失败：{failed}"
            )

        print("\n" + "=" * 80)
        print("✅ 多 GPU 预测完成")
        print("=" * 80)
        print(
            f"📁 结果目录：{PROJECT}/{NAME}/gpu_X"
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
