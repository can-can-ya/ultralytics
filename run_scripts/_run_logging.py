"""将运行输出同时写入终端和结果目录，支持 DDP 子进程输出。"""

import contextlib
import logging
import os
from pathlib import Path
import sys
import shutil
import tempfile
import threading
import traceback


class _PlainTextFilter:
    """按流过滤 ANSI 控制码，支持转义序列跨多次 write/read。

    保留原始 UTF-8 字节；将进度条的回车刷新转换为换行。
    stdout/stderr 各自使用实例，避免控制序列状态互相干扰。
    """

    def __init__(self):
        self.state = "text"
        self.previous_cr = False

    def feed(self, data):
        output = bytearray()
        for char in data:
            if self.state == "escape":
                if char == ord("["):
                    self.state = "csi"
                elif char in (ord("]"), ord("P"), ord("^"), ord("_")):
                    self.state = "string"
                elif 0x20 <= char <= 0x2F:
                    self.state = "escape_intermediate"
                else:
                    self.state = "text"
                continue
            if self.state == "escape_intermediate":
                if 0x30 <= char <= 0x7E:
                    self.state = "text"
                continue
            if self.state == "csi":
                if 0x40 <= char <= 0x7E:
                    self.state = "text"
                continue
            if self.state == "string":
                if char == 7:
                    self.state = "text"
                elif char == 27:
                    self.state = "string_escape"
                continue
            if self.state == "string_escape":
                if char in (ord("\\"), 7):
                    self.state = "text"
                elif char != 27:
                    self.state = "string"
                continue
            if char == 27:
                self.state = "escape"
            elif char == 13:
                output.append(10)
                self.previous_cr = True
            elif char == 10:
                if not self.previous_cr:
                    output.append(10)
                self.previous_cr = False
            else:
                self.previous_cr = False
                if char == 9 or (char >= 32 and char != 127):
                    output.append(char)
        return bytes(output)


class _RecoverableLog:
    """在结果目录外备份日志，防止 DDP 清理目录后日志写入已删除的文件。

    不长期持有目标文件句柄，也避免 Windows 下阻止 DDP 清理目录。
    调用方需加锁，保证 stdout/stderr 的备份和落盘顺序一致。
    """

    def __init__(self, path, backup):
        self.path = path
        self.backup = backup
        if path.exists():
            with path.open("rb") as existing:
                shutil.copyfileobj(existing, backup)
        self.size = backup.tell()
        self.flush()

    def write(self, data):
        previous_size = self.size
        self.backup.seek(0, os.SEEK_END)
        self.backup.write(data)
        self.size += len(data)
        try:
            if self.path.stat().st_size == previous_size:
                with self.path.open("ab") as target:
                    target.write(data)
            else:
                self.flush()
        except FileNotFoundError:
            # DDP 删除/重建结果目录后，从备份恢复，包括启动阶段的日志。
            # 若目录正在被清理，则保留备份，在下次输出或退出时重试。
            try:
                self.flush()
            except FileNotFoundError:
                pass
        return len(data)

    def flush(self):
        try:
            if self.path.stat().st_size == self.size:
                return
        except FileNotFoundError:
            pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backup.seek(0)
        try:
            with self.path.open("wb") as target:
                shutil.copyfileobj(self.backup, target)
        finally:
            self.backup.seek(0, os.SEEK_END)


class _TeeStream:
    """直接复制 Python 输出流，无后台线程或管道退出等待。"""

    def __init__(self, stream, log_file, lock):
        self.stream = stream
        self.log_file = log_file
        self.lock = lock
        self.text_filter = _PlainTextFilter()

    def write(self, text):
        result = self.stream.write(text)
        with self.lock:
            data = self.text_filter.feed(text.encode("utf-8", errors="replace"))
            if data:
                self.log_file.write(data)
        return result

    def flush(self):
        self.stream.flush()
        with self.lock:
            self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def log_to_file(save_dir, *, capture_subprocess=True):
    """复制 stdout/stderr 到 log.txt，退出时恢复输出并记录异常。

    capture_subprocess=True 时重定向底层文件描述符，以捕获训练 DDP
    子进程继承的输出；IDE 自定义输出流则使用 Python 层的 tee。
    预测使用 capture_subprocess=False：主进程和各 worker 独立记录
    print、logging、进度条，不创建可能被子进程持有的日志管道。
    此模式不捕获绕过 Python 输出流的原生库或外部命令输出。
    日志采用追加模式，同名运行不会清空已有日志。
    仅文件端过滤 ANSI 控制码，终端仍保留颜色和进度刷新。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    old_unbuffered = os.environ.get("PYTHONUNBUFFERED")
    lock = threading.Lock()
    redirects = []
    streams = []
    handlers = []
    errors = []

    def copy_output(read_fd, terminal_fd, log_file):
        text_filter = _PlainTextFilter()
        with os.fdopen(read_fd, "rb", buffering=0) as reader:
            while True:
                data = reader.read(65536)
                if not data:
                    break
                # 即使某一输出目的地失败，也必须继续排空管道，避免训练阻塞。
                try:
                    remaining = memoryview(data)
                    while remaining:
                        written = os.write(terminal_fd, remaining)
                        remaining = remaining[written:]
                except OSError as exc:
                    if not errors:
                        errors.append(exc)
                try:
                    clean_data = text_filter.feed(data)
                    if clean_data:
                        with lock:
                            log_file.write(clean_data)
                except OSError as exc:
                    if not errors:
                        errors.append(exc)

    # 系统临时目录不受训练结果目录清理影响；结束时自动关闭临时备份。
    with tempfile.TemporaryFile(mode="w+b") as backup:
        log_file = _RecoverableLog(save_dir / "log.txt", backup)
        os.environ["PYTHONUNBUFFERED"] = "1"
        try:
            for name in ("stdout", "stderr"):
                stream = getattr(sys, name)
                stream.flush()
                fd = None
                if capture_subprocess:
                    try:
                        fd = stream.fileno()
                    except (AttributeError, OSError, ValueError):
                        pass
                if fd is None:
                    replacement = _TeeStream(stream, log_file, lock)
                    streams.append((name, stream))
                    setattr(sys, name, replacement)
                    # logging 的 StreamHandler 可能在导入时已绑定旧输出流。
                    loggers = [logging.getLogger()] + [
                        item for item in logging.Logger.manager.loggerDict.values()
                        if isinstance(item, logging.Logger)
                    ]
                    for logger in loggers:
                        for handler in logger.handlers:
                            if isinstance(handler, logging.StreamHandler) and handler.stream is stream:
                                handlers.append((handler, stream))
                                handler.setStream(replacement)
                    continue

                terminal_fd = os.dup(fd)
                read_fd, write_fd = os.pipe()
                thread = threading.Thread(
                    target=copy_output,
                    args=(read_fd, terminal_fd, log_file),
                    daemon=True,
                )
                try:
                    thread.start()
                except BaseException:
                    os.close(read_fd)
                    os.close(write_fd)
                    os.close(terminal_fd)
                    raise
                redirects.append((fd, terminal_fd, thread))
                try:
                    os.dup2(write_fd, fd)
                finally:
                    os.close(write_fd)

            print(f"日志文件：{save_dir / 'log.txt'}", flush=True)
            try:
                yield
            except BaseException:
                traceback.print_exc()
                raise
        finally:
            for stream in (sys.stdout, sys.stderr):
                stream.flush()
            for handler, stream in reversed(handlers):
                handler.setStream(stream)
            for name, stream in reversed(streams):
                setattr(sys, name, stream)
            for fd, terminal_fd, thread in reversed(redirects):
                os.dup2(terminal_fd, fd)
                thread.join()
                os.close(terminal_fd)
            if old_unbuffered is None:
                os.environ.pop("PYTHONUNBUFFERED", None)
            else:
                os.environ["PYTHONUNBUFFERED"] = old_unbuffered
            # 即使目录清理后没有新输出，也要在退出时恢复日志。
            log_file.flush()
            if errors:
                print(f"警告：日志复制发生错误：{errors[0]}", file=sys.stderr)
