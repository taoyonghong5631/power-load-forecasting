# -*- coding: utf-8 -*-
"""下载并解压 UCI ElectricityLoadDiagrams20112014 数据集。

用法:
    python scripts/fetch_data.py

特点:
    * 先下载到 .part 临时文件，校验 zip 完整性后再原子替换目标文件，
      避免网络中断留下「看似存在、实则损坏」的半个 zip。
    * 支持 --url 指定镜像。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover_truncated_zip import recover  # noqa: E402  （同目录工具脚本）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
ZIP_PATH = os.path.join(DATA_DIR, "LD2011_2014.txt.zip")
TXT_PATH = os.path.join(DATA_DIR, "LD2011_2014.txt")

# 实测（国内网络）static 地址持续速度约 58 KB/s，旧地址约 22 KB/s，故 static 优先。
# 该文件约 140 MB 且服务器不支持 Range 断点续传，下载慢属于正常现象。
MIRRORS = [
    "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
]
DEFAULT_URL = MIRRORS[0]

# 期望的完整文件大小（字节），用于发现"下到一半就断"的情况；未知则填 0
EXPECTED_SIZE = 142_000_000
EXPECTED_TOLERANCE = 0.5  # 允许 50% 偏差（服务器可能不返回 Content-Length）


def _log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def download(url: str, dst: str, timeout: int = 60) -> None:
    """流式下载到 dst+'.part'，校验 zip 后再原子替换到 dst。"""
    tmp = dst + ".part"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; load-forecast-fetch/1.0)"}
    got = 0
    total = 0
    t0 = time.time()
    interrupted = None
    with requests.get(url, stream=True, timeout=(15, timeout), headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        next_log = 0
        try:
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 18):
                    fh.write(chunk)
                    got += len(chunk)
                    if got >= next_log:  # 每约 8MB 打一次进度
                        next_log = got + (8 << 20)
                        speed = got / 1024 / max(time.time() - t0, 1e-6)
                        pct = ("%5.1f%%" % (got / total * 100)) if total else "  ?  "
                        _log("  %s  %7.1f MB  %7.1f KB/s" % (pct, got / 1e6, speed))
        except Exception as exc:
            # UCI 用 chunked 传输，偶尔会在最后丢掉结束标记（Response ended
            # prematurely）。此时数据可能其实已经收全，先直接校验一次再说。
            interrupted = exc
    _log("  下载完成: %.1f MB, 耗时 %.0fs" % (got / 1e6, time.time() - t0))
    if total and got != total:
        raise IOError("下载不完整: %d != %d" % (got, total))
    if not total and EXPECTED_SIZE:
        lo = EXPECTED_SIZE * (1 - EXPECTED_TOLERANCE)
        if got < lo:
            raise IOError("下载疑似不完整: 只拿到 %.1f MB，预期约 %.1f MB"
                          % (got / 1e6, EXPECTED_SIZE / 1e6))

    try:
        zipfile.ZipFile(tmp).testzip()
    except Exception as exc:
        if interrupted is not None:
            _log("  传输中断(%s)且已收数据不是完整 zip，需要重试"
                 % type(interrupted).__name__)
        raise
    if interrupted is not None:
        _log("  传输虽中断(%s)，但已收数据是完整 zip，直接采用"
             % type(interrupted).__name__)
    if os.path.exists(dst):
        os.remove(dst)
    os.replace(tmp, dst)
    _log("  zip 校验通过 -> %s" % dst)


def extract(zip_path: str, txt_path: str) -> None:
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        _log("已存在解压结果，跳过: %s (%.1f MB)"
             % (txt_path, os.path.getsize(txt_path) / 1e6))
        return
    _log("解压中（300+ MB，需要一点时间）...")
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        with zf.open(member) as src, open(txt_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    _log("解压完成: %s (%.1f MB)" % (txt_path, os.path.getsize(txt_path) / 1e6))


def inspect_txt(path: str):
    """返回 (行数估算, 最后一条完整记录的时间戳, 文件大小 MB)。"""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - 300000))
        tail = fh.read().decode("utf-8", "ignore")
    last_ts = "?"
    for line in reversed([ln for ln in tail.split("\n") if ln.strip()]):
        head = line.split(";")[0].strip().strip('"')
        if head[:4].isdigit():
            last_ts = head
            break
    with open(path, "rb") as fh:
        sample = fh.read(400000)
    avg_row = 3000.0
    rows = int(size / avg_row)
    return rows, last_ts, size / 1e6


def salvage(zip_partial: str, keep_name: str, txt_out: str):
    """抢救中断的 zip：转存部分文件 -> 解出文本 -> 报告覆盖范围。"""
    keep = os.path.join(DATA_DIR, keep_name)
    if os.path.exists(keep):
        os.remove(keep)
    shutil.move(zip_partial, keep)
    try:
        recover(keep, txt_out)
    except Exception as exc:
        _log("  抢救失败: %s: %s" % (type(exc).__name__, exc))
        return None
    rows, last_ts, mb = inspect_txt(txt_out)
    _log("  抢救结果: %.0f MB 文本，约 %d 行，最后时间戳 %s" % (mb, rows, last_ts))
    return {"zip": keep, "txt": txt_out, "rows": rows, "last_ts": last_ts, "mb": mb}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="自定义下载地址（默认自动尝试多个镜像）")
    ap.add_argument("--force", action="store_true", help="即使 zip 已存在也重新下载")
    ap.add_argument("--min-rows", type=int, default=60000,
                    help="抢救结果至少覆盖多少行才认为可用（默认 6 万行≈2013 年中）")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    need_download = args.force or not os.path.exists(ZIP_PATH)
    if not need_download:
        try:
            zipfile.ZipFile(ZIP_PATH).testzip()
            _log("已存在且完整的 zip，跳过下载: %s" % ZIP_PATH)
        except Exception as exc:  # 损坏的半个文件
            _log("已有 zip 损坏（%s），重新下载" % type(exc).__name__)
            need_download = True

    if need_download:
        urls = [args.url] if args.url else MIRRORS
        last_err = None
        attempts_per_url = 2
        salvaged = []
        n_attempt = 0
        for url in urls:
            for attempt in range(1, attempts_per_url + 1):
                _log("尝试下载: %s（第 %d/%d 次）" % (url, attempt, attempts_per_url))
                try:
                    download(url, ZIP_PATH)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    _log("  失败: %s: %s" % (type(exc).__name__, exc))
                    # 关键：中断的文件不要丢——deflate 流是自同步的，
                    # 已经收到的部分能解出对应比例的数据行
                    part = ZIP_PATH + ".part"
                    if os.path.exists(part) and os.path.getsize(part) > 1 << 20:
                        n_attempt += 1
                        info = salvage(part,
                                       "LD2011_2014.partial%d.zip" % n_attempt,
                                       os.path.join(CACHE_DIR,
                                                    "recovered_partial%d.txt" % n_attempt))
                        if info:
                            salvaged.append(info)
                    if attempt < attempts_per_url:
                        _log("  5 秒后重试同一镜像（不支持断点续传，只能从 0 开始）")
                        time.sleep(5)
            if last_err is None:
                break
        if last_err is not None:
            # 全部失败时，退回"最完整的一次抢救结果"
            best = max(salvaged, key=lambda d: d["rows"]) if salvaged else None
            if best and best["rows"] >= args.min_rows:
                if not os.path.exists(TXT_PATH):
                    shutil.copyfile(best["txt"], TXT_PATH)
                _log("注意：完整文件始终没下下来，已使用抢救出的部分数据")
                _log("  覆盖到 %s（约 %d 行，%.0f MB）"
                     % (best["last_ts"], best["rows"], best["mb"]))
                _log("  这不是完整数据集，结果会在 README 中标注。")
                return 0
            _log("所有镜像均失败且抢救结果不足（%s），请稍后重试或手动下载。"
                 % (best["last_ts"] if best else "无"))
            return 1

    extract(ZIP_PATH, TXT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
