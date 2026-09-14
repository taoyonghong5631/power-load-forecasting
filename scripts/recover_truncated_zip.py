# -*- coding: utf-8 -*-
"""从下载中断的 zip 里恢复已收到的数据。

zip 下载中断时会留下一个「有本地头、没有中央目录」的半成品，标准工具直接报 BadZipFile。
但 deflate 流是自同步的，可以手动解压出已收到部分对应的全部内容——一次网络抖动
不至于让整个下载白费。

用法:
    python scripts/recover_truncated_zip.py data/LD2011_2014.txt.zip data/cache/partial.txt
"""
from __future__ import annotations

import os
import struct
import sys
import zlib


def recover(zip_path: str, out_path: str) -> int:
    with open(zip_path, "rb") as fh:
        blob = fh.read()

    if blob[:4] != b"PK\x03\x04":
        raise ValueError("不是 zip 本地文件头，无法恢复")
    name_len = struct.unpack_from("<H", blob, 26)[0]
    extra_len = struct.unpack_from("<H", blob, 28)[0]
    start = 30 + name_len + extra_len
    print("本地头之后的数据区: %d bytes" % (len(blob) - start))

    decomp = zlib.decompressobj(-15)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written = 0
    with open(out_path, "wb") as out:
        step = 1 << 20
        for i in range(start, len(blob), step):
            try:
                chunk = decomp.decompress(blob[i:i + step])
            except zlib.error as exc:
                print("遇到不可恢复的 deflate 数据，停止于 %d bytes: %s" % (written, exc))
                break
            if chunk:
                out.write(chunk)
                written += len(chunk)
    print("解压出 %.1f MB -> %s" % (written / 1e6, out_path))
    return written


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/LD2011_2014.txt.zip"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/cache/partial_LD2011_2014.txt"
    recover(src, dst)
    # 检查最后一行是否完整（被截断的最后一列可能残缺）
    with open(dst, "rb") as fh:
        fh.seek(max(0, os.path.getsize(dst) - 300))
        tail = fh.read().decode("utf-8", "ignore")
    lines = [ln for ln in tail.split("\n") if ln.strip()]
    print("末尾整行示例:", lines[0][:60] if lines else "(空)")
