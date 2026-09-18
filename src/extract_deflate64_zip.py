"""Extract a ZIP that uses Deflate64 (method 9), which the standard library
zipfile and Windows bsdtar both refuse.  Needs `pip install inflate64`.

    python src/extract_deflate64_zip.py ARCHIVE.zip [--dest DIR] [--skip SUBSTR ...]

Every entry is CRC-checked after extraction; failures are listed and exit 1.
Used on 2026-09-17 for the LBMC Lyon Example_Anonymized_Video_Data.zip.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import zipfile
import zlib

import inflate64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--dest", default=".")
    ap.add_argument("--skip", nargs="*", default=[], help="skip entries whose path contains any of these")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.archive)
    t0 = time.time()
    n = total = 0
    bad: list[str] = []
    with open(args.archive, "rb") as fp:
        for info in z.infolist():
            if info.is_dir() or any(s in info.filename for s in args.skip):
                continue
            fp.seek(info.header_offset)
            sig, _, _, _, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", fp.read(30))
            if sig != 0x04034B50:
                raise SystemExit(f"bad local header for {info.filename}")
            fp.seek(info.header_offset + 30 + nlen + elen)
            out = os.path.join(args.dest, info.filename)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            remaining = info.compress_size
            crc = 0
            inflater = inflate64.Inflater() if info.compress_type == 9 else None
            if info.compress_type not in (0, 9):
                raise SystemExit(f"unsupported method {info.compress_type}: {info.filename}")
            with open(out, "wb") as fo:
                while remaining > 0:
                    chunk = fp.read(min(1 << 22, remaining))
                    remaining -= len(chunk)
                    data = inflater.inflate(chunk) if inflater else chunk
                    crc = zlib.crc32(data, crc)
                    fo.write(data)
            if (crc & 0xFFFFFFFF) != info.CRC or os.path.getsize(out) != info.file_size:
                bad.append(info.filename)
            n += 1
            total += info.file_size
    print(f"extracted {n} files, {total / 1e6:.0f} MB in {time.time() - t0:.0f} s; CRC failures: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
