"""Pack BigEarthNet-S2 (v2 / reBEN) into compressed .npz shards for Kaggle upload.

Reads straight from the original BigEarthNet-S2.tar.zst archive as one sequential
stream. (Reading the ~6.6 million extracted .tif files one by one is ~500x slower on
Windows because every small-file open is slow, e.g. due to antivirus scanning.)

Each shard  shard_<split>_<NNNNN>.npz  holds up to --shard-size patches at native
resolution, uint16, zlib-compressed:
    patch_id  (N,)             str
    b10       (N, 4, 120, 120) B02 B03 B04 B08           (10 m)
    b20       (N, 6,  60,  60) B05 B06 B07 B8A B11 B12   (20 m)
    b60       (N, 2,  20,  20) B01 B09                   (60 m)
plus  index.csv  (patch_id, shard, row, split).

Labels: pass --metadata path/to/metadata.parquet (BigEarthNet v2 Zenodo record) to
shard by official split (train/validation/test) and drop patches not listed there
(the snow/cloud ones). Without it everything goes under split "all" and labels are
joined later on Kaggle by patch_id.

Resumable: shards already on disk are skipped on rerun (same arguments => same shards).

Examples
    python scripts/pack_bigearthnet.py --limit 4000 --out D:/ben_test        # quick test
    python scripts/pack_bigearthnet.py --metadata metadata.parquet           # full run
    python scripts/pack_bigearthnet.py --frac 0.2 --out D:/ben_shards_20     # 20 % subset
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tarfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "BigEarthNet-S2.tar.zst"

GROUPS = {
    "b10": (["B02", "B03", "B04", "B08"], 120),
    "b20": (["B05", "B06", "B07", "B8A", "B11", "B12"], 60),
    "b60": (["B01", "B09"], 20),
}
ALL_BANDS = {b for bands, _ in GROUPS.values() for b in bands}


def write_shard(out_path: str, patches: list[tuple[str, dict[str, bytes]]]) -> tuple[str, int, list[str]]:
    """Decode the TIFF bytes of one shard, then compress and write it (runs in a worker)."""
    import tifffile

    n = len(patches)
    arrays = {k: np.zeros((n, len(b), s, s), np.uint16) for k, (b, s) in GROUPS.items()}
    errors = []
    for i, (pid, files) in enumerate(patches):
        for key, (bands, size) in GROUPS.items():
            for j, b in enumerate(bands):
                try:
                    a = tifffile.imread(io.BytesIO(files[b]))
                    if a.shape != (size, size):
                        raise ValueError(f"shape {a.shape}")
                    arrays[key][i, j] = a
                except Exception as e:  # leave zeros, report
                    errors.append(f"{pid} {b}: {e!r}")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, patch_id=np.array([p for p, _ in patches]), **arrays)
    os.replace(tmp, out_path)
    return out_path, n, errors


def iter_patches(archive: Path):
    """Yield (patch_id, {band: tif_bytes}) from the tar.zst stream, in archive order."""
    import zstandard

    cur_id, cur = None, {}
    with open(archive, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh, read_size=1 << 22)
        with tarfile.open(fileobj=reader, mode="r|") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".tif")):
                    continue
                stem = m.name.rsplit("/", 1)[-1][:-4]
                pid, band = stem.rsplit("_", 1)
                if pid != cur_id:
                    if cur_id is not None:
                        yield cur_id, cur
                    cur_id, cur = pid, {}
                cur[band] = tf.extractfile(m).read()
    if cur_id is not None:
        yield cur_id, cur


def keep_frac(pid: str, frac: float, seed: int) -> bool:
    if frac >= 1.0:
        return True
    h = int.from_bytes(hashlib.blake2b(f"{seed}:{pid}".encode(), digest_size=8).digest(), "big")
    return h / 2**64 < frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--out", type=Path, default=Path("D:/ben_shards"))
    ap.add_argument("--metadata", type=Path, default=None, help="BigEarthNet v2 metadata.parquet")
    # ~1 GB RAM per in-flight shard of 2000 patches; raise both if you have free RAM.
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frac", type=float, default=1.0, help="deterministic random fraction to keep")
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept patches (testing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split_of: dict[str, str] | None = None
    if args.metadata:
        meta = pd.read_parquet(args.metadata)
        if not {"patch_id", "split"} <= set(meta.columns):
            raise SystemExit(f"metadata needs patch_id and split columns, got {list(meta.columns)}")
        split_of = dict(zip(meta["patch_id"], meta["split"]))
        print(f"metadata: {len(split_of):,} patches, splits {meta['split'].value_counts().to_dict()}")

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    buffers: dict[str, list] = {}
    counters: dict[str, int] = {}
    index_rows: list[tuple[str, str, int, str]] = []
    errors: list[str] = []
    pending: set = set()
    kept = skipped_meta = incomplete = written = reused = 0

    def drain(block_until: int) -> None:
        nonlocal written
        while len(pending) > block_until:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                pending.remove(f)
                path, n, errs = f.result()
                errors.extend(errs)
                written += 1
                el = time.time() - t0
                print(f"wrote {Path(path).name} ({n})  kept {kept:,}  {kept / el:,.0f} patches/s  "
                      f"{el / 60:.1f} min", flush=True)

    def flush(ex: ProcessPoolExecutor, split: str) -> None:
        nonlocal reused
        items = buffers.pop(split, [])
        if not items:
            return
        k = counters.get(split, 0)
        counters[split] = k + 1
        name = f"shard_{split}_{k:05d}.npz"
        index_rows.extend((pid, name, r, split) for r, (pid, _) in enumerate(items))
        path = str(args.out / name)
        if os.path.exists(path):
            reused += 1
            return
        pending.add(ex.submit(write_shard, path, items))
        drain(args.workers + 1)  # bound memory: at most ~workers shards in flight

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for pid, files in iter_patches(args.archive):
            if split_of is not None:
                split = split_of.get(pid)
                if split is None:
                    skipped_meta += 1
                    continue
            else:
                split = "all"
            if not keep_frac(pid, args.frac, args.seed):
                continue
            if not ALL_BANDS <= files.keys():
                incomplete += 1
                errors.append(f"{pid}: missing bands {sorted(ALL_BANDS - files.keys())}")
                continue
            buffers.setdefault(split, []).append((pid, files))
            kept += 1
            if len(buffers[split]) >= args.shard_size:
                flush(ex, split)
            if args.limit and kept >= args.limit:
                break
        for split in list(buffers):
            flush(ex, split)
        drain(0)

    pd.DataFrame(index_rows, columns=["patch_id", "shard", "row", "split"]).to_csv(
        args.out / "index.csv", index=False)
    if errors:
        (args.out / "errors.txt").write_text("\n".join(errors))
        print(f"WARNING: {len(errors)} problems logged to errors.txt")
    size = sum(f.stat().st_size for f in args.out.glob("shard_*.npz")) / 1e9
    print(f"done in {(time.time() - t0) / 60:.1f} min: {kept:,} patches, {written} shards written, "
          f"{reused} already existed, {skipped_meta:,} not in metadata, {incomplete} incomplete; "
          f"{size:.2f} GB in {args.out}")


if __name__ == "__main__":
    main()
