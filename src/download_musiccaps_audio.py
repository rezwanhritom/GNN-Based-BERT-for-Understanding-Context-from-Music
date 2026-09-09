"""
Download MusicCaps YouTube clips as official 10 s segments into data/raw/musiccaps/audio/.

Official MusicCaps provides ytid + start_s/end_s (10 s). We download via yt-dlp then
ffmpeg-cut the exact window (avoids fragile download_ranges / SABR issues).

Usage:
  python -m src.download_musiccaps_audio --limit 2000 --workers 2
"""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg
import pandas as pd
from tqdm import tqdm

def _ffmpeg_cut(
    src: Path,
    dst: Path,
    start_s: float,
    end_s: float,
    ffmpeg_path: str,
) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.1, float(end_s) - float(start_s))
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        str(float(start_s)),
        "-t",
        str(dur),
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "22050",
        "-vn",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return dst.exists() and dst.stat().st_size > 1000
    except (subprocess.SubprocessError, OSError):
        return False

def download_one(
    ytid: str,
    start_s: float,
    end_s: float,
    out_dir: Path,
    ffmpeg_path: str,
) -> tuple[str, bool, str]:
    """Download YouTube audio and cut [start_s, end_s] -> `{ytid}_{start}_{end}.wav`."""
    start_s = float(start_s)
    end_s = float(end_s)
    out_wav = out_dir / f"{ytid}_{int(start_s)}_{int(end_s)}.wav"
    if out_wav.exists() and out_wav.stat().st_size > 1000:
        return ytid, True, "exists"

    try:
        import yt_dlp
    except ImportError:
        return ytid, False, "yt-dlp missing"

    tmp_dir = out_dir / "_yt_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_tmpl = str(tmp_dir / f"{ytid}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_tmpl,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": str(Path(ffmpeg_path).parent),
        "socket_timeout": 30,
        "retries": 3,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    url = f"https://www.youtube.com/watch?v={ytid}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        matches = sorted(tmp_dir.glob(f"{ytid}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return ytid, False, "no_download"
        src = matches[0]
        ok = _ffmpeg_cut(src, out_wav, start_s, end_s, ffmpeg_path)
        # cleanup temps for this id
        for m in tmp_dir.glob(f"{ytid}.*"):
            try:
                m.unlink()
            except OSError:
                pass
        if ok:
            return ytid, True, "ok"
        return ytid, False, "ffmpeg_cut_failed"
    except Exception as exc:  # noqa: BLE001
        return ytid, False, str(exc)[:120]

def download_from_hf(out_dir: Path, ytids: list[str] | None = None, workers: int = 8) -> tuple[int, int]:
    """Optional HF mirror path (prefer YouTube 10 s clips)."""
    raise RuntimeError(
        "HF mirror download is disabled here. "
        "Run: python -m src.download_musiccaps_audio --source youtube"
    )

def list_hf_ytids() -> set[str]:
    return set()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/musiccaps/musiccaps-public.csv")
    parser.add_argument("--out", type=str, default="data/raw/musiccaps/audio")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--source",
        type=str,
        default="youtube",
        choices=["youtube"],
        help="Download YouTube 10 s segments",
    )
    parser.add_argument("--clear-nonstandard", action="store_true", help="Remove non {ytid_start_end}.wav files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if args.limit is not None:
        df = df.head(args.limit)

    if args.clear_nonstandard:
        for p in out_dir.glob("*.wav"):
            parts = p.stem.split("_")
            # official stem: ytid_start_end (ytid may start with '-')
            if len(parts) < 3 or not parts[-1].isdigit() or not parts[-2].isdigit():
                p.unlink(missing_ok=True)

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[musiccaps] YouTube 10s clips -> {out_dir} (n={len(df)}) ffmpeg={ffmpeg_path}")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                download_one,
                str(row["ytid"]),
                float(row["start_s"]),
                float(row["end_s"]),
                out_dir,
                ffmpeg_path,
            )
            for _, row in df.iterrows()
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="musiccaps YT 10s"):
            _, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                if fail <= 8:
                    tqdm.write(f"[warn] {msg}")
            time.sleep(0.05)

    n_files = len(list(out_dir.glob("*_*_*.wav")))
    print(f"[musiccaps] done ok={ok} fail={fail} standard_10s_wavs={n_files}")

if __name__ == "__main__":
    main()
