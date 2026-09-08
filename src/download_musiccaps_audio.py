"""
Download MusicCaps audio into data/raw/musiccaps/audio/.

Primary source: Hugging Face mirror `kelvincai/MusicCaps_30s_wav` (ytid.wav),
because YouTube + yt-dlp is often blocked by bot checks.

Optional fallback: yt-dlp segment download (needs working YouTube access).

Usage:
  python -m src.download_musiccaps_audio --limit 1200
  python -m src.download_musiccaps_audio --source hf
  python -m src.download_musiccaps_audio --source youtube --limit 500
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HF_REPO = "kelvincai/MusicCaps_30s_wav"


def download_one(
    ytid: str,
    start_s: float,
    end_s: float,
    out_dir: Path,
    ffmpeg_path: str,
) -> tuple[str, bool, str]:
    """YouTube segment download fallback (yt-dlp)."""
    start_s = float(start_s)
    end_s = float(end_s)
    out_wav = out_dir / f"{ytid}_{int(start_s)}_{int(end_s)}.wav"
    ytid_wav = out_dir / f"{ytid}.wav"
    if (out_wav.exists() and out_wav.stat().st_size > 1000) or (
        ytid_wav.exists() and ytid_wav.stat().st_size > 1000
    ):
        return ytid, True, "exists"
    try:
        import yt_dlp
        from yt_dlp.utils import download_range_func
    except ImportError:
        return ytid, False, "yt-dlp missing"

    tmp_tmpl = str(out_dir / f"{ytid}_{int(start_s)}_{int(end_s)}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_tmpl,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": str(Path(ffmpeg_path).parent),
        "download_ranges": download_range_func(
            None, [{"start_time": start_s, "end_time": end_s}]
        ),
        "force_keyframes_at_cuts": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "socket_timeout": 30,
        "retries": 2,
    }
    url = f"https://www.youtube.com/watch?v={ytid}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if out_wav.exists() and out_wav.stat().st_size > 1000:
            return ytid, True, "ok"
        matches = list(out_dir.glob(f"{ytid}_{int(start_s)}_{int(end_s)}.*"))
        for m in matches:
            if m.suffix.lower() == ".wav" and m != out_wav:
                m.replace(out_wav)
                return ytid, True, "ok"
            if m.suffix.lower() in {".m4a", ".webm", ".mp3", ".opus"}:
                return ytid, True, f"ok:{m.suffix}"
        return ytid, False, "no_output"
    except Exception as exc:  # noqa: BLE001
        return ytid, False, str(exc)[:120]


def list_hf_ytids() -> set[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
    return {Path(f).stem for f in files if f.endswith(".wav")}


def download_from_hf(out_dir: Path, ytids: list[str] | None = None, workers: int = 8) -> tuple[int, int]:
    """
    Download `{ytid}.wav` files from the HF MusicCaps mirror.
    Only requests IDs that exist in the repo (mirror has ~1710 / 5521).
    Returns (ok, fail).
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    available = list_hf_ytids()
    if ytids is None:
        ytids = sorted(available)
    else:
        requested = list(dict.fromkeys(ytids))
        ytids = [y for y in requested if y in available]
        skipped = len(requested) - len(ytids)
        if skipped:
            print(f"[musiccaps] skipped {skipped} ytids not in HF mirror")

    ok = sum(
        1
        for ytid in ytids
        if (out_dir / f"{ytid}.wav").exists() and (out_dir / f"{ytid}.wav").stat().st_size > 1000
    )
    missing = [
        ytid
        for ytid in ytids
        if not ((out_dir / f"{ytid}.wav").exists() and (out_dir / f"{ytid}.wav").stat().st_size > 1000)
    ]
    print(f"[musiccaps] HF mirror: have={ok} need={len(missing)} repo={HF_REPO}")
    if not missing:
        return ok, 0

    # Bulk snapshot is faster when many files are missing
    if len(missing) >= 50:
        print("[musiccaps] snapshot_download for missing wavs...")
        patterns = [f"{y}.wav" for y in missing]
        # HF allow_patterns has practical size limits; chunk if needed
        chunk = 200
        for i in range(0, len(patterns), chunk):
            part = patterns[i : i + chunk]
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=str(out_dir),
                allow_patterns=part,
                max_workers=workers,
            )
            print(f"[musiccaps] snapshot chunk {i // chunk + 1}/{(len(patterns) + chunk - 1) // chunk}")
        ok = sum(
            1
            for ytid in ytids
            if (out_dir / f"{ytid}.wav").exists() and (out_dir / f"{ytid}.wav").stat().st_size > 1000
        )
        fail = len(ytids) - ok
        return ok, fail

    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        def _one(ytid: str) -> tuple[str, bool, str]:
            dest = out_dir / f"{ytid}.wav"
            try:
                path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=f"{ytid}.wav",
                    repo_type="dataset",
                    local_dir=str(out_dir),
                )
                p = Path(path)
                if p.resolve() != dest.resolve() and p.exists() and not dest.exists():
                    dest.write_bytes(p.read_bytes())
                if dest.exists() and dest.stat().st_size > 1000:
                    return ytid, True, "ok"
                if p.exists() and p.stat().st_size > 1000:
                    return ytid, True, "ok"
                return ytid, False, "empty"
            except Exception as exc:  # noqa: BLE001
                return ytid, False, str(exc)[:120]

        futs = [ex.submit(_one, y) for y in missing]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="musiccaps HF"):
            _, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                if fail <= 5:
                    tqdm.write(f"[warn] {msg}")
    return ok, fail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/musiccaps/musiccaps-public.csv")
    parser.add_argument("--out", type=str, default="data/raw/musiccaps/audio")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--source",
        type=str,
        default="hf",
        choices=["hf", "youtube", "auto"],
        help="hf=HuggingFace mirror (recommended); youtube=yt-dlp; auto=hf then youtube",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    # Prefer IDs that exist on the HF mirror when using hf/auto
    if args.source in {"hf", "auto"}:
        available = list_hf_ytids()
        df = df[df["ytid"].astype(str).isin(available)].reset_index(drop=True)
        print(f"[musiccaps] CSV rows available on HF mirror: {len(df)}")
    if args.limit is not None:
        df = df.head(args.limit)

    ytids = [str(y) for y in df["ytid"].tolist()]

    if args.source in {"hf", "auto"}:
        ok, fail = download_from_hf(out_dir, ytids=ytids, workers=args.workers)
        print(f"[musiccaps] HF done ok~={ok} fail={fail}")

    if args.source == "youtube" or (
        args.source == "auto"
        and sum(1 for y in ytids if (out_dir / f"{y}.wav").exists()) < max(50, len(ytids) // 4)
    ):
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[musiccaps] YouTube fallback for missing clips (ffmpeg={ffmpeg_path})")
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers // 2)) as ex:
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
            for fut in tqdm(as_completed(futs), total=len(futs), desc="musiccaps YT"):
                _, success, msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    if fail <= 5:
                        tqdm.write(f"[warn] {msg}")
                time.sleep(0.01)
        print(f"[musiccaps] YT done ok={ok} fail={fail}")

    n_files = len(list(out_dir.glob("*.wav"))) + len(list(out_dir.glob("*.mp3")))
    print(f"[musiccaps] files_in_dir≈{n_files} -> {out_dir}")


if __name__ == "__main__":
    main()
