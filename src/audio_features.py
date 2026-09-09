"""
Audio feature extraction for GNN-BERT music context understanding.

  1. Resample to 22,050 Hz
  2. Extract log-mel spectrogram (128 bins) or chroma (12 bins); normalize per track
  3. Segment into fixed windows (5-10 s) or beat-synchronous segments (librosa)
  5. Splits: official FMA train/val/test (no artist leakage when using FMA splits)

Usage:
  python -m src.audio_features --config config.yaml --write-splits
  python -m src.audio_features --config config.yaml --dataset fma_small
  python -m src.audio_features --config config.yaml --dataset fma_small --limit 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

TARGET_SR = 22050
N_MELS = 128
N_CHROMA = 12

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_audio(path: str | Path, sample_rate: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load mono audio resampled to `sample_rate`."""
    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return y.astype(np.float32), int(sr)

def log_mel_spectrogram(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Log-mel spectrogram with `n_mels` bins. Shape (n_mels, time)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def chroma_features(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_chroma: int = N_CHROMA,
) -> np.ndarray:
    """Chroma features with `n_chroma` bins. Shape (n_chroma, time)."""
    return librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma).astype(np.float32)

def normalize_per_track(features: np.ndarray) -> np.ndarray:
    """Normalize feature matrix per track (zero-mean / unit-std over all values)."""
    mean = float(features.mean())
    std = float(features.std())
    if std < 1e-8:
        return features - mean
    return ((features - mean) / std).astype(np.float32)

def segment_fixed(
    features: np.ndarray,
    sr: int,
    hop_length: int = 512,
    segment_seconds: float = 5.0,
    hop_seconds: float = 2.5,
) -> list[np.ndarray]:
    """
    Split feature matrix (n_features, time) into fixed-duration windows.
    Returns list of (n_features, seg_frames) arrays.
    """
    frames_per_sec = sr / hop_length
    seg_frames = max(1, int(round(segment_seconds * frames_per_sec)))
    hop_frames = max(1, int(round(hop_seconds * frames_per_sec)))
    n_frames = features.shape[1]
    segments: list[np.ndarray] = []
    start = 0
    while start + seg_frames <= n_frames:
        segments.append(features[:, start : start + seg_frames].astype(np.float32))
        start += hop_frames
    if not segments and n_frames > 0:
        # Track shorter than one window - keep full feature as one segment
        segments.append(features.astype(np.float32))
    return segments

def segment_beat_synchronous(
    y: np.ndarray,
    sr: int = TARGET_SR,
    feature: str = "chroma",
    n_mels: int = N_MELS,
    n_chroma: int = N_CHROMA,
) -> list[np.ndarray]:
    """Beat-synchronous segmentation via librosa."""
    _tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if feature == "mel":
        feats = log_mel_spectrogram(y, sr=sr, n_mels=n_mels)
    else:
        feats = chroma_features(y, sr=sr, n_chroma=n_chroma)
    if len(beats) < 2:
        return [feats.astype(np.float32)]
    # Aggregate between consecutive beat frames
    beat_feats = librosa.util.sync(feats, beats, aggregate=np.mean)
    # Each column is one beat-synchronous frame -> treat as a segment
    segments = [beat_feats[:, i : i + 1].astype(np.float32) for i in range(beat_feats.shape[1])]
    return segments

def extract_track_features(path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Full per-track pipeline: load -> mel/chroma -> normalize -> segment."""
    audio_cfg = cfg.get("audio", {})
    sr = int(audio_cfg.get("sample_rate", TARGET_SR))
    n_mels = int(audio_cfg.get("n_mels", N_MELS))
    n_chroma = int(audio_cfg.get("n_chroma", N_CHROMA))
    normalize = bool(audio_cfg.get("normalize_per_track", True))
    segment_seconds = float(audio_cfg.get("segment_seconds", 5.0))
    hop_seconds = float(audio_cfg.get("hop_seconds", 2.5))
    beat_sync = bool(audio_cfg.get("use_beat_synchronous", False))
    graph_feature = cfg.get("graph", {}).get("feature", "chroma")

    y, sr = load_audio(path, sample_rate=sr)
    mel = log_mel_spectrogram(y, sr=sr, n_mels=n_mels)
    chroma = chroma_features(y, sr=sr, n_chroma=n_chroma)
    if normalize:
        mel = normalize_per_track(mel)
        chroma = normalize_per_track(chroma)

    if beat_sync:
        segments = segment_beat_synchronous(
            y, sr=sr, feature=graph_feature, n_mels=n_mels, n_chroma=n_chroma
        )
        if normalize:
            segments = [normalize_per_track(s) for s in segments]
    else:
        base = chroma if graph_feature == "chroma" else mel
        if graph_feature == "mfcc":
            base = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).astype(np.float32)
            if normalize:
                base = normalize_per_track(base)
        segments = segment_fixed(
            base,
            sr=sr,
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
        )

    # Fixed-length segment embedding for graphs: mean over time per segment
    segment_vectors = np.stack([s.mean(axis=1) for s in segments], axis=0).astype(np.float32)

    return {
        "mel": mel,
        "chroma": chroma,
        "segment_vectors": segment_vectors,
        "n_segments": int(segment_vectors.shape[0]),
        "duration_sec": float(len(y) / sr),
        "sr": sr,
    }

def fma_audio_path(raw_fma_root: Path, track_id: int, subset: str = "fma_small") -> Path:
    """FMA layout: fma_small/000/000002.mp3"""
    tid = f"{int(track_id):06d}"
    return raw_fma_root / subset / tid[:3] / f"{tid}.mp3"

def load_fma_tracks_table(metadata_dir: Path) -> pd.DataFrame:
    tracks_csv = metadata_dir / "tracks.csv"
    if not tracks_csv.exists():
        raise FileNotFoundError(
            f"Missing {tracks_csv}. Extract fma_metadata.zip into data/raw/fma/ "
            "(needed for official FMA splits)."
        )
    return pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

def build_fma_splits(
    metadata_dir: Path,
    subset: str = "small",
    out_path: Path | None = None,
) -> dict[str, Any]:
    """
    Official FMA splits (training / validation / test from tracks.csv.
    Avoids custom random splits (artist leakage handled by FMA split design).
    """
    tracks = load_fma_tracks_table(metadata_dir)
    mask = tracks[("set", "subset")] == subset
    sub = tracks.loc[mask]
    split_col = sub[("set", "split")]
    genre_col = sub[("track", "genre_top")]

    genres = sorted({g for g in genre_col.dropna().unique()})
    genre_to_id = {g: i for i, g in enumerate(genres)}

    def pack(split_name: str) -> list[dict[str, Any]]:
        ids = sub.index[split_col == split_name]
        rows: list[dict[str, Any]] = []
        for tid in ids:
            genre = genre_col.loc[tid]
            if pd.isna(genre):
                continue
            rows.append(
                {
                    "track_id": int(tid),
                    "genre": str(genre),
                    "genre_id": int(genre_to_id[str(genre)]),
                    "split": split_name,
                }
            )
        return rows

    # FMA uses 'training' / 'validation' / 'test'
    payload = {
        "dataset": f"fma_{subset}",
        "subset": subset,
        "genre_to_id": genre_to_id,
        "train": pack("training"),
        "val": pack("validation"),
        "test": pack("test"),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return payload

def process_fma_subset(
    cfg: dict[str, Any],
    subset_folder: str = "fma_small",
    limit: int | None = None,
) -> None:
    paths = cfg["datasets"]["paths"]
    raw = Path(paths["raw"])
    processed = Path(paths["processed"]) / subset_folder
    processed.mkdir(parents=True, exist_ok=True)

    metadata_dir = raw / "fma" / "fma_metadata"
    subset_name = subset_folder.replace("fma_", "")  # small | medium
    splits_path = Path(paths["splits"]) / f"{subset_folder}_splits.json"
    splits = build_fma_splits(metadata_dir, subset=subset_name, out_path=splits_path)
    print(f"[splits] wrote {splits_path}")
    print(
        f"  train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} genres={len(splits['genre_to_id'])}"
    )

    all_items = splits["train"] + splits["val"] + splits["test"]
    if limit is not None:
        all_items = all_items[:limit]

    raw_fma = raw / "fma"
    ok, skipped = 0, 0
    for item in tqdm(all_items, desc=f"preprocess {subset_folder}"):
        tid = int(item["track_id"])
        out_file = processed / f"{tid:06d}.npz"
        if out_file.exists():
            ok += 1
            continue
        audio_path = fma_audio_path(raw_fma, tid, subset=subset_folder)
        if not audio_path.exists():
            skipped += 1
            continue
        try:
            feats = extract_track_features(audio_path, cfg)
            np.savez_compressed(
                out_file,
                mel=feats["mel"],
                chroma=feats["chroma"],
                segment_vectors=feats["segment_vectors"],
                track_id=np.int64(tid),
                genre_id=np.int64(item["genre_id"]),
                n_segments=np.int64(feats["n_segments"]),
                duration_sec=np.float32(feats["duration_sec"]),
                sr=np.int64(feats["sr"]),
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001 - continue batch on bad files
            skipped += 1
            tqdm.write(f"[warn] track {tid}: {exc}")
    print(f"[done] saved={ok} skipped={skipped} -> {processed}")

def process_deam_audio(cfg: dict[str, Any], limit: int | None = None) -> None:
    """Extract mel/chroma/segments for DEAM audio (annotations added when available)."""
    paths = cfg["datasets"]["paths"]
    audio_dir = Path(paths["raw"]) / "deam" / "MEMD_audio"
    processed = Path(paths["processed"]) / "deam"
    processed.mkdir(parents=True, exist_ok=True)

    files = sorted(audio_dir.glob("*.mp3"))
    if limit is not None:
        files = files[:limit]

    ok, skipped = 0, 0
    for path in tqdm(files, desc="preprocess deam"):
        song_id = path.stem
        out_file = processed / f"{song_id}.npz"
        if out_file.exists():
            ok += 1
            continue
        try:
            feats = extract_track_features(path, cfg)
            np.savez_compressed(
                out_file,
                mel=feats["mel"],
                chroma=feats["chroma"],
                segment_vectors=feats["segment_vectors"],
                track_id=np.array(song_id),
                n_segments=np.int64(feats["n_segments"]),
                duration_sec=np.float32(feats["duration_sec"]),
                sr=np.int64(feats["sr"]),
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            tqdm.write(f"[warn] deam {song_id}: {exc}")
    print(f"[done] saved={ok} skipped={skipped} -> {processed}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Audio preprocessing and FMA splits")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fma_small", "fma_medium", "deam", "all"],
        default="fma_small",
    )
    parser.add_argument("--write-splits", action="store_true", help="Only write FMA split JSON")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N tracks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["datasets"]["paths"]
    metadata_dir = Path(paths["raw"]) / "fma" / "fma_metadata"

    if args.write_splits:
        subset = str(cfg["datasets"]["primary_audio"]).replace("fma_", "")
        out = Path(paths["splits"]) / f"{cfg['datasets']['primary_audio']}_splits.json"
        build_fma_splits(metadata_dir, subset=subset, out_path=out)
        print(f"[splits] wrote {out}")
        return

    if args.dataset in {"fma_small", "fma_medium", "all"}:
        folder = args.dataset if args.dataset != "all" else str(cfg["datasets"]["primary_audio"])
        if args.dataset == "all":
            folder = str(cfg["datasets"]["primary_audio"])
        process_fma_subset(cfg, subset_folder=folder, limit=args.limit)

    if args.dataset in {"deam", "all"}:
        process_deam_audio(cfg, limit=args.limit)

if __name__ == "__main__":
    main()
