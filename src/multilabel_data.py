"""
Multi-label FMA tag/genre targets + DEAM valence/arousal loaders (Task 3).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

def _parse_list_cell(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s or s in {"nan", "[]", "None"}:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return parsed
    except (SyntaxError, ValueError):
        pass
    return [t.strip() for t in re.split(r"[,;]", s) if t.strip()]

def build_fma_multilabel_vocab(
    tracks: pd.DataFrame,
    subset: str = "medium",
    top_k_tags: int = 50,
) -> dict[str, Any]:
    """Genre multi-hot + top-K folksonomy tags (mood/context) for FMA subset."""
    mask = tracks[("set", "subset")] == subset
    sub = tracks.loc[mask]

    genre_ids: set[int] = set()
    tag_counts: dict[str, int] = {}
    for tid in sub.index:
        for gid in _parse_list_cell(sub.loc[tid, ("track", "genres_all")]):
            try:
                genre_ids.add(int(gid))
            except (TypeError, ValueError):
                continue
        for col in (("track", "tags"), ("artist", "tags")):
            if col not in sub.columns:
                continue
            for tag in _parse_list_cell(sub.loc[tid, col]):
                t = str(tag).strip().lower()
                if not t or t == "nan":
                    continue
                tag_counts[t] = tag_counts.get(t, 0) + 1

    genre_ids_sorted = sorted(genre_ids)
    genre_to_idx = {g: i for i, g in enumerate(genre_ids_sorted)}
    top_tags = [t for t, _ in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k_tags]]
    tag_to_idx = {t: i for i, t in enumerate(top_tags)}
    return {
        "genre_ids": genre_ids_sorted,
        "genre_to_idx": genre_to_idx,
        "tag_to_idx": tag_to_idx,
        "id_to_tag": {i: t for t, i in tag_to_idx.items()},
        "n_genre": len(genre_to_idx),
        "n_tag": len(tag_to_idx),
        "n_labels": len(genre_to_idx) + len(tag_to_idx),
    }

def fma_multilabel_vector(
    tracks: pd.DataFrame,
    track_id: int,
    vocab: dict[str, Any],
) -> np.ndarray:
    y = np.zeros(int(vocab["n_labels"]), dtype=np.float32)
    row = tracks.loc[int(track_id)]
    for gid in _parse_list_cell(row[("track", "genres_all")]):
        try:
            gi = int(gid)
        except (TypeError, ValueError):
            continue
        if gi in vocab["genre_to_idx"]:
            y[vocab["genre_to_idx"][gi]] = 1.0
    offset = int(vocab["n_genre"])
    for col in (("track", "tags"), ("artist", "tags")):
        if col not in tracks.columns:
            continue
        for tag in _parse_list_cell(row[col]):
            t = str(tag).strip().lower()
            if t in vocab["tag_to_idx"]:
                y[offset + vocab["tag_to_idx"][t]] = 1.0
    return y

def load_deam_static_annotations(ann_root: Path) -> pd.DataFrame:
    """
    Load averaged song-level valence/arousal (scale 1-9).
    Returns DataFrame indexed by song_id with valence, arousal in [0, 1].
    """
    paths = list(
        (ann_root / "annotations" / "annotations averaged per song" / "song_level").glob(
            "static_annotations_averaged_songs_*.csv"
        )
    )
    if not paths:
        # alternate nesting
        paths = list(ann_root.rglob("static_annotations_averaged_songs_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No DEAM static annotation CSVs under {ann_root}")
    frames = [pd.read_csv(p) for p in sorted(paths)]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    if "valence_mean" in df.columns:
        rename["valence_mean"] = "valence"
    if "arousal_mean" in df.columns:
        rename["arousal_mean"] = "arousal"
    df = df.rename(columns=rename)
    if "valence" not in df.columns or "arousal" not in df.columns:
        raise KeyError(f"Expected valence/arousal columns, got {df.columns.tolist()}")
    df["song_id"] = df["song_id"].astype(int)
    # map [1,9] -> [0,1]
    df["valence"] = ((df["valence"].astype(float) - 1.0) / 8.0).clip(0.0, 1.0)
    df["arousal"] = ((df["arousal"].astype(float) - 1.0) / 8.0).clip(0.0, 1.0)
    return df.set_index("song_id")[["valence", "arousal"]]

def write_deam_splits(
    ann_root: Path,
    out_path: Path,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict[str, Any]:
    """Standard random train/val/test over DEAM song-level annotations."""
    df = load_deam_static_annotations(ann_root)
    ids = np.array(sorted(df.index.tolist()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(test_frac * n))
    n_val = max(1, int(val_frac * n))
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]

    def pack(id_list: np.ndarray) -> list[dict[str, Any]]:
        rows = []
        for sid in id_list:
            rows.append(
                {
                    "song_id": int(sid),
                    "valence": float(df.loc[int(sid), "valence"]),
                    "arousal": float(df.loc[int(sid), "arousal"]),
                }
            )
        return rows

    payload = {
        "dataset": "deam",
        "train": pack(train_ids),
        "val": pack(val_ids),
        "test": pack(test_ids),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
