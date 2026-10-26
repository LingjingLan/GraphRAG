"""计算标注 CSV 的 Krippendorff's α。

改下面「配置」里的 CSV_PATHS 即可：一个合并文件，或每人一个文件，数量不限。
同一题目、同一标注人若在多个文件里重复出现，保留 updated_at 最新的一条。

运行：
    python krippendorff_alpha.py
    python krippendorff_alpha.py a.csv b.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== 配置 ==============================
# 标注结果。写成列表，几个文件就放几条路径（相对本脚本，或绝对路径）。
CSV_PATHS = [
    Path(__file__).resolve().parent / "qg_annotations_2_users.csv",
    # Path(r"F:\data\annotator_b.csv"),
]

# 题目、标注人。按题型分组；不需要分组时把 GROUP_COLUMN 设为 None。
ITEM_COLUMN = "question_id"
RATER_COLUMN = "user_name"
GROUP_COLUMN = "question_type"

# 参与计算的评分列。None 表示自动用 DEFAULT_SCORE_COLUMNS 里实际存在的列。
SCORE_COLUMNS = None
DEFAULT_SCORE_COLUMNS = [
    "clarity",
    "esg_relevance",
    "answerability",
    "answer_correctness",
    "evidence_support",
    "question_type_fit",
    "type_specific_validity",
]

# 1–3 有序评分以 ordinal 为主；interval 一并给出，方便对照。
LEVELS = ["ordinal", "interval"]

# 同一题目同一人多行时，用该列保留最新记录。没有这一列就保留文件中的最后一行。
UPDATED_AT_COLUMN = "updated_at"

# 结果表。None 表示只打印、不写文件。
OUTPUT_CSV = Path(__file__).resolve().parent / "krippendorff_alpha_results.csv"
# =================================================================


def krippendorff_alpha(reliability_data: np.ndarray, level: str = "ordinal") -> float:
    """Krippendorff's α。

    reliability_data 形状为 (题目数, 标注人数)，缺失用 np.nan。
    level: nominal / ordinal / interval / ratio。
    少于两个不同分值、或没有任何可配对评分时返回 nan。
    """
    data = np.asarray(reliability_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("reliability_data 必须是二维数组 (题目 × 标注人)")

    observed = data[~np.isnan(data)]
    if observed.size == 0:
        return float("nan")
    categories = np.unique(observed)
    if categories.size < 2:
        return float("nan")

    # value_counts[unit, category]
    value_counts = (data[..., np.newaxis] == categories[np.newaxis, np.newaxis, :]).sum(axis=1)
    pairable = value_counts.sum(axis=1)
    usable = pairable >= 2
    if not np.any(usable):
        return float("nan")
    value_counts = value_counts[usable]
    pairable = pairable[usable]

    n_cat = categories.size
    diagonals = value_counts[:, np.newaxis, :] * np.eye(n_cat)[np.newaxis, ...]
    unnormalized = value_counts[..., np.newaxis] * value_counts[:, np.newaxis, :] - diagonals
    coincidence = (unnormalized / (pairable - 1).reshape((-1, 1, 1))).sum(axis=0)

    n_v = coincidence.sum(axis=1)
    n = float(n_v.sum())
    if n <= 1:
        return float("nan")
    expected = (np.outer(n_v, n_v) - np.diag(n_v)) / (n - 1.0)

    delta = _distance(categories, n_v, level)
    observed_disagreement = float((coincidence * delta).sum())
    expected_disagreement = float((expected * delta).sum())
    if expected_disagreement == 0:
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def _distance(categories: np.ndarray, n_v: np.ndarray, level: str) -> np.ndarray:
    n_cat = categories.size
    if level == "nominal":
        delta = np.ones((n_cat, n_cat), dtype=float)
        np.fill_diagonal(delta, 0.0)
        return delta
    if level == "interval":
        diff = categories.reshape(-1, 1) - categories.reshape(1, -1)
        return diff ** 2
    if level == "ratio":
        denom = categories.reshape(-1, 1) + categories.reshape(1, -1)
        numer = categories.reshape(-1, 1) - categories.reshape(1, -1)
        delta = np.zeros((n_cat, n_cat), dtype=float)
        np.divide(numer, denom, out=delta, where=denom != 0)
        return delta ** 2
    if level == "ordinal":
        delta = np.zeros((n_cat, n_cat), dtype=float)
        for i in range(n_cat):
            for j in range(i + 1, n_cat):
                span = n_v[i : j + 1].sum() - (n_v[i] + n_v[j]) / 2.0
                delta[i, j] = delta[j, i] = span ** 2
        return delta
    raise ValueError(f"不支持的测量水平: {level}（可选 nominal / ordinal / interval / ratio）")


def pairwise_agreement(reliability_data: np.ndarray) -> float:
    """两两完全一致的比例。只统计同一题目上至少有两人评分的配对。"""
    data = np.asarray(reliability_data, dtype=float)
    agree = 0
    pairs = 0
    for row in data:
        values = row[~np.isnan(row)]
        m = values.size
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                pairs += 1
                agree += int(values[i] == values[j])
    if pairs == 0:
        return float("nan")
    return agree / pairs


def load_annotations(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"找不到标注文件: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["_source_file"] = path.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    for column in (ITEM_COLUMN, RATER_COLUMN):
        if column not in data.columns:
            raise KeyError(f"CSV 缺少列 {column!r}。现有列: {list(data.columns)}")
    data = data.dropna(subset=[ITEM_COLUMN, RATER_COLUMN])
    data[ITEM_COLUMN] = data[ITEM_COLUMN].astype(str)
    data[RATER_COLUMN] = data[RATER_COLUMN].astype(str)
    if UPDATED_AT_COLUMN in data.columns:
        data["_updated_at"] = pd.to_datetime(data[UPDATED_AT_COLUMN], errors="coerce", utc=True)
        data = data.sort_values("_updated_at", kind="mergesort")
    data = data.drop_duplicates([ITEM_COLUMN, RATER_COLUMN], keep="last")
    return data


def score_columns(data: pd.DataFrame) -> list[str]:
    columns = list(SCORE_COLUMNS) if SCORE_COLUMNS else list(DEFAULT_SCORE_COLUMNS)
    present = [column for column in columns if column in data.columns]
    missing = [column for column in columns if column not in data.columns]
    if missing:
        print(f"跳过不存在的评分列: {', '.join(missing)}", file=sys.stderr)
    if not present:
        raise KeyError(f"没有可用的评分列。现有列: {list(data.columns)}")
    return present


def to_matrix(data: pd.DataFrame, column: str, raters: list[str]) -> np.ndarray:
    """题目 × 标注人。同一格只应有一个分数。"""
    scored = data[[ITEM_COLUMN, RATER_COLUMN, column]].copy()
    scored[column] = pd.to_numeric(scored[column], errors="coerce")
    wide = scored.pivot(index=ITEM_COLUMN, columns=RATER_COLUMN, values=column)
    wide = wide.reindex(columns=raters)
    return wide.to_numpy(dtype=float)


def summarize(matrix: np.ndarray, level: str) -> dict:
    n_ratings = np.sum(~np.isnan(matrix), axis=1)
    return {
        "alpha": krippendorff_alpha(matrix, level),
        "n_units": int(np.sum(n_ratings >= 2)),
        "n_units_single": int(np.sum(n_ratings == 1)),
        "agreement": pairwise_agreement(matrix),
    }


def collect_results(data: pd.DataFrame, columns: list[str], levels: list[str]) -> pd.DataFrame:
    raters = sorted(data[RATER_COLUMN].unique())
    rows = []

    def add(scope: str, group: str, dimension: str, matrix: np.ndarray) -> None:
        for level in levels:
            stats = summarize(matrix, level)
            rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "dimension": dimension,
                    "level": level,
                    "alpha": stats["alpha"],
                    "n_units": stats["n_units"],
                    "n_units_single_rater": stats["n_units_single"],
                    "pairwise_agreement": stats["agreement"],
                    "n_raters": len(raters),
                }
            )

    scopes: list[tuple[str, str, pd.DataFrame]] = [("all", "", data)]
    if GROUP_COLUMN and GROUP_COLUMN in data.columns:
        grouped = data.dropna(subset=[GROUP_COLUMN])
        for name, part in grouped.groupby(GROUP_COLUMN, sort=False):
            scopes.append(("by_group", str(name), part))
    elif GROUP_COLUMN:
        print(f"跳过分组：CSV 没有列 {GROUP_COLUMN!r}", file=sys.stderr)

    for scope, group, part in scopes:
        stacked = []
        for column in columns:
            matrix = to_matrix(part, column, raters)
            add(scope, group, column, matrix)
            stacked.append(matrix)
        if len(stacked) > 1:
            add(scope, group, "(all dimensions)", np.vstack(stacked))
    return pd.DataFrame(rows)


def print_report(results: pd.DataFrame, paths: list[Path], data: pd.DataFrame) -> None:
    raters = sorted(data[RATER_COLUMN].unique())
    print(f"文件 ({len(paths)}):")
    for path in paths:
        print(f"  {path}")
    print(f"标注人 ({len(raters)}): {', '.join(raters)}")
    print(f"题目数: {data[ITEM_COLUMN].nunique()}    评分记录: {len(data)}")
    print()
    print("Krippendorff 建议：α ≥ 0.800 可作可靠结论；0.667 ≤ α < 0.800 只适合尝试性结论。")
    print("pairwise_agreement 是两人分数完全相同的比例，不受分值分布偏斜的影响。")
    print()

    show = results.copy()
    show["alpha"] = show["alpha"].map(lambda v: "nan" if pd.isna(v) else f"{v:.4f}")
    show["pairwise_agreement"] = show["pairwise_agreement"].map(
        lambda v: "nan" if pd.isna(v) else f"{v:.1%}"
    )
    show["group"] = show["group"].replace("", "—")
    columns = ["scope", "group", "dimension", "level", "alpha", "n_units", "pairwise_agreement"]
    text = show[columns].to_string(index=False)
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从标注 CSV 计算 Krippendorff's α")
    parser.add_argument(
        "csv",
        nargs="*",
        help="标注 CSV 路径，可传任意多个。不传则使用脚本顶部的 CSV_PATHS。",
    )
    parser.add_argument(
        "--level",
        action="append",
        choices=["nominal", "ordinal", "interval", "ratio"],
        help="测量水平，可重复。不传则使用脚本顶部的 LEVELS。",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = [Path(p) for p in args.csv] if args.csv else list(CSV_PATHS)
    if not paths:
        raise SystemExit("CSV_PATHS 为空。请在脚本顶部填写路径，或在命令行传入 CSV。")
    levels = args.level or list(LEVELS)

    data = load_annotations(paths)
    columns = score_columns(data)
    results = collect_results(data, columns, levels)
    print_report(results, paths, data)

    if OUTPUT_CSV is not None:
        output = Path(OUTPUT_CSV)
        results.to_csv(output, index=False, encoding="utf-8-sig")
        print()
        print(f"已写入 {output}")


if __name__ == "__main__":
    main()
