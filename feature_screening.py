# -*- coding: utf-8 -*-
"""Jupyter: %run feature_screening.py

数值变量初筛：主键关联 -> 成熟标签 -> 自动单调分箱 -> IV 筛选。
依赖 pandas / numpy，兼容 Python 3.7。

分箱：最多 20 个等频预分箱，动态规划搜索相邻合并方案，同时比较递增、
递减方向及 1~n_bins 个普通箱。切点只用训练期样本拟合；若启用 OOT
（按 sample_datetime 时间切分），用 OOT 的 IV、单调方向和 PSI 选出
两边都稳的方案（目标为 min(训练 IV, OOT IV)），无可行方案时退回训练期
最优并标记不稳定。缺失值独立成箱，计入整体 IV，不参与单调性和最小箱
占比判断；缺失箱 IV 占比超过 max_missing_iv_share 的特征不入选。
最终按 IV（有 OOT 时取训练与 OOT 的较小值）>= min_iv、普通箱坏账率单调、
OOT 稳定且缺失贡献不超限筛选；入选后再按 WOE 相关性贪心去重
（dedup_corr），剔除与更高 IV 变量高度相关的变量。

筛选完成后可对最终候选特征批量生成分析图（默认开启）：整体分箱占比/
坏账率/WOE/IV 贡献、OOT 对照、分期趋势、WOE 热力图、分期占比与 PSI
来源、自动诊断。绘图与诊断只读取现有结果，不修改 selected、reason、
候选清单或分箱边界。绘图依赖 matplotlib（pip install matplotlib）。
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import textwrap
import re
import numpy as np
import pandas as pd


FEATURE_PREFIXES = (
    "bank_txn_surplus", "bank_txn_balance", "bank_txn_category",
    "bank_txn_income", "bank_txn_expense", "bank_txn_worthiness",
    "bank_txn_lender", "bank_txn_industry", "bank_txn_dishonour",
)
SMOOTHING = 0.5
TOL = 1e-12
STABLE_IV_RATIO = 0.5
DEFAULT_ROOT_DIR = (
    "/opt/workspace/aus_group_drive/Eliam/cdaml_bigq/data/"
    "wagego_new_user_credit_deal_20260801_20261023"
)

# 最近一次绘图运行的产出，便于 notebook 直接读取。
# 结构：{"run_dir": Path, "index": DataFrame, "diagnostics": DataFrame,
#        "time_detail": DataFrame}
LAST_ANALYSIS = None

_TRAIN_COLOR = "#2C7FB8"
_OOT_COLOR = "#E6A23C"
_MISSING_COLOR = "#8C8C8C"
_THRESHOLD_COLOR = "#C0392B"
_LOW_SAMPLE_HATCH = "////"
_BIN_PALETTE = (
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
    "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7",
)


def _check_binning_parameters(n_bins, n_prebins, min_bin_pct):
    for name, value in [("n_bins", n_bins), ("n_prebins", n_prebins)]:
        if not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(name + " 必须是至少为 2 的整数。")
    if not np.isfinite(min_bin_pct) or not 0 <= min_bin_pct <= 0.5:
        raise ValueError("min_bin_pct 必须在 [0, 0.5] 内。")


def _iv_terms(good, bad, total_good, total_bad, k):
    """与原脚本一致：每箱好坏样本各加 0.5，再分别归一化。"""
    good_pct = (good + SMOOTHING) / (total_good + SMOOTHING * k)
    bad_pct = (bad + SMOOTHING) / (total_bad + SMOOTHING * k)
    woe = np.log(good_pct / bad_pct)
    return good_pct, bad_pct, woe, (good_pct - bad_pct) * woe


def _prebin(values, labels, n_prebins):
    unique, inverse = np.unique(values, return_inverse=True)
    if len(unique) <= n_prebins:
        codes = inverse  # 低基数变量先按不同取值分箱。
    else:
        codes = np.asarray(pd.qcut(
            values, q=n_prebins, labels=False, duplicates="drop"
        ), dtype=int)
    n = np.bincount(codes)
    bad = np.bincount(codes, weights=labels, minlength=len(n))
    lower, upper = np.full(len(n), np.inf), np.full(len(n), -np.inf)
    np.minimum.at(lower, codes, values)
    np.maximum.at(upper, codes, values)
    nonempty = n > 0
    return n[nonempty], bad[nonempty], lower[nonempty], upper[nonempty]


def _best_partition(segment_iv, rates, valid, k, direction):
    """固定最终箱数、方向；状态为最后一个区间 [start, end]。

    在所有相邻预分箱合并方案中求最大 IV，而非贪心合箱。
    segment_iv 使用该最终箱数对应的平滑分母，保证搜索与报表口径一致。
    """
    m = len(rates)
    dp = np.full((m, m), -np.inf)
    dp[0] = np.where(valid[0], segment_iv[0], -np.inf)
    parents = []
    for step in range(2, k + 1):
        new_dp = np.full((m, m), -np.inf)
        parent = np.full((m, m), -1, dtype=int)
        for start in range(step - 1, m):
            # 上一箱必须在 start-1 结束，只比较其起点。
            previous_rate = rates[:start, start - 1, None]
            current_rate = rates[start, start:][None, :]
            if direction == "increasing":
                compatible = previous_rate <= current_rate + TOL
            else:
                compatible = previous_rate + TOL >= current_rate
            scores = np.where(
                compatible, dp[:start, start - 1, None], -np.inf
            )
            best_start = scores.argmax(axis=0)
            best_score = scores[best_start, np.arange(m - start)]
            feasible = valid[start, start:] & np.isfinite(best_score)
            new_dp[start, start:] = np.where(
                feasible, best_score + segment_iv[start, start:], -np.inf
            )
            parent[start, start:] = np.where(feasible, best_start, -1)
        dp = new_dp
        parents.append(parent)
    start = int(dp[:, m - 1].argmax())
    score = float(dp[start, m - 1])
    if not np.isfinite(score):
        return None, -np.inf
    segments, end = [], m - 1
    for step in range(k, 0, -1):
        segments.append((start, end))
        if step > 1:
            previous_start = int(parents[step - 2][start, end])
            end, start = start - 1, previous_start
    return segments[::-1], score


def _oot_partition_stats(x_oot, y_oot, segments, upper, has_missing):
    """OOT 复用训练期切点，返回每箱样本/坏样本、IV 和单调方向。"""
    missing = np.isnan(x_oot)
    index = np.searchsorted(
        [upper[end] for _, end in segments[:-1]], x_oot[~missing], side="left"
    )
    n = np.bincount(index, minlength=len(segments)).astype(float)
    bad = np.bincount(index, weights=y_oot[~missing], minlength=len(segments))
    missing_n = float(missing.sum()) if has_missing else 0.0
    missing_bad = float(y_oot[missing].sum()) if has_missing else 0.0
    if has_missing:
        all_good = np.r_[n - bad, missing_n - missing_bad]
        all_bad = np.r_[bad, missing_bad]
    else:
        all_good, all_bad = n - bad, bad
    total_good, total_bad = float(all_good.sum()), float(all_bad.sum())
    if total_good > 0 and total_bad > 0:
        iv = float(_iv_terms(
            all_good, all_bad, total_good, total_bad, len(all_good)
        )[3].sum())
    else:
        iv = np.nan
    rates = np.divide(bad, n, out=np.full(len(n), np.nan), where=n > 0)
    rates = rates[n > 0]
    return {
        "n": n, "bad": bad,
        "missing_n": missing_n, "missing_bad": missing_bad,
        "mapped_n": float(n.sum() + missing_n), "iv": iv,
        "increasing": bool(np.all(np.diff(rates) >= -TOL)),
        "decreasing": bool(np.all(np.diff(rates) <= TOL)),
    }


def _psi(train_counts, oot_counts, eps=1e-6):
    """按分箱占比计算训练集与 OOT 的 PSI。"""
    train_pct = np.asarray(train_counts, dtype=float)
    oot_pct = np.asarray(oot_counts, dtype=float)
    train_pct = np.clip(train_pct / train_pct.sum(), eps, None)
    oot_pct = np.clip(oot_pct / oot_pct.sum(), eps, None)
    return float(np.sum((train_pct - oot_pct) * np.log(train_pct / oot_pct)))


def _time_oot_mask(timestamps, ratio):
    """按时间切分，切点取最接近目标占比的时间点边界。"""
    values = timestamps.to_numpy()
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    position = boundaries[np.abs(boundaries - len(values) * (1 - ratio)).argmin()]
    mask = np.zeros(len(values), dtype=bool)
    mask[order[position:]] = True
    return mask


def _assign_woe(values, detail):
    """按分箱明细把原始值映射成 WOE；缺失值映射到缺失箱。"""
    ordinary = detail.loc[~detail["is_missing"]].sort_values("bin_order")
    upper = ordinary["upper"].to_numpy(dtype=float)
    index = np.searchsorted(upper[:-1], values.to_numpy(dtype=float), side="left")
    woe = ordinary["woe"].to_numpy()[index]
    missing = detail.loc[detail["is_missing"], "woe"]
    if len(missing):
        woe = np.where(values.isna().to_numpy(), float(missing.iloc[0]), woe)
    return woe


def auto_monotonic_binning(x, y, n_bins=5, n_prebins=20, min_bin_pct=0.05,
                           x_oot=None, y_oot=None, max_psi=0.25):
    """返回 (分箱明细, 指标字典)。x 须先将业务缺失默认值转成 NaN。

    n_bins: 最多普通箱数；缺失箱不占此额度，最终箱数自动选择。
    n_prebins: 最多预分箱数，越大候选切点越细、计算越慢。
    min_bin_pct: 每个普通箱至少占非缺失样本的比例；0 表示不限制。
    x_oot/y_oot: 时间外样本，只用训练期拟合切点，OOT 仅用于评估。
    max_psi: OOT 与训练集分箱占比的最大允许 PSI。
    有 OOT 时按 min(训练 IV, OOT IV) 选方案，要求方向一致且 PSI 不超限；
    稳定方案还需保留至少 STABLE_IV_RATIO 的训练期最优 IV，否则退回训练 IV
    最优并标记 is_stable=False。
    单调性按实际坏账率判断，允许相等，缺失箱除外。
    仅 1 个普通箱时免检单调性，允许由缺失/非缺失差异贡献 IV。
    全缺失变量不能入选。
    """
    _check_binning_parameters(n_bins, n_prebins, min_bin_pct)
    if not np.isfinite(max_psi) or max_psi < 0:
        raise ValueError("max_psi 必须为非负有限数值。")
    x = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="raise")
    x = x.replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="raise")
    if len(x) != len(y) or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("x/y 长度须一致，y 须同时包含 0 和 1 且没有空值。")
    if x_oot is not None or y_oot is not None:
        if x_oot is None or y_oot is None:
            raise ValueError("x_oot 与 y_oot 必须同时提供。")
        x_oot = pd.to_numeric(
            pd.Series(x_oot).reset_index(drop=True), errors="raise"
        ).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        y_oot = pd.to_numeric(pd.Series(y_oot).reset_index(drop=True), errors="raise")
        if (len(x_oot) != len(y_oot) or not y_oot.isin([0, 1]).all()
                or y_oot.nunique() != 2):
            raise ValueError("OOT x/y 长度须一致，y 须同时包含 0 和 1 且没有空值。")
        y_oot = y_oot.to_numpy(dtype=float)
    missing = x.isna().to_numpy()
    labels = y.to_numpy(dtype=float)
    total_bad, total_good = float(labels.sum()), float(len(y) - labels.sum())
    missing_n = int(missing.sum())
    missing_bad = float(labels[missing].sum())
    ordinary_n = len(x) - missing_n
    oot_enabled = x_oot is not None
    records = []
    best_iv, best_segments, best_oot = -np.inf, None, None
    best_stable, fallback = -np.inf, None
    chosen_k, chosen_direction = 0, None
    is_stable = None

    if ordinary_n:
        n, bad, lower, upper = _prebin(
            x.loc[~missing].to_numpy(dtype=float), labels[~missing], n_prebins
        )
        m = len(n)
        cum_n, cum_bad = np.r_[0, np.cumsum(n)], np.r_[0, np.cumsum(bad)]
        # [i, j] 表示从第 i 个到第 j 个预分箱合并的统计。
        segment_n = np.maximum(cum_n[1:][None, :] - cum_n[:-1, None], 0)
        segment_bad = np.maximum(cum_bad[1:][None, :] - cum_bad[:-1, None], 0)
        rates = np.divide(segment_bad, segment_n,
                          out=np.zeros_like(segment_bad), where=segment_n > 0)
        min_count = max(1, int(np.ceil(ordinary_n * min_bin_pct)))
        valid = np.triu(np.ones((m, m), dtype=bool)) & (segment_n >= min_count)
        for k in range(1, min(n_bins, m, ordinary_n // min_count) + 1):
            total_k = k + int(missing_n > 0)
            segment_iv = _iv_terms(
                segment_n - segment_bad, segment_bad, total_good, total_bad, total_k
            )[3]
            missing_iv = float(_iv_terms(
                missing_n - missing_bad, missing_bad, total_good, total_bad, total_k
            )[3]) if missing_n else 0.0
            directions = ("increasing",) if k == 1 else ("increasing", "decreasing")
            for direction in directions:
                segments, score = _best_partition(segment_iv, rates, valid, k, direction)
                if segments is None:
                    continue
                train_iv = score + missing_iv
                if not oot_enabled:
                    # 相同 IV 优先保留更少箱；不增加多余细分。
                    if train_iv > best_iv + TOL:
                        best_iv, best_segments = train_iv, segments
                    continue
                stats = _oot_partition_stats(
                    x_oot, y_oot, segments, upper, missing_n > 0)
                train_counts = np.r_[
                    [segment_n[s, e] for s, e in segments],
                    missing_n if missing_n else [],
                ]
                oot_counts = np.r_[
                    stats["n"], stats["missing_n"] if missing_n else [],
                ]
                stats["psi"] = _psi(train_counts, oot_counts)
                mono_ok = k == 1 or (
                    stats["increasing"] if direction == "increasing"
                    else stats["decreasing"]
                )
                feasible = bool(
                    stats["mapped_n"] > 0 and np.isfinite(stats["iv"])
                    and mono_ok and stats["psi"] <= max_psi
                )
                if fallback is None or train_iv > fallback[0] + TOL:
                    fallback = (train_iv, segments, stats, k, direction)
                if feasible:
                    key = min(train_iv, stats["iv"])
                    if key > best_stable + TOL:
                        best_stable = key
                        best_iv, best_segments, best_oot = train_iv, segments, stats
                        chosen_k, chosen_direction = k, direction
        if oot_enabled:
            # 稳定方案需保留至少 STABLE_IV_RATIO 的训练期最优 IV；
            # 否则（如单箱退化方案）退回训练期最优并标记不稳定。
            if (best_segments is None
                    or best_stable < STABLE_IV_RATIO * fallback[0] - TOL):
                (best_iv, best_segments, best_oot,
                 chosen_k, chosen_direction) = fallback
                is_stable = False
            else:
                is_stable = True

        cut_lower = -np.inf
        for order, (start, end) in enumerate(best_segments, 1):
            cut_upper = np.inf if end == m - 1 else float(upper[end])
            record = {
                "bin_order": order, "bin": "({:.17g}, {:.17g}]".format(cut_lower, cut_upper),
                "is_missing": False, "lower": cut_lower, "upper": cut_upper,
                "observed_min": float(lower[start]), "observed_max": float(upper[end]),
                "n": int(segment_n[start, end]), "bad": int(segment_bad[start, end]),
            }
            if best_oot is not None:
                record["oot_n"] = int(best_oot["n"][order - 1])
                record["oot_bad"] = int(best_oot["bad"][order - 1])
            records.append(record)
            cut_lower = cut_upper
    if missing_n:
        record = {
            "bin_order": len(records) + 1, "bin": "缺失", "is_missing": True,
            "lower": np.nan, "upper": np.nan,
            "observed_min": np.nan, "observed_max": np.nan,
            "n": missing_n, "bad": int(missing_bad),
        }
        if best_oot is not None:
            record["oot_n"] = int(best_oot["missing_n"])
            record["oot_bad"] = int(best_oot["missing_bad"])
        records.append(record)

    detail = pd.DataFrame(records)
    detail["good"] = detail["n"] - detail["bad"]
    if "oot_n" in detail:
        detail["oot_bad_rate"] = np.divide(
            detail["oot_bad"], detail["oot_n"],
            out=np.full(len(detail), np.nan), where=detail["oot_n"].to_numpy() > 0,
        )
    detail["sample_pct"] = detail["n"] / len(x)
    detail["bad_rate"] = detail["bad"] / detail["n"]
    good_pct, bad_pct, woe, iv_part = _iv_terms(
        detail["good"], detail["bad"], total_good, total_bad, len(detail)
    )
    detail["good_pct"], detail["bad_pct"] = good_pct, bad_pct
    detail["woe"], detail["iv_part"] = woe, iv_part
    ordinary_rates = detail.loc[~detail["is_missing"], "bad_rate"].to_numpy()
    increasing = bool(np.all(np.diff(ordinary_rates) >= -TOL))
    decreasing = bool(np.all(np.diff(ordinary_rates) <= TOL))
    monotonic = bool(ordinary_n and (increasing or decreasing))
    if not ordinary_n:
        direction = "全缺失"
    elif len(ordinary_rates) == 1:
        direction = "单普通箱，免检"
    elif increasing and decreasing:
        direction = "持平"
    else:
        direction = "递增" if increasing else "递减"
    iv = float(detail["iv_part"].sum())
    normal_mask = ~detail["is_missing"]
    iv_normal = float(detail.loc[normal_mask, "iv_part"].sum())
    iv_missing = float(detail.loc[~normal_mask, "iv_part"].sum())
    info = {
        "iv": iv, "iv_normal": iv_normal, "iv_missing": iv_missing,
        "iv_missing_share": iv_missing / iv if iv > 0 else 0.0,
        "missing_rate": missing_n / len(x), "n_bins": len(detail),
        "n_normal_bins": len(ordinary_rates), "is_monotonic": monotonic,
        "monotonic_direction": direction,
        "iv_min": iv, "oot_iv": np.nan, "psi": np.nan,
        "oot_is_monotonic": None, "is_stable": None,
    }
    if best_oot is not None:
        oot_iv = float(best_oot["iv"])
        info["oot_iv"] = oot_iv
        info["iv_min"] = min(iv, oot_iv) if np.isfinite(oot_iv) else np.nan
        info["psi"] = float(best_oot["psi"])
        info["oot_is_monotonic"] = bool(
            chosen_k <= 1
            or (best_oot["increasing"] if chosen_direction == "increasing"
                else best_oot["decreasing"])
        )
        info["is_stable"] = is_stable
    return detail, info


def _derive_obs_column(target, columns):
    """由标签列推出成熟标记列，如 y_fpd15_ovd -> y_fpd15_obs。"""
    matched = re.match(r"^(y_(?:fpd\d+|mob\d+))_", target)
    if matched:
        candidate = matched.group(1) + "_obs"
        if candidate in columns:
            return candidate
    return None


def _safe_filename(name, max_len=80):
    """把变量名转成安全的文件名片段。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name))
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return (cleaned or "feature")[:max_len]


def _fmt_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return ("{:." + str(digits) + "f}").format(float(value))


def _fmt_pct(value, digits=1):
    if value is None or not np.isfinite(value):
        return "NA"
    return ("{:." + str(digits) + "%}").format(float(value))


def _short_bin_label(lower, upper, is_missing, digits=4):
    """图内使用的短区间标签；完整切点保留在明细 CSV。"""
    if is_missing:
        return "缺失"
    fmt = "{:." + str(digits) + "g}"
    if lower == -np.inf:
        return "≤" + fmt.format(upper)
    if upper == np.inf:
        return ">" + fmt.format(lower)
    return "({}, {}]".format(fmt.format(lower), fmt.format(upper))


def _bin_labels(detail, with_order=True):
    """按 bin_order 排序，返回明细和展示标签。"""
    ordered = detail.sort_values("bin_order").reset_index(drop=True)
    labels = []
    for row in ordered.itertuples():
        text = _short_bin_label(row.lower, row.upper, bool(row.is_missing))
        labels.append("B{} {}".format(int(row.bin_order), text)
                      if with_order else text)
    return ordered, labels


def _wilson_interval(bad, n, z=1.96):
    """坏账率的 Wilson 95% 置信区间；n=0 时返回 NA。"""
    bad = np.asarray(bad, dtype=float)
    n = np.asarray(n, dtype=float)
    shape = np.broadcast(bad, n).shape
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(bad, n, out=np.full(shape, np.nan), where=n > 0)
        denom = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    low = np.where(n > 0, center - half, np.nan)
    high = np.where(n > 0, center + half, np.nan)
    return low, high


def _map_values_to_bins(values, detail):
    """按训练期分箱把值映射成 detail 行位置；-1 表示未覆盖。

    训练期没有缺失箱时，后续新出现的缺失值保持未覆盖（不并入普通箱）。
    """
    ordered = detail.sort_values("bin_order").reset_index(drop=True)
    values = np.asarray(values, dtype=float)
    mapped = np.full(len(values), -1, dtype=int)
    if not len(values):
        return mapped, ordered
    is_missing = ordered["is_missing"].to_numpy(dtype=bool)
    ordinary_pos = np.flatnonzero(~is_missing)
    missing_pos = np.flatnonzero(is_missing)
    if len(ordinary_pos):
        upper = ordered.loc[ordinary_pos, "upper"].to_numpy(dtype=float)
        non_missing = ~np.isnan(values)
        count = int(non_missing.sum())
        if count:
            if len(upper) == 1:
                index = np.zeros(count, dtype=int)
            else:
                index = np.searchsorted(
                    upper[:-1], values[non_missing], side="left")
            mapped[non_missing] = ordinary_pos[index]
    if len(missing_pos):
        mapped[np.isnan(values)] = missing_pos[0]
    return mapped, ordered


def _direction_from_rates(rates, tol=TOL):
    rates = np.asarray([r for r in rates if np.isfinite(r)], dtype=float)
    if len(rates) < 2:
        return "不足"
    increasing = bool(np.all(np.diff(rates) >= -tol))
    decreasing = bool(np.all(np.diff(rates) <= tol))
    if increasing and decreasing:
        return "持平"
    if increasing:
        return "递增"
    if decreasing:
        return "递减"
    return "非单调"


def _psi_components(reference, current, eps=1e-6):
    """与 _psi() 完全一致的口径，返回整体 PSI 和各箱贡献。"""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.sum() <= 0 or current.sum() <= 0:
        return np.nan, np.full(len(reference), np.nan)
    reference_pct = np.clip(reference / reference.sum(), eps, None)
    current_pct = np.clip(current / current.sum(), eps, None)
    contribution = (reference_pct - current_pct) * np.log(
        reference_pct / current_pct)
    return float(contribution.sum()), contribution


def _build_period_context(datetimes, labels, oot_mask, freq):
    """构建统一的时间分组；所有特征复用同一套时间段。"""
    if freq not in ("day", "week", "month"):
        raise ValueError("time_freq 只能是 day、week 或 month。")
    timestamps = pd.Series(pd.to_datetime(datetimes)).reset_index(drop=True)
    if freq == "day":
        start = timestamps.dt.floor("D")
        end = start.copy()
    elif freq == "week":
        day = timestamps.dt.floor("D")
        start = (day - pd.to_timedelta(day.dt.weekday, unit="D")).dt.floor("D")
        end = start + pd.Timedelta(days=6)
    else:
        period = timestamps.dt.to_period("M")
        start = period.dt.start_time.dt.floor("D")
        end = period.dt.end_time.dt.floor("D")
    raw = pd.DataFrame({
        "period_start": pd.to_datetime(start),
        "period_end": pd.to_datetime(end),
    })
    periods = (raw.drop_duplicates("period_start")
               .sort_values("period_start").reset_index(drop=True))
    if freq == "month":
        periods["period_label"] = periods["period_start"].dt.strftime("%Y-%m")
    else:
        periods["period_label"] = periods["period_start"].dt.strftime("%Y-%m-%d")
    code_map = {value: index for index, value in enumerate(
        periods["period_start"].tolist())}
    codes = np.array([code_map[value] for value in raw["period_start"].tolist()],
                     dtype=int)
    n_periods = len(periods)
    labels = np.asarray(labels, dtype=float)
    oot_mask = np.asarray(oot_mask, dtype=bool)
    if len(labels) != len(codes) or len(oot_mask) != len(codes):
        raise ValueError("时间分组与标签/OOT 掩码长度不一致。")
    n_total = np.bincount(codes, minlength=n_periods).astype(float)
    n_bad = np.bincount(codes, weights=labels, minlength=n_periods)
    n_good = n_total - n_bad
    n_oot = np.bincount(codes[oot_mask], minlength=n_periods).astype(float)
    n_train = n_total - n_oot
    split = np.where((n_train > 0) & (n_oot > 0), "跨界",
                     np.where(n_oot > 0, "OOT", "训练"))
    bad_rate = np.divide(n_bad, n_total, out=np.full(n_periods, np.nan),
                         where=n_total > 0)
    periods["n_total"] = n_total
    periods["n_train"] = n_train
    periods["n_oot"] = n_oot
    periods["n_bad"] = n_bad
    periods["n_good"] = n_good
    periods["bad_rate"] = bad_rate
    periods["split"] = split
    periods["both_classes"] = (n_good > 0) & (n_bad > 0)
    first_oot = int(np.flatnonzero(n_oot > 0)[0]) if (n_oot > 0).any() else None
    starts = periods["period_start"].tolist()
    gaps = []
    for index in range(1, len(starts)):
        previous, current = starts[index - 1], starts[index]
        if freq == "month":
            delta = ((current.year - previous.year) * 12
                     + current.month - previous.month)
            expected, unit = 1, "个月"
        else:
            delta = (current - previous).days
            expected, unit = (1, "天") if freq == "day" else (7, "天")
        if delta > expected:
            gaps.append({
                "previous": periods.loc[index - 1, "period_label"],
                "next": periods.loc[index, "period_label"],
                "gap": delta - expected, "unit": unit,
            })
    return {
        "periods": periods, "codes": codes, "n_periods": n_periods,
        "first_oot": first_oot, "gaps": gaps, "freq": freq,
    }


def _time_bin_matrices(values, labels, detail, period_context,
                       min_bin_n, min_bin_good, min_bin_bad, min_period_n):
    """按固定训练分箱计算分期样本、坏账率、WOE、IV 与 PSI。"""
    mapped, ordered = _map_values_to_bins(values, detail)
    bins = len(ordered)
    codes = period_context["codes"]
    n_periods = period_context["n_periods"]
    period_frame = period_context["periods"]
    valid = mapped >= 0
    flat_index = codes[valid] * bins + mapped[valid]
    bin_n = np.bincount(flat_index, minlength=n_periods * bins).reshape(
        n_periods, bins).astype(float)
    bin_bad = np.bincount(
        flat_index, weights=np.asarray(labels, dtype=float)[valid],
        minlength=n_periods * bins).reshape(n_periods, bins)
    bin_good = bin_n - bin_bad
    mapped_n = bin_n.sum(axis=1)
    total_n = np.bincount(codes, minlength=n_periods).astype(float)
    uncovered_n = total_n - mapped_n
    coverage = np.divide(mapped_n, total_n, out=np.full(n_periods, np.nan),
                         where=total_n > 0)
    mapped_bad = bin_bad.sum(axis=1)
    mapped_good = mapped_n - mapped_bad
    period_valid = (mapped_good > 0) & (mapped_bad > 0)
    bad_rate = np.divide(mapped_bad, mapped_n,
                         out=np.full(n_periods, np.nan), where=mapped_n > 0)
    sample_pct = np.divide(bin_n, mapped_n[:, None],
                           out=np.full((n_periods, bins), np.nan),
                           where=mapped_n[:, None] > 0)
    bin_bad_rate = np.divide(bin_bad, bin_n,
                             out=np.full((n_periods, bins), np.nan),
                             where=bin_n > 0)
    woe = np.full((n_periods, bins), np.nan)
    iv_part = np.full((n_periods, bins), np.nan)
    for index in range(n_periods):
        if not period_valid[index]:
            continue
        _, _, period_woe, period_iv = _iv_terms(
            bin_good[index], bin_bad[index],
            mapped_good[index], mapped_bad[index], bins)
        occupied = bin_n[index] > 0
        woe[index] = np.where(occupied, period_woe, np.nan)
        iv_part[index] = np.where(occupied, period_iv, np.nan)
    period_iv = np.nansum(iv_part, axis=1)
    period_iv[~period_valid] = np.nan
    period_psi = np.full(n_periods, np.nan)
    psi_contrib = np.full((n_periods, bins), np.nan)
    reference = ordered["n"].to_numpy(dtype=float)
    for index in range(n_periods):
        if mapped_n[index] <= 0:
            continue
        value, contribution = _psi_components(reference, bin_n[index])
        period_psi[index] = value
        psi_contrib[index] = contribution
    rows = n_periods * bins
    period_labels = np.repeat(period_frame["period_label"].to_numpy(), bins)
    period_start = np.repeat(
        period_frame["period_start"].dt.strftime("%Y-%m-%d").to_numpy(), bins)
    period_end = np.repeat(
        period_frame["period_end"].dt.strftime("%Y-%m-%d").to_numpy(), bins)
    period_split = np.repeat(period_frame["split"].to_numpy(), bins)
    bin_order = np.tile(ordered["bin_order"].to_numpy(), n_periods)
    bin_text = np.tile(ordered["bin"].to_numpy(), n_periods)
    is_missing = np.tile(ordered["is_missing"].to_numpy(dtype=bool), n_periods)
    reliability = []
    empty_flag = (bin_n == 0)
    low_n_flag = (bin_n > 0) & (bin_n < min_bin_n)
    zero_good_flag = (bin_n > 0) & (bin_good <= 0)
    zero_bad_flag = (bin_n > 0) & (bin_bad <= 0)
    low_good_flag = (bin_good > 0) & (bin_good < min_bin_good)
    low_bad_flag = (bin_bad > 0) & (bin_bad < min_bin_bad)
    invalid_period_flag = ~period_valid[:, None]
    low_period_flag = total_n[:, None] < min_period_n
    for index in range(rows):
        period_index, _ = divmod(index, bins)
        flags = []
        if empty_flag[period_index, index % bins]:
            flags.append("空箱")
        if invalid_period_flag[period_index, 0]:
            flags.append("期间单类")
        if low_period_flag[period_index, 0]:
            flags.append("低样本期")
        if low_n_flag[period_index, index % bins]:
            flags.append("箱样本少")
        if zero_good_flag[period_index, index % bins]:
            flags.append("无好样本")
        elif low_good_flag[period_index, index % bins]:
            flags.append("好样本少")
        if zero_bad_flag[period_index, index % bins]:
            flags.append("无坏样本")
        elif low_bad_flag[period_index, index % bins]:
            flags.append("坏样本少")
        reliability.append("; ".join(flags) if flags else "ok")
    reliability = np.asarray(reliability, dtype=object)
    long = pd.DataFrame({
        "period": period_labels,
        "period_start": period_start,
        "period_end": period_end,
        "period_split": period_split,
        "period_total_n": np.repeat(total_n, bins),
        "period_mapped_n": np.repeat(mapped_n, bins),
        "period_uncovered_n": np.repeat(uncovered_n, bins),
        "period_coverage": np.repeat(coverage, bins),
        "period_bad": np.repeat(mapped_bad, bins),
        "period_good": np.repeat(mapped_good, bins),
        "period_bad_rate": np.repeat(bad_rate, bins),
        "period_both_classes": np.repeat(period_valid, bins),
        "period_iv": np.repeat(period_iv, bins),
        "period_psi": np.repeat(period_psi, bins),
        "bin_order": bin_order,
        "bin": bin_text,
        "is_missing": is_missing,
        "bin_n": bin_n.ravel(),
        "bin_bad": bin_bad.ravel(),
        "bin_good": bin_good.ravel(),
        "sample_pct": sample_pct.ravel(),
        "bad_rate": bin_bad_rate.ravel(),
        "woe": woe.ravel(),
        "iv_part": iv_part.ravel(),
        "psi_contrib": psi_contrib.ravel(),
        "reliability_flags": reliability,
    })
    matrices = {
        "bin_n": bin_n, "bin_bad": bin_bad, "bin_good": bin_good,
        "mapped_n": mapped_n, "total_n": total_n, "uncovered_n": uncovered_n,
        "coverage": coverage, "bad_rate": bad_rate, "sample_pct": sample_pct,
        "woe": woe, "iv_part": iv_part, "period_iv": period_iv,
        "period_psi": period_psi, "psi_contrib": psi_contrib,
        "period_valid": period_valid, "low_n": low_n_flag,
        "zero_good": zero_good_flag, "zero_bad": zero_bad_flag,
        "low_good": low_good_flag, "low_bad": low_bad_flag,
        "empty": empty_flag, "low_period": total_n < min_period_n,
        "ordered_detail": ordered,
    }
    return matrices, long


def _overall_table(detail, info, train, oot, target):
    """整理整体分箱表；OOT WOE/IV 用原 _iv_terms 补充计算。"""
    ordered, labels = _bin_labels(detail)
    table = ordered.copy()
    table["bin_label"] = labels
    table["ci_low"], table["ci_high"] = _wilson_interval(
        table["bad"].to_numpy(dtype=float), table["n"].to_numpy(dtype=float))
    meta = {
        "oot_enabled": oot is not None,
        "train_total": len(train),
        "train_bad_rate": float(train[target].mean()),
        "oot_total": len(oot) if oot is not None else 0,
        "oot_bad_rate": float(oot[target].mean()) if oot is not None else np.nan,
        "oot_iv_recomputed": np.nan,
        "oot_iv_check": np.nan,
        "oot_mapped": 0.0,
        "oot_uncovered": 0.0,
    }
    if oot is not None:
        oot_n = table["oot_n"].to_numpy(dtype=float)
        oot_bad = table["oot_bad"].to_numpy(dtype=float)
        oot_good = oot_n - oot_bad
        total_good = float(oot_good.sum())
        total_bad = float(oot_bad.sum())
        table["oot_good"] = oot_good
        table["oot_bad_rate"] = np.divide(
            oot_bad, oot_n, out=np.full(len(table), np.nan),
            where=oot_n > 0)
        table["oot_ci_low"], table["oot_ci_high"] = _wilson_interval(
            oot_bad, oot_n)
        if total_good > 0 and total_bad > 0:
            _, _, oot_woe, oot_iv_part = _iv_terms(
                oot_good, oot_bad, total_good, total_bad, len(table))
            table["oot_woe"] = oot_woe
            table["oot_iv_part"] = oot_iv_part
            meta["oot_iv_recomputed"] = float(np.sum(oot_iv_part))
            if np.isfinite(info.get("oot_iv", np.nan)):
                meta["oot_iv_check"] = abs(
                    meta["oot_iv_recomputed"] - float(info["oot_iv"]))
        else:
            table["oot_woe"] = np.nan
            table["oot_iv_part"] = np.nan
        meta["oot_mapped"] = float(oot_n.sum())
        meta["oot_uncovered"] = float(meta["oot_total"] - meta["oot_mapped"])
        table["oot_sample_pct"] = np.divide(
            oot_n, float(meta["oot_total"]),
            out=np.full(len(table), np.nan),
            where=meta["oot_total"] > 0)
    return table, meta


def _adverse_steps(rates, ns, bads, goods, direction, tol,
                   min_bin_n, min_bin_good, min_bin_bad,
                   jump_threshold, z=1.96):
    """按训练方向找出反向跳点；区分可靠反转与证据不足。"""
    rates = np.asarray(rates, dtype=float)
    ns = np.asarray(ns, dtype=float)
    bads = np.asarray(bads, dtype=float)
    goods = np.asarray(goods, dtype=float)
    if direction not in ("递增", "递减") or len(rates) < 2:
        return []
    low, high = _wilson_interval(bads, ns)
    steps = []
    for index in range(len(rates) - 1):
        if not (np.isfinite(rates[index]) and np.isfinite(rates[index + 1])):
            continue
        change = rates[index + 1] - rates[index]
        adverse = (change < -tol) if direction == "递增" else (change > tol)
        if not adverse:
            continue
        reliable = bool(
            ns[index] >= min_bin_n and ns[index + 1] >= min_bin_n
            and bads[index] >= min_bin_bad and bads[index + 1] >= min_bin_bad
            and goods[index] >= min_bin_good
            and goods[index + 1] >= min_bin_good)
        separated = bool(
            np.isfinite(low[index]) and np.isfinite(high[index + 1])
            and (low[index] > high[index + 1]
                 or low[index + 1] > high[index]))
        magnitude = abs(change)
        steps.append({
            "index": index, "magnitude": magnitude,
            "rate_left": rates[index], "rate_right": rates[index + 1],
            "n_left": ns[index], "n_right": ns[index + 1],
            "bad_left": bads[index], "bad_right": bads[index + 1],
            "reliable": reliable, "ci_separated": separated,
            "clear": bool(reliable and separated and magnitude >= tol),
            "jump": bool(reliable and separated
                         and magnitude >= jump_threshold),
        })
    return steps


def _prepare_feature_analysis(feature, export_name, detail, info, data,
                              target, train, oot, period_context, params):
    """计算单个特征的整体/分期统计和诊断，不修改任何筛选字段。"""
    overall, overall_meta = _overall_table(detail, info, train, oot, target)
    matrices, time_long = _time_bin_matrices(
        data[feature].to_numpy(dtype=float),
        data[target].to_numpy(dtype=float),
        detail, period_context,
        params["diag_min_bin_n"], params["diag_min_bin_good"],
        params["diag_min_bin_bad"], params["diag_min_period_n"])
    diagnostics = _diagnose_feature(
        feature, export_name, overall, overall_meta, info, matrices,
        period_context, train, oot, target, params)
    return {
        "feature": feature, "export_name": export_name,
        "overall": overall, "overall_meta": overall_meta,
        "matrices": matrices, "time_long": time_long,
        "diagnostics": diagnostics, "period_context": period_context,
        "train": train, "oot": oot,
    }


def _diagnose_feature(feature, export_name, overall, overall_meta, info,
                      matrices, period_context, train, oot, target, params):
    """生成结构化诊断和保留/观察/建议复查结论。"""
    severe, warn, notes = [], [], []
    oot_enabled = overall_meta["oot_enabled"]
    iv_train = float(info.get("iv", np.nan))
    iv_oot = float(info.get("oot_iv", np.nan)) if oot_enabled else np.nan
    iv_min = float(info.get("iv_min", np.nan))
    iv_floor = params["diag_iv_floor"]
    min_bin_n = params["diag_min_bin_n"]
    min_bin_good = params["diag_min_bin_good"]
    min_bin_bad = params["diag_min_bin_bad"]
    max_missing_share = params["max_missing_iv_share"]
    max_psi = params["max_psi"]
    train_dir = str(info.get("monotonic_direction", "NA"))
    train_n = int(overall_meta["train_total"])
    oot_total = int(overall_meta["oot_total"])
    n_counts = overall["n"].to_numpy(dtype=float)
    n_bads = overall["bad"].to_numpy(dtype=float)
    n_goods = n_counts - n_bads
    iv_parts = overall["iv_part"].to_numpy(dtype=float)
    missing_share = float(info.get("iv_missing_share", np.nan))
    max_bin = None
    max_share = np.nan
    if np.isfinite(iv_train) and iv_train > iv_floor:
        max_index = int(np.nanargmax(iv_parts))
        max_bin = overall.iloc[max_index]
        max_share = float(iv_parts[max_index] / iv_train)
    max_bin_small = bool(max_bin is not None
                         and float(max_bin["n"]) < min_bin_n)
    shares = (iv_parts / iv_train) if (np.isfinite(iv_train)
                                       and iv_train > iv_floor) else np.full(
        len(iv_parts), np.nan)
    small_mask = (n_counts > 0) & (n_counts < min_bin_n) & np.isfinite(shares) \
        & (shares >= params["diag_small_bin_iv_share"])
    small_concentrated = bool(np.any(small_mask))

    n_low_bins = int(np.sum((n_counts > 0) & (n_counts < min_bin_n)))
    n_zero_good_bins = int(np.sum((n_counts > 0) & (n_goods <= 0)))
    n_zero_bad_bins = int(np.sum((n_counts > 0) & (n_bads <= 0)))
    n_normal_bins = int((~overall["is_missing"]).sum())

    train_mapped = float(n_counts.sum())
    oot_mapped = float(overall_meta["oot_mapped"])
    coverage_all = ((train_mapped + oot_mapped)
                    / float(train_n + oot_total)) if (train_n + oot_total) else np.nan
    coverage_oot = (oot_mapped / oot_total) if oot_total else np.nan
    uncovered_oot = float(overall_meta["oot_uncovered"])
    period_coverages = matrices["coverage"]
    min_period_coverage = (
        float(np.nanmin(period_coverages))
        if np.isfinite(period_coverages).any() else np.nan)
    uncovered_total = float(np.nansum(matrices["uncovered_n"]))

    periods = period_context["periods"]
    n_periods = period_context["n_periods"]
    n_low_periods = int((periods["n_total"] < params["diag_min_period_n"]).sum())
    n_invalid_periods = int(
        ((periods["n_total"] > 0) & (~periods["both_classes"])).sum())
    n_cross_periods = int((periods["split"] == "跨界").sum())
    valid_period_mask = matrices["mapped_n"] > 0
    n_empty_cells = int(np.sum((matrices["bin_n"] == 0)
                               & valid_period_mask[:, None]))
    n_low_cells = int(np.sum(matrices["low_n"]))
    n_zero_good_cells = int(np.sum(matrices["zero_good"]))
    n_zero_bad_cells = int(np.sum(matrices["zero_bad"]))

    period_psi = matrices["period_psi"]
    period_n = periods["n_total"].to_numpy(dtype=float)
    reliable_period = period_n >= params["diag_min_period_n"]
    max_period_psi = (float(np.nanmax(period_psi))
                      if np.isfinite(period_psi).any() else np.nan)
    max_period_psi_label = "NA"
    if np.isfinite(max_period_psi):
        max_period_psi_label = str(
            periods.loc[int(np.nanargmax(period_psi)), "period_label"])
    reliable_psi = np.where(reliable_period, period_psi, np.nan)
    max_reliable_period_psi = (float(np.nanmax(reliable_psi))
                               if np.isfinite(reliable_psi).any() else np.nan)
    max_reliable_period_psi_label = "NA"
    if np.isfinite(max_reliable_period_psi):
        max_reliable_period_psi_label = str(
            periods.loc[int(np.nanargmax(reliable_psi)), "period_label"])
    psi_contrib = matrices["psi_contrib"]
    max_psi_contrib_value = np.nan
    max_psi_contrib_bin = "NA"
    if np.isfinite(psi_contrib).any():
        flat_index = int(np.nanargmax(psi_contrib))
        period_index, bin_index = divmod(flat_index, len(overall))
        max_psi_contrib_value = float(psi_contrib[period_index, bin_index])
        max_psi_contrib_bin = "{} / {}".format(
            periods.loc[period_index, "period_label"],
            overall.iloc[bin_index]["bin_label"])
    reliable_contrib = np.where(reliable_period[:, None], psi_contrib, np.nan)
    max_reliable_psi_contrib_value = np.nan
    max_reliable_psi_contrib_bin = "NA"
    if np.isfinite(reliable_contrib).any():
        flat_index = int(np.nanargmax(reliable_contrib))
        period_index, bin_index = divmod(flat_index, len(overall))
        max_reliable_psi_contrib_value = float(
            reliable_contrib[period_index, bin_index])
        max_reliable_psi_contrib_bin = "{} / {}".format(
            periods.loc[period_index, "period_label"],
            overall.iloc[bin_index]["bin_label"])
    psi_sum_diff = np.nan
    for index in range(n_periods):
        if np.isfinite(period_psi[index]):
            total = np.nansum(psi_contrib[index])
            diff = abs(total - period_psi[index])
            if not np.isfinite(psi_sum_diff) or diff > psi_sum_diff:
                psi_sum_diff = diff

    oot_dir = "NA"
    direction_consistent = None
    if oot_enabled:
        ordinary = overall.loc[~overall["is_missing"]]
        oot_rates = ordinary["oot_bad_rate"].to_numpy(dtype=float)
        oot_ns = ordinary["oot_n"].to_numpy(dtype=float)
        oot_dir = _direction_from_rates(oot_rates[oot_ns > 0])
        if train_dir in ("递增", "递减") and oot_dir in ("递增", "递减"):
            direction_consistent = (train_dir == oot_dir)

    ordered_detail = matrices["ordered_detail"]
    ordinary_mask = ~ordered_detail["is_missing"].to_numpy(dtype=bool)
    period_steps = []
    for index in range(n_periods):
        if not matrices["period_valid"][index]:
            continue
        rates = np.full(len(overall), np.nan)
        occupied = matrices["bin_n"][index] > 0
        rates[occupied] = (matrices["bin_bad"][index, occupied]
                           / matrices["bin_n"][index, occupied])
        steps = _adverse_steps(
            rates[ordinary_mask], matrices["bin_n"][index, ordinary_mask],
            matrices["bin_bad"][index, ordinary_mask],
            matrices["bin_good"][index, ordinary_mask], train_dir,
            params["diag_direction_tol"], min_bin_n, min_bin_good,
            min_bin_bad, params["diag_jump_bad_rate"])
        for step in steps:
            step["period"] = str(periods.loc[index, "period_label"])
            step["bin_left"] = str(
                overall.iloc[np.flatnonzero(ordinary_mask)[step["index"]]][
                    "bin_label"])
            step["bin_right"] = str(
                overall.iloc[np.flatnonzero(ordinary_mask)[step["index"] + 1]][
                    "bin_label"])
        period_steps.extend(steps)
    reversal_periods = sorted(set(step["period"] for step in period_steps))
    clear_steps = [step for step in period_steps if step["clear"]]
    jump_steps = [step for step in period_steps if step["jump"]]
    oot_steps = []
    if oot_enabled:
        ordinary = overall.loc[~overall["is_missing"]]
        oot_ns = ordinary["oot_n"].to_numpy(dtype=float)
        oot_bads = ordinary["oot_bad"].to_numpy(dtype=float)
        oot_steps = _adverse_steps(
            ordinary["oot_bad_rate"].to_numpy(dtype=float), oot_ns, oot_bads,
            oot_ns - oot_bads, train_dir, params["diag_direction_tol"],
            min_bin_n, min_bin_good, min_bin_bad,
            params["diag_jump_bad_rate"])
    oot_clear = [step for step in oot_steps if step["clear"]]
    oot_jumps = [step for step in oot_steps if step["jump"]]

    if oot_enabled:
        notes.append("OOT 已参与分箱方案选择，不是独立验证集")
    else:
        warn.append("未启用 OOT，缺少时间外验证")
    notes.append("训练集单调性由分箱算法约束，不作为额外验证证据")
    notes.append("时间分析固定使用训练期分箱边界，未重新拟合")
    notes.append("诊断阈值仅用于报告提示，不改变筛选结果")

    if direction_consistent is False:
        severe.append("OOT 风险方向与训练相反（训练{}，OOT{}）".format(
            train_dir, oot_dir))
    if oot_enabled and np.isfinite(iv_oot) and iv_train > iv_floor:
        drop = (iv_train - iv_oot) / iv_train
        if drop > 0.5:
            severe.append("OOT IV 较训练下降 {:.1%}（{:.4f}→{:.4f}）".format(
                drop, iv_train, iv_oot))
        elif drop > 0.25:
            warn.append("OOT IV 较训练下降 {:.1%}（{:.4f}→{:.4f}）".format(
                drop, iv_train, iv_oot))
    if np.isfinite(coverage_all) and coverage_all < params["diag_review_coverage"]:
        severe.append(
            "数据覆盖率 {:.1%} 低于复查阈值 {:.0%}（未覆盖 {} 条，"
            "其中 OOT 未覆盖 {} 条）".format(
                coverage_all, params["diag_review_coverage"],
                int(round(uncovered_total)), int(round(uncovered_oot))))
    elif np.isfinite(coverage_all) and coverage_all < params["diag_min_coverage"]:
        warn.append("数据覆盖率 {:.1%} 低于提示阈值 {:.0%}".format(
            coverage_all, params["diag_min_coverage"]))
    if uncovered_oot > 0:
        warn.append(
            "OOT 有 {} 条缺失值未被训练分箱覆盖（OOT 覆盖率 {:.1%}），"
            "已排除出 OOT IV/PSI".format(
                int(round(uncovered_oot)), coverage_oot))
    if max_missing_share is not None and np.isfinite(missing_share) \
            and missing_share > max_missing_share + TOL:
        severe.append("缺失箱 IV 占比 {:.1%} 超过筛选阈值 {:.0%}".format(
            missing_share, max_missing_share))
    elif np.isfinite(missing_share) and missing_share >= 0.3:
        warn.append("缺失箱 IV 占比 {:.1%}，风险区分较依赖缺失".format(
            missing_share))
    if iv_train <= iv_floor:
        warn.append("训练 IV={:.4f} 接近 0，IV 贡献比例不稳定".format(iv_train))
    if n_normal_bins <= 1:
        warn.append("仅 {} 个普通箱，无法判断数值型风险趋势".format(
            n_normal_bins))
    if n_low_bins:
        warn.append("训练期有 {} 个箱样本量低于诊断阈值 {}".format(
            n_low_bins, min_bin_n))
    if n_zero_good_bins or n_zero_bad_bins:
        warn.append(
            "训练期有 {} 个箱无好样本、{} 个箱无坏样本；"
            "平滑后 WOE 可计算但证据不足".format(
                n_zero_good_bins, n_zero_bad_bins))
    if max_bin is not None and np.isfinite(max_share):
        if max_bin_small and max_share >= params["diag_max_bin_iv_share"]:
            severe.append(
                "最大 IV 贡献来自小样本箱 {}（n={}，占训练 {:.2%}，"
                "贡献 {:.1%}）".format(
                    max_bin["bin_label"], int(max_bin["n"]),
                    float(max_bin["n"]) / max(train_n, 1), max_share))
        elif max_share >= params["diag_max_bin_iv_share"]:
            warn.append(
                "最大单箱 IV 贡献 {:.1%} 集中在 {}（n={}，占训练 {:.2%}）".format(
                    max_share, max_bin["bin_label"], int(max_bin["n"]),
                    float(max_bin["n"]) / max(train_n, 1)))
    if small_concentrated and not (
            max_bin is not None and max_bin_small
            and np.isfinite(max_share)
            and max_share >= params["diag_max_bin_iv_share"]):
        warn.append("存在小样本箱贡献集中（箱样本 < {} 且 IV 贡献 ≥ {:.0%}）".format(
            min_bin_n, params["diag_small_bin_iv_share"]))
    if n_periods <= 1:
        warn.append("仅 {} 个时间段，无法评估时间稳定性".format(n_periods))
    if n_invalid_periods:
        labels = periods.loc[
            (periods["n_total"] > 0) & (~periods["both_classes"]),
            "period_label"].tolist()
        warn.append("{} 个期间只有单一标签（{}），WOE/IV 记 NA".format(
            n_invalid_periods, "、".join(labels[:3])
            + ("等" if len(labels) > 3 else "")))
    if n_low_periods:
        min_n = int(periods["n_total"].min())
        warn.append("{} 个期间样本量低于诊断阈值 {}（最小 {}）".format(
            n_low_periods, params["diag_min_period_n"], min_n))
    if clear_steps:
        worst = max(clear_steps, key=lambda item: item["magnitude"])
        severe.append(
            "期间出现明显方向反转 {} 次，最大一次在 {}（{}→{}，"
            "坏账率 {:.2%}→{:.2%}，n={}/{}）".format(
                len(clear_steps), worst["period"], worst["bin_left"],
                worst["bin_right"], worst["rate_left"], worst["rate_right"],
                int(worst["n_left"]), int(worst["n_right"])))
    elif period_steps:
        worst = max(period_steps, key=lambda item: item["magnitude"])
        warn.append(
            "期间存在 {} 次局部方向波动（未达明显反转标准），最大在 {}"
            "（{:.2%}→{:.2%}，n={}/{}）".format(
                len(period_steps), worst["period"], worst["rate_left"],
                worst["rate_right"], int(worst["n_left"]),
                int(worst["n_right"])))
    if jump_steps:
        worst = max(jump_steps, key=lambda item: item["magnitude"])
        warn.append(
            "期间有 {} 个可靠异常跳点，最大在 {}（{}→{}，变化 {:.2%}）".format(
                len(jump_steps), worst["period"], worst["bin_left"],
                worst["bin_right"], worst["magnitude"]))
    if oot_clear:
        worst = max(oot_clear, key=lambda item: item["magnitude"])
        severe.append(
            "OOT 整体出现明显方向反转（B{}→B{}，坏账率 {:.2%}→{:.2%}，"
            "n={}/{}）".format(
                worst["index"] + 1, worst["index"] + 2, worst["rate_left"],
                worst["rate_right"], int(worst["n_left"]),
                int(worst["n_right"])))
    elif oot_jumps:
        warn.append("OOT 整体存在 {} 个可靠异常跳点".format(len(oot_jumps)))
    if np.isfinite(max_reliable_period_psi):
        if max_reliable_period_psi > params["diag_psi_review"]:
            severe.append(
                "可靠期间 PSI 最大 {:.4f}（{}，n≥{}）超过复查阈值 {:.4f}".format(
                    max_reliable_period_psi, max_reliable_period_psi_label,
                    params["diag_min_period_n"], params["diag_psi_review"]))
        elif max_reliable_period_psi > max_psi:
            warn.append(
                "可靠期间 PSI 最大 {:.4f}（{}，n≥{}）超过筛选阈值 {:.4f}".format(
                    max_reliable_period_psi, max_reliable_period_psi_label,
                    params["diag_min_period_n"], max_psi))
    else:
        warn.append(
            "没有样本量达到 {} 的期间，期间 PSI 不参与稳定性判定".format(
                params["diag_min_period_n"]))
    if np.isfinite(max_period_psi) and (
            not np.isfinite(max_reliable_period_psi)
            or max_period_psi > max_reliable_period_psi + TOL):
        raw_index = int(np.nanargmax(period_psi))
        warn.append(
            "全期间 PSI 最大 {:.4f}（{}，n={}）来自低样本期间，"
            "不参与严重判定".format(
                max_period_psi, max_period_psi_label,
                int(period_n[raw_index])))
    if n_empty_cells:
        notes.append("分期空箱 {} 个，对应 WOE/占比显示为 NA".format(n_empty_cells))
    if n_low_cells or n_zero_good_cells or n_zero_bad_cells:
        notes.append(
            "分期低样本箱 {} 个、零好样本箱 {} 个、零坏样本箱 {} 个；"
            "热力图中用斜线标记".format(
                n_low_cells, n_zero_good_cells, n_zero_bad_cells))
    if period_context["gaps"]:
        gap = max(period_context["gaps"], key=lambda item: item["gap"])
        notes.append("时间存在 {} 个间隔，最大间隔 {} {}{}".format(
            len(period_context["gaps"]), gap["gap"], gap["unit"],
            "（{} → {}）".format(gap["previous"], gap["next"])))
    if n_cross_periods:
        notes.append("{} 个期间跨越训练/OOT 切点，按跨界期间标注".format(
            n_cross_periods))
    if info.get("is_stable") is False:
        severe.append("分箱方案在 OOT 评估中不稳定（已由筛选层标记）")

    if severe:
        conclusion = "建议复查"
    elif warn:
        conclusion = "观察"
    else:
        conclusion = "保留"

    if max_bin is not None and np.isfinite(max_share):
        max_bin_text = "{}（n={}，占训练 {:.2%}，IV 贡献 {:.1%}）".format(
            max_bin["bin_label"], int(max_bin["n"]),
            float(max_bin["n"]) / max(train_n, 1), max_share)
    else:
        max_bin_text = "NA（IV≈0）"
    base_evidence = (
        "训练 IV={}；OOT IV={}；iv_min={}；最大单箱贡献 {}；"
        "缺失箱 IV 占比 {}；可靠期间 PSI 最大 {}（{}）；"
        "方向训练{} / OOT{}；覆盖率 {}；期间数 {}").format(
            _fmt_float(iv_train), _fmt_float(iv_oot),
            _fmt_float(iv_min), max_bin_text,
            _fmt_pct(missing_share), _fmt_float(max_reliable_period_psi),
            max_reliable_period_psi_label, train_dir, oot_dir,
            _fmt_pct(coverage_all), n_periods)
    evidence_parts = [base_evidence] + severe + warn
    evidence = "；".join(evidence_parts)
    if len(evidence) > 1600:
        evidence = evidence[:1597] + "..."
    warnings_text = "；".join(warn)
    severe_text = "；".join(severe)
    limitations = "；".join(notes)
    thresholds = (
        "箱n≥{}；好样本≥{}；坏样本≥{}；期间n≥{}；覆盖率提示≥{:.0%}；"
        "覆盖率复查≥{:.0%}；单箱贡献提示≥{:.0%}；小样本贡献≥{:.0%}；"
        "方向容差{:.1%}；跳点{:.1%}；期间PSI仅统计n≥{}的期间；"
        "期间PSI复查>{:.4f}").format(
            min_bin_n, min_bin_good, min_bin_bad,
            params["diag_min_period_n"], params["diag_min_coverage"],
            params["diag_review_coverage"], params["diag_max_bin_iv_share"],
            params["diag_small_bin_iv_share"], params["diag_direction_tol"],
            params["diag_jump_bad_rate"], params["diag_min_period_n"],
            params["diag_psi_review"])
    iv_change_pct = (
        (iv_oot / iv_train - 1.0) * 100.0
        if (np.isfinite(iv_oot) and np.isfinite(iv_train)
            and iv_train > iv_floor) else np.nan)
    return {
        "conclusion": conclusion, "evidence": evidence,
        "severe_issues": severe_text, "warnings": warnings_text,
        "limitations": limitations, "thresholds": thresholds,
        "iv_train": iv_train, "iv_oot": iv_oot, "iv_min": iv_min,
        "iv_change_pct": iv_change_pct,
        "iv_missing": info.get("iv_missing", np.nan),
        "iv_missing_share": missing_share,
        "n_normal_bins": n_normal_bins,
        "max_iv_bin": max_bin["bin_label"] if max_bin is not None else "NA",
        "max_iv_bin_n": int(max_bin["n"]) if max_bin is not None else 0,
        "max_iv_share": max_share,
        "max_iv_bin_small": max_bin_small,
        "small_bin_concentrated": small_concentrated,
        "n_low_bins": n_low_bins,
        "n_zero_good_bins": n_zero_good_bins,
        "n_zero_bad_bins": n_zero_bad_bins,
        "train_direction": train_dir, "oot_direction": oot_dir,
        "direction_consistent": direction_consistent,
        "n_periods": n_periods, "n_cross_periods": n_cross_periods,
        "n_low_periods": n_low_periods,
        "n_invalid_periods": n_invalid_periods,
        "n_empty_cells": n_empty_cells,
        "n_low_cells": n_low_cells,
        "n_zero_good_cells": n_zero_good_cells,
        "n_zero_bad_cells": n_zero_bad_cells,
        "n_reversal_periods": len(reversal_periods),
        "n_clear_reversal_periods": len(set(
            step["period"] for step in clear_steps)),
        "n_jump_steps": len(jump_steps),
        "max_period_psi": max_period_psi,
        "max_period_psi_label": max_period_psi_label,
        "max_reliable_period_psi": max_reliable_period_psi,
        "max_reliable_period_psi_label": max_reliable_period_psi_label,
        "max_psi_contrib_bin": max_psi_contrib_bin,
        "max_psi_contrib_value": max_psi_contrib_value,
        "max_reliable_psi_contrib_bin": max_reliable_psi_contrib_bin,
        "max_reliable_psi_contrib_value": max_reliable_psi_contrib_value,
        "psi_sum_max_diff": psi_sum_diff,
        "coverage_all": coverage_all, "coverage_oot": coverage_oot,
        "min_period_coverage": min_period_coverage,
        "uncovered_total": uncovered_total,
        "uncovered_oot": uncovered_oot,
        "date_gap_count": len(period_context["gaps"]),
        "oot_iv_recomputed": overall_meta["oot_iv_recomputed"],
        "oot_iv_check": overall_meta["oot_iv_check"],
        "time_freq": period_context["freq"],
    }


def _bin_color(index, is_missing):
    if is_missing:
        return _MISSING_COLOR
    return _BIN_PALETTE[index % len(_BIN_PALETTE)]


def _period_tick_positions(count, max_labels=14):
    if count <= max_labels:
        return np.arange(count)
    step = int(np.ceil(count / float(max_labels)))
    positions = np.arange(0, count, step)
    if positions[-1] != count - 1:
        positions = np.append(positions, count - 1)
    return positions


def _draw_period_overlay(ax, periods, first_oot):
    cross = periods["split"].to_numpy() == "跨界"
    for index in np.flatnonzero(cross):
        ax.axvspan(index - 0.5, index + 0.5, color="#9E9E9E",
                   alpha=0.16, zorder=2)
    if first_oot is not None:
        ax.axvline(first_oot - 0.5, color="#333333", ls="--",
                   lw=1.2, zorder=4)


def _rate_yerr(rate, low, high):
    rate = np.asarray(rate, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    return np.vstack([np.maximum(rate - low, 0.0),
                      np.maximum(high - rate, 0.0)])


def _fmt_n(value):
    return "{:.0f}".format(value) if np.isfinite(value) else "NA"


def _text_display_width(text):
    return float(sum(2.0 if ord(char) > 127 else 1.0 for char in str(text)))


def _fmt_rate(value):
    return "{:.1%}".format(value) if np.isfinite(value) else "NA"


def _plot_share_panel(ax, ax_table, ordered, meta, info, short_labels,
                      min_bin_n):
    oot_enabled = meta["oot_enabled"]
    count = len(ordered)
    x = np.arange(count)
    width = 0.38 if oot_enabled else 0.68
    train_total = max(meta["train_total"], 1)
    share_train = ordered["n"].to_numpy(dtype=float) / train_total * 100.0
    bars_train = ax.bar(
        x - (width / 2.0 if oot_enabled else 0.0), share_train, width,
        color=_TRAIN_COLOR, alpha=0.82, edgecolor="white",
        linewidth=0.6, label="训练占比", zorder=2)
    bars_oot = None
    if oot_enabled:
        oot_total = max(meta["oot_total"], 1)
        share_oot = ordered["oot_n"].to_numpy(dtype=float) / oot_total * 100.0
        bars_oot = ax.bar(
            x + width / 2.0, share_oot, width, color=_OOT_COLOR,
            alpha=0.72, edgecolor="white", linewidth=0.6,
            label="OOT 占比", zorder=2)
    is_missing = ordered["is_missing"].to_numpy(dtype=bool)
    for index in range(count):
        for bars, values in [(bars_train, ordered["n"].to_numpy(dtype=float)),
                             (bars_oot,
                              ordered["oot_n"].to_numpy(dtype=float)
                              if oot_enabled else None)]:
            if bars is None:
                continue
            patch = bars[index]
            if is_missing[index]:
                patch.set_facecolor(_MISSING_COLOR)
                patch.set_hatch("//")
            if values is not None and 0 < values[index] < min_bin_n:
                patch.set_edgecolor(_THRESHOLD_COLOR)
                patch.set_linewidth(1.3)
    ax.set_ylabel("样本占比 (%)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=7)
    upper = np.nanmax(share_train) if np.isfinite(share_train).any() else 10.0
    if oot_enabled:
        oot_share = (ordered["oot_n"].to_numpy(dtype=float)
                     / max(meta["oot_total"], 1) * 100.0)
        if np.isfinite(oot_share).any():
            upper = max(upper, float(np.nanmax(oot_share)))
    ax.set_ylim(0, max(upper * 1.28, 5.0))
    ax.set_title("分箱占比与坏账率（训练为主，OOT 对照）"
                 if oot_enabled else "分箱占比与坏账率（仅训练）",
                 fontsize=9, loc="left")
    ax.text(0.01, 0.98, "柱=样本占比；线=坏账率（95% Wilson CI）",
            transform=ax.transAxes, fontsize=6.8, color="#555555",
            va="top")
    ax2 = ax.twinx()
    ordinary = ~is_missing
    train_rate = ordered["bad_rate"].to_numpy(dtype=float) * 100.0
    valid = ordinary & np.isfinite(train_rate)
    ax2.plot(x[valid], train_rate[valid], color=_TRAIN_COLOR,
             marker="o", lw=1.7, label="训练坏账率", zorder=3)
    ax2.errorbar(
        x[valid], train_rate[valid],
        yerr=_rate_yerr(train_rate[valid],
                        ordered["ci_low"].to_numpy(dtype=float)[valid] * 100.0,
                        ordered["ci_high"].to_numpy(dtype=float)[valid] * 100.0),
        fmt="none", ecolor=_TRAIN_COLOR, elinewidth=1, capsize=2.5,
        alpha=0.8, zorder=3)
    missing_mask = is_missing & np.isfinite(train_rate)
    if missing_mask.any():
        ax2.scatter(x[missing_mask], train_rate[missing_mask],
                    color=_TRAIN_COLOR, marker="D", s=26, zorder=4,
                    label="训练缺失箱")
    if oot_enabled:
        oot_rate = ordered["oot_bad_rate"].to_numpy(dtype=float) * 100.0
        oot_valid = ordinary & np.isfinite(oot_rate)
        ax2.plot(x[oot_valid], oot_rate[oot_valid], color=_OOT_COLOR,
                 marker="s", lw=1.4, ls="--", label="OOT 坏账率", zorder=3)
        ax2.errorbar(
            x[oot_valid], oot_rate[oot_valid],
            yerr=_rate_yerr(
                oot_rate[oot_valid],
                ordered["oot_ci_low"].to_numpy(dtype=float)[oot_valid] * 100.0,
                ordered["oot_ci_high"].to_numpy(dtype=float)[oot_valid] * 100.0),
            fmt="none", ecolor=_OOT_COLOR, elinewidth=0.9, capsize=2.2,
            alpha=0.75, zorder=3)
        oot_missing = is_missing & np.isfinite(oot_rate)
        if oot_missing.any():
            ax2.scatter(x[oot_missing], oot_rate[oot_missing],
                        color=_OOT_COLOR, marker="D", s=22, zorder=4,
                        label="OOT 缺失箱")
    ax2.axhline(meta["train_bad_rate"] * 100.0, color=_TRAIN_COLOR,
                ls=":", lw=1.0, alpha=0.7, label="训练整体坏账率")
    if oot_enabled:
        ax2.axhline(meta["oot_bad_rate"] * 100.0, color=_OOT_COLOR,
                    ls=":", lw=1.0, alpha=0.7, label="OOT 整体坏账率")
    ax2.set_ylabel("坏账率 (%)", fontsize=8)
    max_rate = np.nanmax(train_rate) if np.isfinite(train_rate).any() else 10.0
    if oot_enabled:
        oot_rate_all = ordered["oot_bad_rate"].to_numpy(dtype=float) * 100.0
        if np.isfinite(oot_rate_all).any():
            max_rate = max(max_rate, float(np.nanmax(oot_rate_all)))
    ax2.set_ylim(0, max(max_rate * 1.35, 5.0))
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=6.4,
              loc="upper right", ncol=2, framealpha=0.88)

    ax_table.axis("off")
    row_labels, cell_text = [], []
    if oot_enabled:
        row_labels = ["训练 n", "训练 坏", "训练 坏账率",
                      "OOT n", "OOT 坏", "OOT 坏账率"]
        cell_text = [
            [_fmt_n(value) for value in ordered["n"]],
            [_fmt_n(value) for value in ordered["bad"]],
            [_fmt_rate(value) for value in ordered["bad_rate"]],
            [_fmt_n(value) for value in ordered["oot_n"]],
            [_fmt_n(value) for value in ordered["oot_bad"]],
            [_fmt_rate(value) for value in ordered["oot_bad_rate"]],
        ]
    else:
        row_labels = ["训练 n", "训练 坏", "训练 坏账率"]
        cell_text = [
            [_fmt_n(value) for value in ordered["n"]],
            [_fmt_n(value) for value in ordered["bad"]],
            [_fmt_rate(value) for value in ordered["bad_rate"]],
        ]
    table = ax_table.table(cellText=cell_text, rowLabels=row_labels,
                           colLabels=short_labels, loc="center",
                           cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5.6 if count > 5 else 6.2)
    table.scale(1.0, 1.15)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#EDEDED")
            cell.set_text_props(weight="bold")
        if col >= 0 and is_missing[col]:
            cell.set_facecolor("#F2F2F2")
    ax_table.set_title("分箱明细（n=样本量，坏=坏样本量）",
                       fontsize=7.2, pad=1.5)


def _plot_woe_panel(ax, ordered, meta, short_labels, min_bin_n):
    count = len(ordered)
    x = np.arange(count)
    width = 0.38 if meta["oot_enabled"] else 0.68
    train_woe = ordered["woe"].to_numpy(dtype=float)
    bars_train = ax.bar(
        x - (width / 2.0 if meta["oot_enabled"] else 0.0), train_woe,
        width, color=_TRAIN_COLOR, alpha=0.82, label="训练 WOE", zorder=2)
    bars_oot = None
    if meta["oot_enabled"]:
        bars_oot = ax.bar(
            x + width / 2.0, ordered["oot_woe"].to_numpy(dtype=float),
            width, color=_OOT_COLOR, alpha=0.75, label="OOT WOE", zorder=2)
    is_missing = ordered["is_missing"].to_numpy(dtype=bool)
    train_n = ordered["n"].to_numpy(dtype=float)
    oot_n = (ordered["oot_n"].to_numpy(dtype=float)
             if meta["oot_enabled"] else np.zeros(count))
    for index in range(count):
        for bars, values in [(bars_train, train_n), (bars_oot, oot_n)]:
            if bars is None:
                continue
            patch = bars[index]
            if is_missing[index]:
                patch.set_facecolor(_MISSING_COLOR)
                patch.set_hatch("//")
            if 0 < values[index] < min_bin_n:
                patch.set_edgecolor(_THRESHOLD_COLOR)
                patch.set_linewidth(1.2)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("WOE", fontsize=8)
    ax.set_title("分箱 WOE（正=相对风险低，负=相对风险高）",
                 fontsize=9, loc="left")
    ax.text(0.01, 0.98, "WOE=ln(好样本占比/坏样本占比)",
            transform=ax.transAxes, fontsize=6.8, color="#555555",
            va="top")
    if count <= 8:
        for index in range(count):
            value = train_woe[index]
            if np.isfinite(value):
                ax.text(index - (width / 2.0 if meta["oot_enabled"] else 0.0),
                        value, "{:.2f}".format(value), ha="center",
                        va="bottom" if value >= 0 else "top",
                        fontsize=6)
            if meta["oot_enabled"]:
                value = ordered["oot_woe"].to_numpy(dtype=float)[index]
                if np.isfinite(value):
                    ax.text(index + width / 2.0, value,
                            "{:.2f}".format(value), ha="center",
                            va="bottom" if value >= 0 else "top",
                            fontsize=6)
    ax.legend(fontsize=6.8, loc="upper right")
    values = np.r_[train_woe, ordered["oot_woe"].to_numpy(dtype=float)] \
        if meta["oot_enabled"] else train_woe
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(finite.max()) * 1.35, 0.5) if len(finite) else 1.0
    ax.set_ylim(-limit, limit)


def _plot_iv_panel(ax, ordered, meta, info, short_labels, min_bin_n,
                   iv_floor):
    count = len(ordered)
    x = np.arange(count)
    width = 0.38 if meta["oot_enabled"] else 0.68
    train_iv = ordered["iv_part"].to_numpy(dtype=float)
    bars_train = ax.bar(
        x - (width / 2.0 if meta["oot_enabled"] else 0.0), train_iv,
        width, color=_TRAIN_COLOR, alpha=0.82, label="训练 IV 贡献", zorder=2)
    bars_oot = None
    if meta["oot_enabled"]:
        bars_oot = ax.bar(
            x + width / 2.0, ordered["oot_iv_part"].to_numpy(dtype=float),
            width, color=_OOT_COLOR, alpha=0.75, label="OOT IV 贡献", zorder=2)
    is_missing = ordered["is_missing"].to_numpy(dtype=bool)
    train_n = ordered["n"].to_numpy(dtype=float)
    oot_n = (ordered["oot_n"].to_numpy(dtype=float)
             if meta["oot_enabled"] else np.zeros(count))
    for index in range(count):
        for bars, values in [(bars_train, train_n), (bars_oot, oot_n)]:
            if bars is None:
                continue
            patch = bars[index]
            if is_missing[index]:
                patch.set_facecolor(_MISSING_COLOR)
                patch.set_hatch("//")
            if 0 < values[index] < min_bin_n:
                patch.set_edgecolor(_THRESHOLD_COLOR)
                patch.set_linewidth(1.2)
    iv_train = float(info.get("iv", np.nan))
    if np.isfinite(iv_train) and iv_train > iv_floor:
        max_index = int(np.nanargmax(train_iv))
        bars_train[max_index].set_edgecolor(_THRESHOLD_COLOR)
        bars_train[max_index].set_linewidth(2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("IV 贡献", fontsize=8)
    ax.set_title("IV 贡献（训练 vs OOT）" if meta["oot_enabled"]
                 else "IV 贡献（仅训练，未启用 OOT）",
                 fontsize=9, loc="left")
    caption = "训练 IV={}；OOT IV={}；缺失箱占比 {}".format(
        _fmt_float(info.get("iv", np.nan)),
        _fmt_float(info.get("oot_iv", np.nan)),
        _fmt_pct(info.get("iv_missing_share", np.nan)))
    if np.isfinite(iv_train) and iv_train > iv_floor:
        max_index = int(np.nanargmax(train_iv))
        caption += "\n最大贡献箱 {}（n={}，占训练 IV {:.1%}）".format(
            short_labels[max_index], int(train_n[max_index]),
            train_iv[max_index] / iv_train)
        if 0 < train_n[max_index] < min_bin_n:
            caption += "；该箱样本少，贡献集中需谨慎"
    ax.text(0.01, 0.98, caption, transform=ax.transAxes, fontsize=6.6,
            color="#555555", va="top", linespacing=1.35)
    if np.isfinite(iv_train) and iv_train > iv_floor:
        if count <= 8:
            for index in range(count):
                value = train_iv[index]
                share = value / iv_train
                ax.text(
                    index - (width / 2.0 if meta["oot_enabled"] else 0.0),
                    value, "{:.1%}".format(share), ha="center",
                    va="bottom", fontsize=6)
    else:
        ax.text(0.5, 0.55, "IV≈0，贡献比例不稳定，不展示占比",
                transform=ax.transAxes, ha="center", fontsize=8,
                color=_THRESHOLD_COLOR)
    ax.legend(fontsize=6.8, loc="upper right")
    values = np.r_[train_iv, ordered["oot_iv_part"].to_numpy(dtype=float)] \
        if meta["oot_enabled"] else train_iv
    finite = values[np.isfinite(values)]
    upper = float(finite.max()) * 1.35 if len(finite) else 1.0
    ax.set_ylim(0, max(upper, 1e-6))


def _plot_time_trend(axes, periods, matrices, info, period_context,
                     max_psi, min_period_n, tick_labels):
    n_periods = len(periods)
    x = np.arange(n_periods)
    ax_n, ax_bad, ax_iv, ax_psi = axes
    oot_enabled = period_context["first_oot"] is not None
    for axis in axes:
        _draw_period_overlay(axis, periods, period_context["first_oot"])
    ax_n.bar(x, periods["n_train"].to_numpy(dtype=float),
             color=_TRAIN_COLOR, alpha=0.82, label="训练", zorder=2)
    if oot_enabled:
        ax_n.bar(x, periods["n_oot"].to_numpy(dtype=float),
                 bottom=periods["n_train"].to_numpy(dtype=float),
                 color=_OOT_COLOR, alpha=0.8, label="OOT", zorder=2)
    low_period = periods["n_total"].to_numpy(dtype=float) < min_period_n
    if low_period.any():
        ax_n.scatter(x[low_period],
                     periods["n_total"].to_numpy(dtype=float)[low_period],
                     marker="v", s=24, color=_THRESHOLD_COLOR, zorder=4,
                     label="低样本期")
    ax_n.set_ylabel("样本量", fontsize=7.5)
    ax_n.set_title(
        "分期趋势（固定训练分箱）：样本量 / 坏账率 / IV / PSI",
        fontsize=9, loc="left")
    ax_n.legend(fontsize=6.2, loc="upper right", ncol=3)
    if period_context["first_oot"] is not None:
        ax_n.text(period_context["first_oot"] - 0.5,
                  ax_n.get_ylim()[1] * 0.96, " 训练/OOT 切点",
                  fontsize=6.6, va="top", color="#333333")

    bad_rate = matrices["bad_rate"] * 100.0
    mapped_n = matrices["mapped_n"]
    mapped_bad = matrices["bin_bad"].sum(axis=1)
    low, high = _wilson_interval(mapped_bad, mapped_n)
    valid = periods["both_classes"].to_numpy(dtype=bool) & np.isfinite(bad_rate)
    ax_bad.plot(x[valid], bad_rate[valid], color="#4B4B4B",
                marker="o", lw=1.4, label="分期坏账率", zorder=3)
    if valid.any():
        ax_bad.errorbar(
            x[valid], bad_rate[valid],
            yerr=_rate_yerr(bad_rate[valid] / 100.0, low[valid],
                            high[valid]) * 100.0,
            fmt="none", ecolor="#4B4B4B", elinewidth=0.9, capsize=2.2,
            alpha=0.75, zorder=3)
    invalid = ~periods["both_classes"].to_numpy(dtype=bool)
    if invalid.any():
        ax_bad.scatter(x[invalid], matrices["bad_rate"][invalid] * 100.0,
                       marker="x", s=28, color=_THRESHOLD_COLOR, zorder=4,
                       label="单一标签期")
    ax_bad.axhline(info.get("_train_bad_rate", np.nan) * 100.0,
                   color=_TRAIN_COLOR, ls=":", lw=1.0, label="训练整体")
    if period_context["first_oot"] is not None:
        oot_mask = periods["split"].to_numpy() != "训练"
        if oot_mask.any():
            oot_bad = (matrices["bin_bad"].sum(axis=1)[oot_mask].sum()
                       / max(matrices["mapped_n"][oot_mask].sum(), 1))
            ax_bad.axhline(oot_bad * 100.0, color=_OOT_COLOR,
                           ls=":", lw=1.0, label="OOT 整体")
    ax_bad.set_ylabel("坏账率 (%)", fontsize=7.5)
    ax_bad.legend(fontsize=6.0, loc="upper right", ncol=3)

    period_iv = matrices["period_iv"]
    ax_iv.bar(x, np.where(np.isfinite(period_iv), period_iv, 0.0),
              color="#6A51A3", alpha=0.78, label="分期 IV", zorder=2)
    if np.isfinite(info.get("iv", np.nan)):
        ax_iv.axhline(info["iv"], color=_TRAIN_COLOR, ls="--", lw=1.0,
                      label="训练整体 IV")
    invalid_iv = ~np.isfinite(period_iv)
    if invalid_iv.any():
        ax_iv.scatter(x[invalid_iv], np.zeros(int(invalid_iv.sum())),
                      marker="x", s=24, color=_THRESHOLD_COLOR, zorder=4,
                      label="IV=NA")
    ax_iv.set_ylabel("IV", fontsize=7.5)
    ax_iv.legend(fontsize=6.0, loc="upper right", ncol=3)

    period_psi = matrices["period_psi"]
    low_period = matrices["low_period"]
    reliable_psi = np.where(low_period, np.nan, period_psi)
    ax_psi.plot(x, reliable_psi, color="#6A51A3", marker="o", lw=1.4,
                label="PSI（可靠期间，相对训练）", zorder=3)
    low_finite = low_period & np.isfinite(period_psi)
    if low_finite.any():
        ax_psi.scatter(x[low_finite], period_psi[low_finite], marker="^",
                       s=34, facecolors="none", edgecolors=_THRESHOLD_COLOR,
                       linewidths=1.0, zorder=5,
                       label="低样本期 PSI（不参与判定）")
    ax_psi.axhline(max_psi, color=_THRESHOLD_COLOR, ls="--", lw=1.0,
                   label="筛选阈值 {:g}".format(max_psi))
    if np.isfinite(reliable_psi).any():
        index = int(np.nanargmax(reliable_psi))
        ax_psi.scatter([x[index]], [reliable_psi[index]], s=38,
                       facecolors="none", edgecolors=_THRESHOLD_COLOR,
                       zorder=4, label="可靠期间最大 PSI")
        upper = max(float(np.nanmax(reliable_psi)) * 1.35, max_psi * 1.2,
                    1e-6)
    else:
        finite_all = period_psi[np.isfinite(period_psi)]
        upper = (max(float(finite_all.max()) * 1.35, max_psi * 1.2, 1e-6)
                 if len(finite_all) else 1.0)
        ax_psi.text(0.5, 0.55,
                    "无样本量达到诊断阈值的期间\nPSI 仅作图参考，不参与判定",
                    transform=ax_psi.transAxes, ha="center", va="center",
                    fontsize=6.4, color=_THRESHOLD_COLOR)
    for index in np.flatnonzero(low_finite):
        value = period_psi[index]
        if value > upper:
            ax_psi.annotate("{:.3g}".format(value),
                            xy=(x[index], upper * 0.96),
                            xytext=(x[index], upper * 0.82), ha="center",
                            fontsize=5.6, color=_THRESHOLD_COLOR,
                            arrowprops=dict(arrowstyle="-|>",
                                            color=_THRESHOLD_COLOR, lw=0.6))
    ax_psi.set_ylabel("PSI", fontsize=7.5)
    ax_psi.legend(fontsize=5.8, loc="upper right", ncol=2)
    ax_psi.set_ylim(0, upper)

    positions = _period_tick_positions(n_periods, 14)
    ax_psi.set_xticks(x[positions])
    ax_psi.set_xticklabels([tick_labels[index] for index in positions],
                           rotation=45, ha="right", fontsize=6.2)
    for axis in axes[:-1]:
        axis.set_xticks([])
    for axis in axes:
        axis.set_xlim(-0.6, n_periods - 0.4)


def _plot_woe_heatmap(ax, ordered, matrices, periods, period_context,
                      short_labels, min_bin_n, plt):
    from matplotlib.patches import Patch, Rectangle
    woe = matrices["woe"].T
    masked = np.ma.masked_invalid(woe)
    finite = np.abs(woe[np.isfinite(woe)])
    limit = max(float(finite.max()) * 1.05, 1e-3) if len(finite) else 1.0
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#D9D9D9")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-limit,
                      vmax=limit, interpolation="nearest", zorder=1)
    bins, n_periods = woe.shape
    ax.set_yticks(np.arange(bins))
    ax.set_yticklabels(short_labels, fontsize=7)
    positions = _period_tick_positions(n_periods, 12)
    labels = []
    for index in range(n_periods):
        label = str(periods.loc[index, "period_label"])
        if periods.loc[index, "split"] == "跨界":
            label += "*"
        labels.append(label)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[index] for index in positions],
                       rotation=45, ha="right", fontsize=6.2)
    _draw_period_overlay(ax, periods, period_context["first_oot"])
    unreliable = (matrices["low_n"] | matrices["zero_good"]
                  | matrices["zero_bad"] | matrices["low_good"]
                  | matrices["low_bad"]).T
    empty = matrices["empty"].T
    valid_period = matrices["period_valid"]
    for row in range(bins):
        for column in range(n_periods):
            if unreliable[row, column] and not empty[row, column] \
                    and valid_period[column]:
                ax.add_patch(Rectangle(
                    (column - 0.5, row - 0.5), 1, 1, fill=False,
                    hatch=_LOW_SAMPLE_HATCH, edgecolor="#444444",
                    linewidth=0.0, zorder=3))
    if bins * n_periods <= 90:
        for row in range(bins):
            for column in range(n_periods):
                value = woe[row, column]
                if np.isfinite(value):
                    ax.text(column, row, "{:.2f}".format(value),
                            ha="center", va="center", fontsize=5.6,
                            color="white" if abs(value) > 0.62 * limit
                            else "#222222", zorder=4)
    ax.set_title("分期 WOE 热力图（固定训练分箱）", fontsize=9, loc="left")
    ax.text(1.0, -0.19, "正=低风险（蓝）；负=高风险（红）；灰=空箱/期间单类；"
            "斜线=低样本或零好/零坏", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color="#555555")
    ax.legend(handles=[Patch(facecolor="none", edgecolor="#444444",
                             hatch=_LOW_SAMPLE_HATCH,
                             label="低样本/零好/零坏")],
              fontsize=6.0, loc="upper right")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    colorbar.set_label("WOE", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def _plot_share_heatmap(ax, matrices, periods, period_context, short_labels,
                        plt):
    share = matrices["sample_pct"].T
    masked = np.ma.masked_where(matrices["bin_n"].T == 0, share)
    finite = share[np.isfinite(share)]
    upper = max(float(finite.max()) * 1.05, 1e-3) if len(finite) else 1.0
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#D9D9D9")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0,
                      vmax=upper, interpolation="nearest", zorder=1)
    bins, n_periods = share.shape
    ax.set_yticks(np.arange(bins))
    ax.set_yticklabels(short_labels, fontsize=7)
    positions = _period_tick_positions(n_periods, 12)
    labels = []
    for index in range(n_periods):
        label = str(periods.loc[index, "period_label"])
        if periods.loc[index, "split"] == "跨界":
            label += "*"
        labels.append(label)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[index] for index in positions],
                       rotation=45, ha="right", fontsize=6.2)
    _draw_period_overlay(ax, periods, period_context["first_oot"])
    if bins * n_periods <= 90:
        for row in range(bins):
            for column in range(n_periods):
                value = share[row, column]
                if np.isfinite(value):
                    ax.text(column, row, "{:.0%}".format(value),
                            ha="center", va="center", fontsize=5.6,
                            color="white" if value > 0.55 * upper
                            else "#222222", zorder=4)
    ax.set_title("分期样本占比（分母=当期已映射样本）",
                 fontsize=9, loc="left")
    ax.text(1.0, -0.19, "灰=空箱；* = 跨越训练/OOT 切点的期间；"
            "覆盖不足见下方诊断与明细 CSV", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color="#555555")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    colorbar.set_label("样本占比", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def _plot_psi_panel(ax, ordered, matrices, periods, period_context,
                    short_labels, max_psi):
    n_periods = len(periods)
    x = np.arange(n_periods)
    is_missing = ordered["is_missing"].to_numpy(dtype=bool)
    low_period = matrices["low_period"]
    bottom = np.zeros(n_periods)
    for index in range(len(ordered)):
        values = matrices["psi_contrib"][:, index]
        values = np.where(np.isfinite(values) & ~low_period, values, 0.0)
        ax.bar(x, values, bottom=bottom,
               color=_bin_color(index, is_missing[index]),
               edgecolor="white", linewidth=0.35,
               label=short_labels[index], zorder=2)
        bottom += values
    period_psi = matrices["period_psi"]
    reliable_psi = np.where(low_period, np.nan, period_psi)
    ax.plot(x, reliable_psi, color="#222222", marker="o", lw=1.4,
            label="整体 PSI（可靠期间）", zorder=4)
    low_finite = low_period & np.isfinite(period_psi)
    if low_finite.any():
        ax.scatter(x[low_finite], period_psi[low_finite], marker="^", s=42,
                   facecolors="none", edgecolors=_THRESHOLD_COLOR,
                   linewidths=1.2, zorder=6,
                   label="低样本期 PSI（不参与判定）")
    ax.axhline(max_psi, color=_THRESHOLD_COLOR, ls="--", lw=1.0,
               label="筛选阈值 {:g}".format(max_psi), zorder=3)
    if np.isfinite(reliable_psi).any():
        index = int(np.nanargmax(reliable_psi))
        ax.scatter([x[index]], [reliable_psi[index]], s=48,
                   facecolors="none", edgecolors="#222222", zorder=5)
        if n_periods <= 14:
            ax.text(x[index], reliable_psi[index], " 可靠最大 {:.4f}".format(
                reliable_psi[index]), fontsize=6, va="bottom")
        upper = max(float(np.nanmax(reliable_psi)) * 1.35, max_psi * 1.2,
                    1e-6)
    else:
        finite_all = period_psi[np.isfinite(period_psi)]
        upper = (max(float(finite_all.max()) * 1.35, max_psi * 1.2, 1e-6)
                 if len(finite_all) else 1.0)
        ax.text(0.5, 0.55,
                "无样本量达到诊断阈值的期间\nPSI 仅作图参考，不参与判定",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.0, color=_THRESHOLD_COLOR)
    if n_periods <= 14:
        for index in range(n_periods):
            if np.isfinite(period_psi[index]) and not low_period[index]:
                ax.text(index, bottom[index], "{:.3f}".format(
                    period_psi[index]), ha="center", va="bottom",
                    fontsize=5.6, color="#333333")
    for index in np.flatnonzero(low_finite):
        value = period_psi[index]
        if value > upper:
            ax.annotate("{:.3g}".format(value), xy=(x[index], upper * 0.96),
                        xytext=(x[index], upper * 0.82), ha="center",
                        fontsize=5.6, color=_THRESHOLD_COLOR,
                        arrowprops=dict(arrowstyle="-|>",
                                        color=_THRESHOLD_COLOR, lw=0.6))
    positions = _period_tick_positions(n_periods, 16)
    labels = [str(periods.loc[index, "period_label"]) for index in positions]
    ax.set_xticks(x[positions])
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.2)
    _draw_period_overlay(ax, periods, period_context["first_oot"])
    ax.set_ylabel("PSI", fontsize=7.5)
    ax.set_title("分期 PSI 与分箱贡献（相对训练分布；各箱贡献之和=整体 PSI）",
                 fontsize=9, loc="left")
    psi_contrib = matrices["psi_contrib"]
    reliable_contrib = np.where(low_period[:, None], np.nan, psi_contrib)
    source = (reliable_contrib if np.isfinite(reliable_contrib).any()
              else psi_contrib)
    if np.isfinite(source).any():
        flat_index = int(np.nanargmax(source))
        period_index, bin_index = divmod(flat_index, len(ordered))
        prefix = ("可靠期间最大贡献" if np.isfinite(reliable_contrib).any()
                  else "全期间最大贡献")
        ax.text(0.01, 0.98,
                "{}：{} / {}（{:.4f}）".format(
                    prefix, periods.loc[period_index, "period_label"],
                    short_labels[bin_index], source[period_index, bin_index]),
                transform=ax.transAxes, fontsize=6.2, color="#333333",
                va="top", bbox=dict(facecolor="white", alpha=0.75,
                                    edgecolor="none", pad=1.0))
    ax.legend(fontsize=5.8, loc="upper right",
              ncol=min(4, len(ordered) + 2))
    ax.set_xlim(-0.6, n_periods - 0.4)
    ax.set_ylim(0, upper)


def _render_feature_figure_body(fig, analysis, info, params, plt, png_path,
                                pdf_path=None):
    overall = analysis["overall"]
    meta = analysis["overall_meta"]
    matrices = analysis["matrices"]
    period_context = analysis["period_context"]
    diagnostics = analysis["diagnostics"]
    periods = period_context["periods"]
    feature = analysis["feature"]
    export_name = analysis["export_name"]
    ordered = overall
    short_labels = [_short_bin_label(row.lower, row.upper,
                                     bool(row.is_missing))
                    for row in ordered.itertuples()]
    count = len(ordered)
    n_periods = len(periods)

    grid = fig.add_gridspec(
        5, 6, height_ratios=[3.4, 2.8, 2.8, 2.0, 1.15],
        hspace=0.48, wspace=0.34, left=0.055, right=0.97,
        top=0.905, bottom=0.035)
    overall_grid = grid[0, :].subgridspec(1, 3, wspace=0.34)
    left_grid = overall_grid[0].subgridspec(
        2, 1, height_ratios=[3.2, 1.25], hspace=0.08)
    ax_share = fig.add_subplot(left_grid[0])
    ax_table = fig.add_subplot(left_grid[1])
    ax_woe = fig.add_subplot(overall_grid[1])
    ax_iv = fig.add_subplot(overall_grid[2])
    trend_grid = grid[1, :].subgridspec(4, 1, hspace=0.10)
    trend_axes = [fig.add_subplot(trend_grid[index]) for index in range(4)]
    ax_woe_heat = fig.add_subplot(grid[2, 0:3])
    ax_share_heat = fig.add_subplot(grid[2, 3:6])
    ax_psi = fig.add_subplot(grid[3, :])
    ax_diag = fig.add_subplot(grid[4, :])

    title = "特征分析：{}".format(feature)
    if export_name != feature:
        title += "（导出名：{}）".format(export_name)
    title_size = min(15.0, max(7.0, 15.0 * 90.0
                               / max(_text_display_width(title), 1.0)))
    fig.text(0.01, 0.988, title, fontsize=title_size, weight="bold",
             ha="left", va="top")
    train_text = "{} ~ {}".format(
        analysis["train"]["sample_datetime"].min().strftime("%Y-%m-%d"),
        analysis["train"]["sample_datetime"].max().strftime("%Y-%m-%d"))
    if meta["oot_enabled"]:
        oot_text = "{} ~ {}（n={:,}，坏账率 {:.2%}）".format(
            analysis["oot"]["sample_datetime"].min().strftime("%Y-%m-%d"),
            analysis["oot"]["sample_datetime"].max().strftime("%Y-%m-%d"),
            meta["oot_total"], meta["oot_bad_rate"])
    else:
        oot_text = "未启用"
    fig.text(0.01, 0.965,
             "标签：{} | 训练：{}（n={:,}，坏账率 {:.2%}） | OOT：{}".format(
                 params["target"], train_text, meta["train_total"],
                 meta["train_bad_rate"], oot_text),
             fontsize=9, ha="left", va="top")
    fig.text(0.01, 0.947,
             "训练 IV={} | OOT IV={} | iv_min={} | PSI={} | 缺失率={} | "
             "普通箱={} | 训练方向={}".format(
                 _fmt_float(diagnostics["iv_train"]),
                 _fmt_float(diagnostics["iv_oot"]),
                 _fmt_float(diagnostics["iv_min"]),
                 _fmt_float(info.get("psi", np.nan)),
                 _fmt_pct(info.get("missing_rate", np.nan)),
                 diagnostics["n_normal_bins"], diagnostics["train_direction"]),
             fontsize=9, ha="left", va="top")
    fig.text(0.01, 0.929,
             ("注：OOT 参与分箱方案选择，不是独立验证集；OOT 未覆盖样本={}；"
              "诊断为报告层辅助判断，不改变筛选结果。".format(
                  int(round(meta["oot_uncovered"])))
              if meta["oot_enabled"] else
              "注：未启用 OOT，OOT 相关指标为 NA，无法据此判断时间外稳定性；"
              "诊断为报告层辅助判断，不改变筛选结果。"),
             fontsize=7.8, color="#555555", ha="left", va="top")

    _plot_share_panel(ax_share, ax_table, ordered, meta, info,
                      short_labels, params["diag_min_bin_n"])
    _plot_woe_panel(ax_woe, ordered, meta, short_labels,
                    params["diag_min_bin_n"])
    _plot_iv_panel(ax_iv, ordered, meta, info, short_labels,
                   params["diag_min_bin_n"], params["diag_iv_floor"])

    tick_labels = []
    for index in range(n_periods):
        label = str(periods.loc[index, "period_label"])
        coverage = matrices["coverage"][index]
        if np.isfinite(coverage) and coverage < 0.999:
            label += " (cov {:.0%})".format(coverage)
        if periods.loc[index, "split"] == "跨界":
            label += "*"
        tick_labels.append(label)
    trend_info = dict(info)
    trend_info["_train_bad_rate"] = meta["train_bad_rate"]
    _plot_time_trend(trend_axes, periods, matrices, trend_info,
                     period_context, params["max_psi"],
                     params["diag_min_period_n"], tick_labels)
    _plot_woe_heatmap(ax_woe_heat, ordered, matrices, periods,
                      period_context, short_labels,
                      params["diag_min_bin_n"], plt)
    _plot_share_heatmap(ax_share_heat, matrices, periods, period_context,
                        short_labels, plt)
    _plot_psi_panel(ax_psi, ordered, matrices, periods, period_context,
                    short_labels, params["max_psi"])

    conclusion = diagnostics["conclusion"]
    color = {"保留": "#1B7F3A", "观察": "#B7791F",
             "建议复查": "#C0392B"}.get(conclusion, "#222222")
    ax_diag.axis("off")
    ax_diag.text(0.0, 0.98, "结论：{}".format(conclusion),
                 fontsize=12, weight="bold", color=color, va="top")
    ax_diag.text(0.0, 0.73, textwrap.fill(
        "依据：" + diagnostics["evidence"], width=165),
        fontsize=8.0, va="top")
    ax_diag.text(0.0, 0.39, textwrap.fill(
        "限制：" + diagnostics["limitations"], width=165),
        fontsize=7.4, color="#555555", va="top")
    ax_diag.text(0.0, 0.10, textwrap.fill(
        "诊断阈值：" + diagnostics["thresholds"], width=165),
        fontsize=6.8, color="#777777", va="top")

    fig.savefig(str(png_path), dpi=params["plot_dpi"],
                bbox_inches="tight", facecolor="white")
    if pdf_path is not None:
        fig.savefig(str(pdf_path), format="pdf",
                    bbox_inches="tight", facecolor="white")


def _render_feature_figure(analysis, info, params, plt, png_path,
                           pdf_path=None):
    fig = plt.figure(figsize=(17, 22))
    try:
        _render_feature_figure_body(
            fig, analysis, info, params, plt, png_path, pdf_path)
    finally:
        plt.close(fig)


_INDEX_COLUMNS = [
    "original_feature", "export_name", "plot_png", "plot_pdf",
    "plot_status", "plot_error", "conclusion", "evidence",
    "name_collision", "collision_group", "notes",
]
_DIAGNOSTIC_COLUMNS = [
    "feature", "export_name", "plot_status", "plot_error", "plot_png",
    "plot_pdf", "conclusion", "evidence", "severe_issues", "warnings",
    "limitations", "thresholds", "iv_train", "iv_oot", "iv_min",
    "iv_change_pct", "iv_missing", "iv_missing_share", "n_normal_bins",
    "max_iv_bin", "max_iv_bin_n", "max_iv_share", "max_iv_bin_small",
    "small_bin_concentrated", "n_low_bins", "n_zero_good_bins",
    "n_zero_bad_bins", "train_direction", "oot_direction",
    "direction_consistent", "n_periods", "n_cross_periods",
    "n_low_periods", "n_invalid_periods", "n_empty_cells", "n_low_cells",
    "n_zero_good_cells", "n_zero_bad_cells", "n_reversal_periods",
    "n_clear_reversal_periods", "n_jump_steps", "max_period_psi",
    "max_period_psi_label", "max_reliable_period_psi",
    "max_reliable_period_psi_label", "max_psi_contrib_bin",
    "max_psi_contrib_value", "max_reliable_psi_contrib_bin",
    "max_reliable_psi_contrib_value",
    "psi_sum_max_diff", "coverage_all", "coverage_oot",
    "min_period_coverage", "uncovered_total", "uncovered_oot",
    "date_gap_count", "oot_iv_recomputed", "oot_iv_check", "time_freq",
]
_TIME_COLUMNS = [
    "feature", "export_name", "period", "period_start", "period_end",
    "period_split", "period_total_n", "period_mapped_n",
    "period_uncovered_n", "period_coverage", "period_bad", "period_good",
    "period_bad_rate", "period_both_classes", "period_iv", "period_psi",
    "bin_order", "bin", "is_missing", "bin_n", "bin_bad", "bin_good",
    "sample_pct", "bad_rate", "woe", "iv_part", "psi_contrib",
    "reliability_flags",
]


def _setup_matplotlib():
    try:
        import matplotlib
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError(
            "绘图需要 matplotlib，请先安装：pip install matplotlib") from exc
    candidates = [
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei",
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
        "Arial Unicode MS", "DejaVu Sans",
    ]
    available = set(font.name for font in font_manager.fontManager.ttflist)
    chosen = [name for name in candidates if name in available]
    if not chosen:
        chosen = ["DejaVu Sans"]
        print("警告：未找到常用中文字体，图中中文可能显示为方框。")
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    return plt


def generate_feature_analysis_plots(data, train, oot, target, summary,
                                    bin_detail, oot_mask, output_root,
                                    params):
    """对去重后的最终候选特征批量生成分析图、索引和诊断 CSV。"""
    selected = summary.loc[summary["selected"], "feature"].tolist()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(output_root)
               / "analysis_{}_{}".format(_safe_filename(target), timestamp))
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    index_path = run_dir / "image_index.csv"
    diagnostics_path = run_dir / "diagnostics.csv"
    time_path = run_dir / "time_bin_detail.csv"
    if not selected:
        pd.DataFrame(columns=_INDEX_COLUMNS).to_csv(
            index_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=_DIAGNOSTIC_COLUMNS).to_csv(
            diagnostics_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=_TIME_COLUMNS).to_csv(
            time_path, index=False, encoding="utf-8-sig")
        print("最终候选为空，未生成分析图；空结果目录：{}".format(run_dir))
        return {"run_dir": run_dir, "success": 0, "failed": 0,
                "index": pd.DataFrame(columns=_INDEX_COLUMNS),
                "diagnostics": pd.DataFrame(columns=_DIAGNOSTIC_COLUMNS),
                "time_detail": pd.DataFrame(columns=_TIME_COLUMNS)}
    try:
        plt = _setup_matplotlib()
    except RuntimeError as exc:
        print("绘图初始化失败：{}；本次仅输出筛选结果。".format(exc))
        return {"run_dir": run_dir, "success": 0, "failed": len(selected),
                "index": pd.DataFrame(columns=_INDEX_COLUMNS),
                "diagnostics": pd.DataFrame(columns=_DIAGNOSTIC_COLUMNS),
                "time_detail": pd.DataFrame(columns=_TIME_COLUMNS)}

    labels = data[target].to_numpy(dtype=float)
    if oot is not None and oot_mask is not None:
        oot_array = np.asarray(oot_mask, dtype=bool)
    else:
        oot_array = np.zeros(len(data), dtype=bool)
    period_context = _build_period_context(
        data["sample_datetime"].to_numpy(), labels, oot_array,
        params["time_freq"])
    print("分析图：{} 个候选特征；时间粒度 {}；时间段 {} 个；输出 {}"
          .format(len(selected), params["time_freq"],
                  period_context["n_periods"], run_dir))
    if period_context["first_oot"] is not None:
        print("  训练/OOT 切点位于第 {} 个时间段（{}）".format(
            period_context["first_oot"] + 1,
            period_context["periods"].loc[
                period_context["first_oot"], "period_label"]))
    if period_context["gaps"]:
        print("  时间缺口 {} 个，明细见 diagnostics.csv".format(
            len(period_context["gaps"])))

    export_names = {
        feature: str(feature).replace("_woe", "") for feature in selected}
    name_counts = Counter(export_names.values())
    collision_groups = {name for name, count in name_counts.items()
                        if count > 1}
    detail_groups = {feature: group
                     for feature, group in bin_detail.groupby("feature")}
    info_map = summary.set_index("feature").to_dict("index")
    index_rows, diagnostic_rows, time_rows = [], [], []
    failures = []
    for position, feature in enumerate(selected, 1):
        export_name = export_names[feature]
        file_stem = "{:04d}_{}".format(position, _safe_filename(export_name))
        png_path = plots_dir / (file_stem + ".png")
        pdf_path = plots_dir / (file_stem + ".pdf") if params["plot_pdf"] else None
        status, error = "成功", ""
        try:
            detail = detail_groups.get(feature)
            if detail is None:
                raise ValueError("bin_detail 中缺少该特征的分箱结果")
            info = info_map[feature]
            analysis = _prepare_feature_analysis(
                feature, export_name, detail.copy(), info, data, target,
                train, oot, period_context, params)
            _render_feature_figure(
                analysis, info, params, plt, png_path, pdf_path)
            diagnostic = dict(analysis["diagnostics"])
            time_detail = analysis["time_long"]
            time_detail.insert(0, "export_name", export_name)
            time_detail.insert(0, "feature", feature)
            time_rows.append(time_detail)
            index_rows.append({
                "original_feature": feature,
                "export_name": export_name,
                "plot_png": str(png_path),
                "plot_pdf": str(pdf_path) if pdf_path is not None else "",
                "plot_status": status,
                "plot_error": error,
                "conclusion": diagnostic["conclusion"],
                "evidence": diagnostic["evidence"],
                "name_collision": export_name in collision_groups,
                "collision_group": (export_name
                                    if export_name in collision_groups else ""),
                "notes": diagnostic["limitations"],
            })
        except Exception as exc:
            status = "失败"
            error = "{}: {}".format(type(exc).__name__, exc)
            failures.append((feature, error))
            print("警告：特征 {} 绘图失败：{}".format(feature, error))
            diagnostic = {
                "feature": feature, "export_name": export_name,
                "conclusion": "观察",
                "evidence": "绘图失败，未完成诊断：{}".format(error),
                "severe_issues": "", "warnings": "绘图失败",
                "limitations": "绘图失败，结果未评估",
                "thresholds": "",
            }
            index_rows.append({
                "original_feature": feature,
                "export_name": export_name,
                "plot_png": str(png_path),
                "plot_pdf": str(pdf_path) if pdf_path is not None else "",
                "plot_status": status,
                "plot_error": error,
                "conclusion": diagnostic["conclusion"],
                "evidence": diagnostic["evidence"],
                "name_collision": export_name in collision_groups,
                "collision_group": (export_name
                                    if export_name in collision_groups else ""),
                "notes": diagnostic["limitations"],
            })
        diagnostic.update({
            "feature": feature, "export_name": export_name,
            "plot_status": status, "plot_error": error,
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path) if pdf_path is not None else "",
        })
        diagnostic_rows.append(diagnostic)
        if position % 10 == 0 or position == len(selected):
            print("分析绘图：{}/{}".format(position, len(selected)))
    index_frame = pd.DataFrame(index_rows, columns=_INDEX_COLUMNS)
    diagnostic_frame = pd.DataFrame(
        diagnostic_rows, columns=_DIAGNOSTIC_COLUMNS)
    time_frame = (pd.concat(time_rows, ignore_index=True)
                  if time_rows else pd.DataFrame(columns=_TIME_COLUMNS))
    index_frame.to_csv(index_path, index=False, encoding="utf-8-sig")
    diagnostic_frame.to_csv(
        diagnostics_path, index=False, encoding="utf-8-sig")
    time_frame.to_csv(time_path, index=False, encoding="utf-8-sig")
    success = int((index_frame["plot_status"] == "成功").sum())
    failed = len(index_frame) - success
    if failures:
        print("失败清单：")
        for feature, error in failures:
            print("  - {}：{}".format(feature, error))
    print("分析图完成：成功 {}，失败 {}；目录：{}".format(
        success, failed, run_dir))
    result = {
        "run_dir": run_dir, "success": success, "failed": failed,
        "index": index_frame, "diagnostics": diagnostic_frame,
        "time_detail": time_frame,
    }
    return result


def run_feature_screening(root_dir, min_iv=0.02, n_bins=5,
                          n_prebins=20, min_bin_pct=0.05,
                          sample_file="samples.csv",
                          variable_file="variable_bank_statement_surplus.csv",
                          target="y_fpd15_ovd", obs_col=None,
                          feature_prefixes=FEATURE_PREFIXES,
                          oot_start=None, oot_ratio=0.3,
                          max_psi=0.25, max_missing_iv_share=0.5,
                          dedup_corr=0.8,
                          plot_results=True, plot_pdf=False, plot_dpi=200,
                          time_freq="week",
                          diag_min_bin_n=30, diag_min_bin_good=5,
                          diag_min_bin_bad=5, diag_min_period_n=100,
                          diag_min_coverage=0.99, diag_review_coverage=0.90,
                          diag_max_bin_iv_share=0.5,
                          diag_small_bin_iv_share=0.3,
                          diag_direction_tol=0.05,
                          diag_jump_bad_rate=0.10,
                          diag_psi_review=0.5, diag_iv_floor=1e-6):
    """读取指定的样本表/变量表，按指定标签筛选特征。

    返回 data, summary, bin_detail, candidates。
    sample_file / variable_file: 根目录下的样本表、变量表文件名。
    target: 标签列；obs_col 为成熟标记列，None 时按 target 自动匹配。
    feature_prefixes: 参与筛选的列前缀；None 表示变量表除主键和样本表同名列
        外的全部数值列（非数值列自动跳过），显式传前缀时非数值列会报错。
        与样本表同名的列（标签、ID 等）始终排除，避免泄漏。
    oot_start / oot_ratio: 仅按 sample_datetime 时间切分，oot_start 优先；
        oot_ratio 取最近的样本占比，切点落在时间点边界上；
        两者都为 None 时不做 OOT 评估。切点只在训练期拟合。
    max_psi: OOT 与训练集分箱占比的最大允许 PSI。
    max_missing_iv_share: 缺失箱 IV 占整体 IV 的上限；None 不限制。
    dedup_corr: 入选变量间 WOE 相关去重阈值（按 iv_min 降序贪心保留）；
        None 表示不去重。
    min_iv 是 IV 阈值（有 OOT 时取训练与 OOT 的较小值）；
    n_bins/n_prebins/min_bin_pct 控制分箱。

    分析绘图（去重完成后仅对最终候选特征执行，不改变筛选结果）：
    plot_results: 是否生成分析图与诊断；默认开启。
    plot_pdf: 是否同时导出 PDF；默认只导出 PNG。
    plot_dpi: PNG 分辨率，默认 200，支持 200~300。
    time_freq: 分期粒度 day/week/month，默认 week；一次运行内所有特征一致。
    diag_min_bin_n / diag_min_bin_good / diag_min_bin_bad: 分箱可靠性提示
        阈值（样本量、好样本、坏样本），默认 30/5/5。
    diag_min_period_n: 期间样本量提示阈值，默认 100。
    diag_min_coverage: 覆盖率提示阈值，默认 0.99；diag_review_coverage:
        覆盖率复查阈值，默认 0.90。
    diag_max_bin_iv_share: 单箱 IV 贡献集中提示阈值，默认 0.50。
    diag_small_bin_iv_share: 小样本箱贡献集中提示阈值，默认 0.30。
    diag_direction_tol: 期间方向波动容差，默认 0.05。
    diag_jump_bad_rate: 异常跳点的坏账率变化阈值，默认 0.10。
    diag_psi_review: 期间 PSI 复查阈值，默认 0.5。
    diag_iv_floor: 判断 IV≈0 的下限，默认 1e-6。
    以上诊断阈值只用于报告提示，不是筛选标准。
    """
    global LAST_ANALYSIS
    LAST_ANALYSIS = None
    _check_binning_parameters(n_bins, n_prebins, min_bin_pct)
    if not np.isfinite(min_iv) or min_iv < 0:
        raise ValueError("min_iv 必须为非负有限数值。")
    if oot_ratio is not None and not 0 < oot_ratio < 0.5:
        raise ValueError("oot_ratio 必须在 (0, 0.5) 内。")
    if not np.isfinite(max_psi) or max_psi < 0:
        raise ValueError("max_psi 必须为非负有限数值。")
    if max_missing_iv_share is not None and not 0 <= max_missing_iv_share <= 1:
        raise ValueError("max_missing_iv_share 必须在 [0, 1] 内。")
    if dedup_corr is not None and not 0 < dedup_corr <= 1:
        raise ValueError("dedup_corr 必须在 (0, 1] 内。")
    if time_freq not in ("day", "week", "month"):
        raise ValueError("time_freq 只能是 day、week 或 month。")
    if not isinstance(plot_dpi, (int, np.integer)) or not 72 <= plot_dpi <= 600:
        raise ValueError("plot_dpi 必须是 72~600 的整数。")
    for name, value in [
            ("diag_min_bin_n", diag_min_bin_n),
            ("diag_min_bin_good", diag_min_bin_good),
            ("diag_min_bin_bad", diag_min_bin_bad),
            ("diag_min_period_n", diag_min_period_n),
            ("diag_direction_tol", diag_direction_tol),
            ("diag_jump_bad_rate", diag_jump_bad_rate),
            ("diag_psi_review", diag_psi_review),
            ("diag_iv_floor", diag_iv_floor)]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(name + " 必须为非负有限数值。")
    for name, value in [
            ("diag_min_coverage", diag_min_coverage),
            ("diag_review_coverage", diag_review_coverage),
            ("diag_max_bin_iv_share", diag_max_bin_iv_share),
            ("diag_small_bin_iv_share", diag_small_bin_iv_share)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + " 必须在 [0, 1] 内。")
    if diag_review_coverage > diag_min_coverage:
        raise ValueError("diag_review_coverage 不能大于 diag_min_coverage。")
    keys = ["user_id", "sample_datetime"]
    root = Path(root_dir)
    sample_columns = pd.read_csv(root / sample_file, nrows=0).columns
    if target not in sample_columns:
        raise ValueError("样本表缺少标签列：" + target)
    obs = obs_col or _derive_obs_column(target, sample_columns)
    samples = pd.read_csv(
        root / sample_file, dtype={"user_id": str},
        usecols=lambda c: c in keys + [target, obs], low_memory=False,
    )
    variables = pd.read_csv(
        root / variable_file, dtype={"user_id": str}, low_memory=False,
    )
    for name, frame in [("样本表", samples), ("变量表", variables)]:
        if not set(keys).issubset(frame.columns):
            raise ValueError(name + " 缺少主键列。")
        frame["user_id"] = frame["user_id"].str.strip().replace("", np.nan)
        frame["sample_datetime"] = pd.to_datetime(frame["sample_datetime"], errors="raise")
        if frame[keys].isna().any().any():
            raise ValueError(name + " 存在空主键，请先处理。")
        if frame.duplicated(keys).any():
            raise ValueError(name + " 主键重复，请先确认原因，不自动去重。")

    # 变量表可能混入样本表的列（标签、ID 等），一律不作为特征。
    excluded = [c for c in variables.columns
                if c in sample_columns and c not in keys]
    if excluded:
        print("已排除变量表中与样本表同名的列：{}".format("、".join(excluded)))
    if feature_prefixes is None:
        candidates = [c for c in variables.columns
                      if c not in keys and c not in sample_columns]
        strict = False
    else:
        candidates = [c for c in variables.columns
                      if c.startswith(tuple(feature_prefixes))
                      and c not in sample_columns]
        strict = True
    features, skipped = [], []
    for col in candidates:
        try:
            numeric = pd.to_numeric(variables[col], errors="raise")
        except (ValueError, TypeError) as exc:
            if strict:
                raise ValueError("特征 {} 含非数值内容，请检查。".format(col)) from exc
            skipped.append(col)
            continue
        default = -1 if col.startswith("bank_txn_lender") else -999999.99999
        variables[col] = numeric.replace([default, np.inf, -np.inf], np.nan)
        features.append(col)
    if skipped:
        print("已跳过非数值列：{}".format("、".join(skipped)))
    if not features:
        raise ValueError("变量表没有可用的数值特征列。")

    joined = samples.merge(
        variables[keys + features], on=keys, how="left",
        validate="one_to_one", indicator=True,
    )
    matched = joined["_merge"].eq("both")
    joined[target] = pd.to_numeric(joined[target], errors="coerce")
    eligible = matched & joined[target].isin([0, 1])
    if obs is not None and obs in joined:
        eligible &= pd.to_numeric(joined[obs], errors="coerce").eq(1)
    else:
        print("未发现成熟标记列，未按成熟度过滤；请确认未成熟标签没有被填成 0。")
    data = joined.loc[eligible, keys + [target] + features].copy()
    print("样本数：{:,}；匹配数：{:,}；匹配率：{:.2%}".format(
        len(samples), int(matched.sum()), matched.mean()))
    print("分析样本：{:,}；逾期率：{:.2%}；数值特征：{:,}".format(
        len(data), data[target].mean(), len(features)))
    if data[target].nunique() != 2:
        raise ValueError("分析样本必须同时包含标签 0 和 1，请检查匹配及成熟度。")

    train, oot, split_note = data, None, None
    oot_mask = None
    if oot_start is not None:
        cutoff = pd.Timestamp(oot_start)
        oot_mask = (data["sample_datetime"] >= cutoff).to_numpy()
        if oot_mask.all() or not oot_mask.any():
            raise ValueError("oot_start 切分后训练集或 OOT 为空，请调整 oot_start。")
        split_note = "时间切点 {}".format(cutoff)
    elif oot_ratio is not None:
        if data["sample_datetime"].nunique() < 2:
            raise ValueError(
                "sample_datetime 只有一个取值，无法按时间切分；"
                "请提供跨多个时间点的样本，或设置 oot_ratio=None。")
        oot_mask = _time_oot_mask(data["sample_datetime"], oot_ratio)
        split_note = "时间切点 {}".format(data.loc[oot_mask, "sample_datetime"].min())
    if oot_mask is not None:
        train, oot = data.loc[~oot_mask], data.loc[oot_mask]
        for name, part in [("训练集", train), ("OOT", oot)]:
            bad_n = int(part[target].sum())
            if part[target].nunique() != 2:
                raise ValueError("{} 必须同时包含标签 0 和 1（0: {:,}，1: {:,}），"
                                 "请调整 oot_start/oot_ratio。".format(
                                     name, int(len(part) - bad_n), bad_n))
            if bad_n < 50:
                print("警告：{} 仅 {:,} 个坏样本，IV/PSI 估计会不稳定。".format(name, bad_n))
        print("OOT 切分：{}；训练集：{:,}（逾期率 {:.2%}）；OOT：{:,}（逾期率 {:.2%}）".format(
            split_note, len(train), train[target].mean(),
            len(oot), oot[target].mean()))

    rows, details = [], []
    for index, col in enumerate(features, 1):
        grouped, info = auto_monotonic_binning(
            train[col], train[target], n_bins=n_bins,
            n_prebins=n_prebins, min_bin_pct=min_bin_pct,
            x_oot=None if oot is None else oot[col],
            y_oot=None if oot is None else oot[target],
            max_psi=max_psi,
        )
        missing_share_ok = (
            max_missing_iv_share is None
            or info["iv_missing_share"] <= max_missing_iv_share + TOL
        )
        stable_ok = info["is_stable"] is not False
        selected = bool(info["iv_min"] >= min_iv and info["is_monotonic"]
                        and missing_share_ok and stable_ok)
        if not info["is_monotonic"]:
            reason = "无法判断单调性（全缺失）"
        elif not missing_share_ok:
            reason = "缺失箱 IV 占比过高"
        elif info["iv_min"] < min_iv:
            reason = "IV 低于阈值"
        elif not stable_ok:
            reason = "OOT 不稳定"
        else:
            reason = "入选"
        rows.append(dict(feature=col, selected=selected, reason=reason, **info))
        grouped.insert(0, "feature", col)
        details.append(grouped)
        if index % 100 == 0:
            print("已处理：{}/{}".format(index, len(features)))

    summary = pd.DataFrame(rows).sort_values("iv_min", ascending=False).reset_index(drop=True)
    summary["dedup_with"] = np.nan
    bin_detail = pd.concat(details, ignore_index=True)
    if dedup_corr is not None:
        selected_features = summary.loc[summary["selected"], "feature"].tolist()
        if selected_features:
            woe_matrix = pd.DataFrame({
                feature: _assign_woe(
                    train[feature],
                    bin_detail.loc[bin_detail["feature"] == feature])
                for feature in selected_features
            })
            corr = woe_matrix.corr(method="spearman").abs()
            kept, correlated_with = [], {}
            for feature in selected_features:
                hits = [k for k in kept if corr.loc[feature, k] >= dedup_corr - TOL]
                if hits:
                    correlated_with[feature] = hits[0]
                else:
                    kept.append(feature)
            if correlated_with:
                dropped = summary["feature"].isin(correlated_with)
                summary.loc[dropped, "selected"] = False
                summary.loc[dropped, "reason"] = "与已选变量高相关"
                summary["dedup_with"] = summary["feature"].map(correlated_with)
                print("相关性去重：{} -> {}（阈值 {:g}）".format(
                    len(selected_features), len(kept), dedup_corr))
    candidates = summary.loc[summary["selected"], ["feature"]].reset_index(drop=True)
    candidates["feature"] = candidates["feature"].str.replace("_woe", "", regex=False)
    candidates = candidates.drop_duplicates().reset_index(drop=True)
    output = root / "feature_screening_reports"
    output.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    for name, frame in [("特征筛选结果", summary), ("特征分箱结果", bin_detail),
                        ("候选变量清单", candidates)]:
        frame.to_csv(output / "{}_{}_{}.csv".format(name, target, date),
                     index=False, encoding="utf-8-sig")
    print("入选特征：{}；结果目录：{}".format(len(candidates), output))
    if plot_results:
        analysis_params = {
            "target": target,
            "min_iv": float(min_iv),
            "max_psi": float(max_psi),
            "max_missing_iv_share": max_missing_iv_share,
            "plot_pdf": bool(plot_pdf),
            "plot_dpi": int(plot_dpi),
            "time_freq": time_freq,
            "diag_min_bin_n": int(diag_min_bin_n),
            "diag_min_bin_good": int(diag_min_bin_good),
            "diag_min_bin_bad": int(diag_min_bin_bad),
            "diag_min_period_n": int(diag_min_period_n),
            "diag_min_coverage": float(diag_min_coverage),
            "diag_review_coverage": float(diag_review_coverage),
            "diag_max_bin_iv_share": float(diag_max_bin_iv_share),
            "diag_small_bin_iv_share": float(diag_small_bin_iv_share),
            "diag_direction_tol": float(diag_direction_tol),
            "diag_jump_bad_rate": float(diag_jump_bad_rate),
            "diag_psi_review": float(diag_psi_review),
            "diag_iv_floor": float(diag_iv_floor),
        }
        LAST_ANALYSIS = generate_feature_analysis_plots(
            data, train, oot, target, summary, bin_detail, oot_mask,
            output, analysis_params)
    else:
        print("分析绘图已关闭（plot_results=False）。")
    return data, summary, bin_detail, candidates


if __name__ == "__main__":
    # 固定数据根目录，与脚本所在位置无关。
    data, summary, bin_detail, candidates = run_feature_screening(DEFAULT_ROOT_DIR)
    try:
        from IPython.display import display
        display(summary.head(30))
    except ImportError:
        print(summary.head(30).to_string(index=False))
