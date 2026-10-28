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
来源，并在图底部标注纳入候选的原因。也可通过 plot_features 指定变量
清单，只绘制清单中的变量（不论是否入选）。绘图只读取现有结果，不修改
selected、reason、候选清单或分箱边界。绘图依赖 matplotlib。

结果按样本表文件名、变量表文件名和标签归档到
{根目录}/feature_screening_reports/{样本文件名}_{变量表文件名}_{标签名}/，
同一组合重复运行会覆盖同名文件；分析图按特征名命名，绘图完成后自动打包为
同目录下的 {样本文件名}_{变量表文件名}_{标签名}_plots.zip，便于下载。

大数据/线上运行：
- feature_chunk_size: 按变量分块读取和筛选，显著降低峰值内存；None 表示
  一次性载入全部变量。分块模式返回的 data 只含主键、标签和时间列（绘图时
  再加上本次绘制的变量列），不包含全部特征。
- n_jobs: 分箱并行进程数，-1 表示全部 CPU；无 fork 的平台自动退回单进程。
- dedup_max_features / dedup_sample_rows: WOE 相关性去重的变量数上限和
  训练样本行数上限；分块模式未指定行数时默认最多 20 万行。
- plot_max_features: 只绘制 IV 最高的前 N 个候选，None 表示全部。
- plot_n_jobs: 并行绘图进程数，-1 表示全部 CPU。
"""
from pathlib import Path
from collections import Counter
import textwrap
import re
import os
import time
import zipfile
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
# 结构：{"run_dir": Path, "index": DataFrame, "time_detail": DataFrame}
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

# 绘图统一字号（只影响版式，不影响筛选结果）。
_FS_BASE = 9.0
_FS_TITLE = 14.0
_FS_HEADER = 9.5
_FS_NOTE = 8.5
_FS_PANEL_TITLE = 10.0
_FS_AXIS_LABEL = 9.0
_FS_TICK = 8.5
_FS_LEGEND = 8.0
_FS_CELL_MIN = 7.0
_FS_CELL_MAX = 8.5
_FS_FOOTER = 10.0
_FS_TABLE = 7.8
_STACK_HEATMAPS_AFTER = 16


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


def _screen_feature_record(feature, info, min_iv, max_missing_iv_share):
    """按筛选门槛生成单特征结果行，口径与逐特征主流程一致。"""
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
    return dict(feature=feature, selected=selected, reason=reason, **info)


def _screen_one_feature(feature, train_x, train_y, oot_x, oot_y, params):
    grouped, info = auto_monotonic_binning(
        train_x, train_y, n_bins=params["n_bins"],
        n_prebins=params["n_prebins"], min_bin_pct=params["min_bin_pct"],
        x_oot=oot_x, y_oot=oot_y, max_psi=params["max_psi"],
    )
    record = _screen_feature_record(
        feature, info, params["min_iv"], params["max_missing_iv_share"])
    grouped.insert(0, "feature", feature)
    return record, grouped


_SCREEN_WORKER_CONTEXT = None


def _screen_feature_worker(feature):
    ctx = _SCREEN_WORKER_CONTEXT
    oot = ctx["oot"]
    return _screen_one_feature(
        feature, ctx["train"][feature], ctx["train_y"],
        None if oot is None else oot[feature], ctx["oot_y"], ctx["params"])


def _screen_feature_set(features, train, oot, target, params, n_jobs=1,
                        progress=True):
    """逐特征分箱筛选；n_jobs>1 时在 POSIX 上用 fork 并行，结果顺序不变。"""
    if not features:
        return [], []
    train_y = train[target]
    oot_y = None if oot is None else oot[target]
    workers = int(n_jobs)
    if workers == -1:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, len(features)))
    use_pool = workers > 1 and hasattr(os, "fork")
    rows, details = [], []
    if not use_pool:
        for index, feature in enumerate(features, 1):
            record, grouped = _screen_one_feature(
                feature, train[feature], train_y,
                None if oot is None else oot[feature], oot_y, params)
            rows.append(record)
            details.append(grouped)
            if progress and index % 100 == 0:
                print("已处理：{}/{}".format(index, len(features)))
        return rows, details

    global _SCREEN_WORKER_CONTEXT
    import multiprocessing
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        for index, feature in enumerate(features, 1):
            record, grouped = _screen_one_feature(
                feature, train[feature], train_y,
                None if oot is None else oot[feature], oot_y, params)
            rows.append(record)
            details.append(grouped)
        return rows, details
    _SCREEN_WORKER_CONTEXT = {
        "train": train, "train_y": train_y, "oot": oot, "oot_y": oot_y,
        "params": params,
    }
    if progress:
        print("并行分箱：{} 个进程".format(workers))
    chunksize = max(1, len(features) // (workers * 4))
    try:
        with context.Pool(processes=workers) as pool:
            iterator = pool.imap(
                _screen_feature_worker, features, chunksize=chunksize)
            for index, (record, grouped) in enumerate(iterator, 1):
                rows.append(record)
                details.append(grouped)
                if progress and index % 100 == 0:
                    print("已处理：{}/{}".format(index, len(features)))
    finally:
        _SCREEN_WORKER_CONTEXT = None
    return rows, details


def _build_summary_frames(rows, details):
    summary = pd.DataFrame(rows).sort_values(
        "iv_min", ascending=False).reset_index(drop=True)
    summary["dedup_with"] = np.nan
    bin_detail = pd.concat(details, ignore_index=True)
    return summary, bin_detail


def _dedup_feature_list(selected_features, max_features):
    features = list(selected_features)
    if max_features is not None and len(features) > max_features:
        print("相关性去重只比较 IV 最高的 {} 个变量（共 {} 个），"
              "其余变量保持入选".format(int(max_features), len(features)))
        features = features[:int(max_features)]
    return features


def _dedup_train_frame(train, sample_rows):
    if sample_rows is None or int(sample_rows) >= len(train):
        return train
    rng = np.random.RandomState(20240101)
    positions = np.sort(
        rng.choice(len(train), size=int(sample_rows), replace=False))
    print("相关性去重使用 {} / {} 行训练样本".format(len(positions), len(train)))
    return train.iloc[positions]


def _dedup_from_woe(summary, woe_matrix, threshold):
    """在 WOE 矩阵上做 Spearman 贪心去重，行为与原全量矩阵一致。"""
    features = list(woe_matrix.columns)
    corr = woe_matrix.corr(method="spearman").abs()
    kept, correlated_with = [], {}
    for feature in features:
        hits = [k for k in kept if corr.loc[feature, k] >= threshold - TOL]
        if hits:
            correlated_with[feature] = hits[0]
        else:
            kept.append(feature)
    if correlated_with:
        dropped = summary["feature"].isin(correlated_with)
        summary.loc[dropped, "selected"] = False
        summary.loc[dropped, "reason"] = "与已选变量高相关"
        summary["dedup_with"] = summary["feature"].map(correlated_with)
    return len(features), len(kept)


def _apply_dedup(summary, bin_detail, train, threshold, max_features=None,
                 sample_rows=None):
    if threshold is None:
        return
    selected_features = summary.loc[summary["selected"], "feature"].tolist()
    if not selected_features:
        return
    dedup_features = _dedup_feature_list(selected_features, max_features)
    dedup_train = _dedup_train_frame(train, sample_rows)
    detail_groups = {feature: group
                     for feature, group in bin_detail.groupby("feature")}
    woe_matrix = pd.DataFrame({
        feature: _assign_woe(dedup_train[feature], detail_groups[feature])
        for feature in dedup_features
    })
    used, kept = _dedup_from_woe(summary, woe_matrix, threshold)
    print("相关性去重：参与 {} 个 -> 保留 {} 个（阈值 {:g}）".format(
        used, kept, threshold))


def _resolve_plot_targets(summary, manual_features, plot_max_features,
                          verbose=True):
    if manual_features:
        return list(manual_features)
    targets = summary.loc[summary["selected"], "feature"].tolist()
    if plot_max_features is not None and len(targets) > plot_max_features:
        if verbose:
            print("候选特征 {} 个，按 IV 只绘制前 {} 个（plot_max_features）"
                  .format(len(targets), int(plot_max_features)))
        targets = targets[:int(plot_max_features)]
    return targets


def _read_variable_columns(path, keys, columns):
    """只读取主键和指定特征列，避免把整张变量表载入内存。"""
    usecols = list(keys) + [c for c in columns if c not in keys]
    return pd.read_csv(
        path, usecols=usecols, dtype={"user_id": str}, low_memory=False)


def _clean_variable_keys(frame, keys, name="变量表"):
    frame["user_id"] = frame["user_id"].str.strip().replace("", np.nan)
    frame["sample_datetime"] = pd.to_datetime(
        frame["sample_datetime"], errors="raise")
    if frame[keys].isna().any().any():
        raise ValueError(name + " 存在空主键，请先处理。")
    if frame.duplicated(keys).any():
        raise ValueError(name + " 主键重复，请先确认原因，不自动去重。")
    return frame


def _numeric_feature_block(frame, columns, strict):
    """把一块特征转为数值并清理业务默认值；返回可用列和被跳过的列。"""
    features, skipped = [], []
    for col in columns:
        try:
            numeric = pd.to_numeric(frame[col], errors="raise")
        except (ValueError, TypeError) as exc:
            if strict:
                raise ValueError("特征 {} 含非数值内容，请检查。".format(col)) from exc
            skipped.append(col)
            continue
        default = -1 if col.startswith("bank_txn_lender") else -999999.99999
        frame[col] = numeric.replace([default, np.inf, -np.inf], np.nan)
        features.append(col)
    return features, skipped


def _column_blocks(columns, size):
    size = max(1, int(size))
    return [columns[i:i + size] for i in range(0, len(columns), size)]


def _load_feature_blocks(path, keys, features, base, chunk_size, strict=False):
    """按列块读取特征并对齐到 base 的行顺序，避免整表常驻内存。"""
    if not features:
        return base.copy()
    parts = [base.reset_index(drop=True)]
    for block in _column_blocks(features, chunk_size):
        frame = _read_variable_columns(path, keys, block)
        _clean_variable_keys(frame, keys)
        block_features, _ = _numeric_feature_block(frame, block, strict)
        if not block_features:
            continue
        merged = base[keys].merge(
            frame[keys + block_features], on=keys, how="left",
            validate="one_to_one")
        parts.append(merged[block_features].reset_index(drop=True))
    return pd.concat(parts, axis=1)


def _load_woe_blocks(path, keys, features, detail_groups, base, chunk_size):
    """按列块读取特征并映射为 WOE，返回与 features 同序的 WOE 矩阵。"""
    parts = []
    for block in _column_blocks(features, chunk_size):
        frame = _read_variable_columns(path, keys, block)
        _clean_variable_keys(frame, keys)
        _numeric_feature_block(frame, block, strict=True)
        merged = base[keys].merge(
            frame[keys + block], on=keys, how="left", validate="one_to_one")
        parts.append(pd.DataFrame({
            feature: _assign_woe(merged[feature], detail_groups[feature])
            for feature in block
        }))
    if not parts:
        return pd.DataFrame(index=range(len(base)))
    return pd.concat(parts, axis=1)[list(features)]


def _split_train_oot(data, target, oot_start, oot_ratio):
    """按 sample_datetime 切分训练/OOT，返回 train、oot、掩码和说明。"""
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
    return train, oot, oot_mask, split_note


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


_DIRECTION_EN = {
    "递增": "increasing",
    "递减": "decreasing",
    "持平": "flat",
    "单普通箱，免检": "single bin",
    "全缺失": "all missing",
}


def _direction_en(value):
    return _DIRECTION_EN.get(str(value), str(value))


_SCREEN_REASON_EN = {
    "入选": "selected",
    "IV 低于阈值": "IV below threshold",
    "OOT 不稳定": "OOT unstable",
    "缺失箱 IV 占比过高": "missing-bin IV share too high",
    "无法判断单调性（全缺失）": "monotonicity unavailable (all missing)",
    "与已选变量高相关": "highly correlated with a selected feature",
}


def _screen_reason_en(value):
    return _SCREEN_REASON_EN.get(str(value), "not selected")


def _short_bin_label(lower, upper, is_missing, digits=4):
    """图内使用的短区间标签；完整切点保留在明细 CSV。"""
    if is_missing:
        return "Missing"
    fmt = "{:." + str(digits) + "g}"
    if lower == -np.inf:
        return "<=" + fmt.format(upper)
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


def _prepare_feature_analysis(feature, export_name, detail, info, data,
                              target, train, oot, period_context, params):
    """计算单个特征的整体/分期统计和候选原因，不修改任何筛选字段。"""
    overall, overall_meta = _overall_table(detail, info, train, oot, target)
    matrices, time_long = _time_bin_matrices(
        data[feature].to_numpy(dtype=float),
        data[target].to_numpy(dtype=float),
        detail, period_context,
        params["plot_min_bin_n"], params["plot_min_bin_good"],
        params["plot_min_bin_bad"], params["plot_min_period_n"])
    reason = _build_candidate_reason(
        overall, overall_meta, info, period_context, params)
    return {
        "feature": feature, "export_name": export_name,
        "overall": overall, "overall_meta": overall_meta,
        "matrices": matrices, "time_long": time_long,
        "reason": reason, "period_context": period_context,
    }


def _build_candidate_reason(overall, overall_meta, info, period_context,
                            params):
    """汇总该特征通过筛选的依据，不改动任何筛选结果。"""
    oot_enabled = overall_meta["oot_enabled"]
    iv_train = float(info.get("iv", np.nan))
    iv_oot = float(info.get("oot_iv", np.nan)) if oot_enabled else np.nan
    iv_min = float(info.get("iv_min", np.nan))
    iv_floor = params["plot_iv_floor"]
    train_dir = str(info.get("monotonic_direction", "NA"))
    n_normal_bins = int(info.get("n_normal_bins", 0))
    missing_share = float(info.get("iv_missing_share", np.nan))
    train_n = int(overall_meta["train_total"])
    oot_total = int(overall_meta["oot_total"])
    train_mapped = float(overall["n"].sum())
    oot_mapped = float(overall_meta["oot_mapped"])
    coverage_all = ((train_mapped + oot_mapped)
                    / float(train_n + oot_total)) if (train_n + oot_total) else np.nan
    coverage_oot = (oot_mapped / oot_total) if oot_total else np.nan
    uncovered_oot = float(overall_meta["oot_uncovered"])

    oot_dir = "NA"
    direction_consistent = None
    if oot_enabled:
        ordinary = overall.loc[~overall["is_missing"]]
        oot_rates = ordinary["oot_bad_rate"].to_numpy(dtype=float)
        oot_ns = ordinary["oot_n"].to_numpy(dtype=float)
        oot_dir = _direction_from_rates(oot_rates[oot_ns > 0])
        if train_dir in ("递增", "递减") and oot_dir in ("递增", "递减"):
            direction_consistent = (train_dir == oot_dir)

    parts = []
    if train_dir in ("递增", "递减"):
        if oot_enabled and info.get("oot_is_monotonic"):
            parts.append("Train/OOT both monotonic {}".format(
                _direction_en(train_dir)))
        else:
            parts.append("Train monotonic {}".format(_direction_en(train_dir)))
    elif train_dir == "持平":
        parts.append("Ordinary-bin bad rate flat")
    elif train_dir == "单普通箱，免检":
        parts.append("Only one ordinary bin; no monotonic direction")
    else:
        parts.append("Train direction={}".format(_direction_en(train_dir)))
    if oot_enabled:
        parts.append("Train IV={}, OOT IV={}, iv_min={} >= {}".format(
            _fmt_float(iv_train), _fmt_float(iv_oot), _fmt_float(iv_min),
            _fmt_float(params["min_iv"])))
        parts.append("OOT PSI={} <= {}".format(
            _fmt_float(info.get("psi", np.nan)),
            _fmt_float(params["max_psi"])))
    else:
        parts.append("Train IV={} >= {}".format(
            _fmt_float(iv_train), _fmt_float(params["min_iv"])))
        parts.append("OOT disabled")
    if params["max_missing_iv_share"] is None:
        parts.append("Missing-bin IV share={}".format(
            _fmt_pct(missing_share)))
    else:
        parts.append("Missing-bin IV share={} <= {:.0%}".format(
            _fmt_pct(missing_share), params["max_missing_iv_share"]))
    if params.get("dedup_corr") is not None:
        parts.append("Passed WOE correlation dedup (threshold {:g})".format(
            params["dedup_corr"]))
    else:
        parts.append("Correlation dedup disabled")
    candidate_reason = "Candidate reason: " + "; ".join(parts) + "."

    iv_change_pct = (
        (iv_oot / iv_train - 1.0) * 100.0
        if (np.isfinite(iv_oot) and np.isfinite(iv_train)
            and iv_train > iv_floor) else np.nan)
    return {
        "candidate_reason": candidate_reason,
        "iv_train": iv_train, "iv_oot": iv_oot, "iv_min": iv_min,
        "iv_change_pct": iv_change_pct,
        "iv_missing": info.get("iv_missing", np.nan),
        "iv_missing_share": missing_share,
        "n_normal_bins": n_normal_bins,
        "train_direction": train_dir, "oot_direction": oot_dir,
        "direction_consistent": direction_consistent,
        "psi": float(info.get("psi", np.nan)),
        "missing_rate": float(info.get("missing_rate", np.nan)),
        "n_periods": period_context["n_periods"],
        "coverage_all": coverage_all, "coverage_oot": coverage_oot,
        "uncovered_oot": uncovered_oot,
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
    value = float(value)
    return "{:,.0f}".format(value) if np.isfinite(value) else "NA"


def _text_char_width(fontsize):
    # DejaVu Sans 平均字符宽度的保守估计，用于按图宽自动换行。
    return max(fontsize, 1.0) * 0.60 / 72.0


def _wrap_text_for_width(text, width_in, fontsize):
    width_in = max(float(width_in), 1.0)
    max_chars = max(24, int(width_in / _text_char_width(fontsize)))
    return textwrap.fill(str(text), width=max_chars,
                         break_long_words=True, break_on_hyphens=False)


def _wrap_text_for_figure(text, fig, fontsize, width_fraction=0.97):
    width_in = fig.get_size_inches()[0] * width_fraction
    return _wrap_text_for_width(text, width_in, fontsize)


def _wrap_text_for_axes(ax, text, fontsize, width_fraction=0.98):
    fig = ax.figure
    position = ax.get_position()
    width_in = position.width * fig.get_size_inches()[0] * width_fraction
    return _wrap_text_for_width(text, width_in, fontsize)


def _add_wrapped_fig_text(fig, x, y, text, fontsize, color="#111111",
                          weight="normal", ha="left"):
    wrapped = _wrap_text_for_figure(text, fig, fontsize)
    height_in = max(fig.get_size_inches()[1], 1.0)
    line_step = fontsize * 1.42 / (height_in * 72.0)
    for line in wrapped.split("\n"):
        fig.text(x, y, line, fontsize=fontsize, color=color,
                 weight=weight, ha=ha, va="top")
        y -= line_step
    return y


def _heatmap_cell_inches(ax, rows, columns):
    position = ax.get_position()
    fig_width, fig_height = ax.figure.get_size_inches()
    # 预留色标、刻度和说明文字占用的空间。
    width_in = position.width * fig_width * 0.86
    height_in = position.height * fig_height * 0.66
    return width_in / max(columns, 1), height_in / max(rows, 1)


def _heatmap_value_fontsize(cell_height_in):
    return max(_FS_CELL_MIN,
               min(_FS_CELL_MAX, cell_height_in * 72.0 * 0.40))


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
        linewidth=0.6, label="Train share", zorder=2)
    bars_oot = None
    if oot_enabled:
        oot_total = max(meta["oot_total"], 1)
        share_oot = ordered["oot_n"].to_numpy(dtype=float) / oot_total * 100.0
        bars_oot = ax.bar(
            x + width / 2.0, share_oot, width, color=_OOT_COLOR,
            alpha=0.72, edgecolor="white", linewidth=0.6,
            label="OOT share", zorder=2)
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
    ax.set_ylabel("Sample share (%)", fontsize=_FS_AXIS_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha="right",
                       fontsize=_FS_TICK)
    ax.tick_params(axis="x", pad=4)
    upper = np.nanmax(share_train) if np.isfinite(share_train).any() else 10.0
    if oot_enabled:
        oot_share = (ordered["oot_n"].to_numpy(dtype=float)
                     / max(meta["oot_total"], 1) * 100.0)
        if np.isfinite(oot_share).any():
            upper = max(upper, float(np.nanmax(oot_share)))
    ax.set_ylim(0, max(upper * 1.30, 5.0))
    ax.set_title("Bin share and bad rate (train vs OOT)"
                 if oot_enabled else "Bin share and bad rate (train only)",
                 fontsize=_FS_PANEL_TITLE, loc="left", pad=8)
    ax.text(0.01, 0.98,
            "Bars = sample share; line = bad rate (95% Wilson CI); "
            "grey hatch = missing bin; red edge = low sample.",
            transform=ax.transAxes, fontsize=_FS_NOTE, color="#555555",
            va="top", bbox=dict(facecolor="white", alpha=0.86,
                                edgecolor="none", pad=1.2), zorder=6)
    ax2 = ax.twinx()
    ordinary = ~is_missing
    train_rate = ordered["bad_rate"].to_numpy(dtype=float) * 100.0
    valid = ordinary & np.isfinite(train_rate)
    ax2.plot(x[valid], train_rate[valid], color=_TRAIN_COLOR,
             marker="o", lw=1.7, label="Train bad rate", zorder=3)
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
                    label="Train missing bin")
    if oot_enabled:
        oot_rate = ordered["oot_bad_rate"].to_numpy(dtype=float) * 100.0
        oot_valid = ordinary & np.isfinite(oot_rate)
        ax2.plot(x[oot_valid], oot_rate[oot_valid], color=_OOT_COLOR,
                 marker="s", lw=1.4, ls="--", label="OOT bad rate", zorder=3)
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
                        label="OOT missing bin")
    ax2.axhline(meta["train_bad_rate"] * 100.0, color=_TRAIN_COLOR,
                ls=":", lw=1.0, alpha=0.7, label="Train overall bad rate")
    if oot_enabled:
        ax2.axhline(meta["oot_bad_rate"] * 100.0, color=_OOT_COLOR,
                    ls=":", lw=1.0, alpha=0.7, label="OOT overall bad rate")
    ax2.set_ylabel("Bad rate (%)", fontsize=_FS_AXIS_LABEL)
    ax2.tick_params(axis="y", labelsize=_FS_TICK)
    max_rate = np.nanmax(train_rate) if np.isfinite(train_rate).any() else 10.0
    if oot_enabled:
        oot_rate_all = ordered["oot_bad_rate"].to_numpy(dtype=float) * 100.0
        if np.isfinite(oot_rate_all).any():
            max_rate = max(max_rate, float(np.nanmax(oot_rate_all)))
    ax2.set_ylim(0, max(max_rate * 1.35, 5.0))

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend_handles = handles1 + [
        handle for handle, label in zip(handles2, labels2)
        if "missing bin" not in label]
    legend_labels = labels1 + [
        label for label in labels2 if "missing bin" not in label]

    ax_table.axis("off")
    row_labels, cell_text = [], []
    if oot_enabled:
        row_labels = ["Train n", "Train bad", "OOT n", "OOT bad"]
        cell_text = [
            [_fmt_n(value) for value in ordered["n"]],
            [_fmt_n(value) for value in ordered["bad"]],
            [_fmt_n(value) for value in ordered["oot_n"]],
            [_fmt_n(value) for value in ordered["oot_bad"]],
        ]
    else:
        row_labels = ["Train n", "Train bad"]
        cell_text = [
            [_fmt_n(value) for value in ordered["n"]],
            [_fmt_n(value) for value in ordered["bad"]],
        ]
    table = ax_table.table(cellText=cell_text, rowLabels=row_labels,
                           colLabels=short_labels, loc="center",
                           cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(_FS_TABLE if count <= 6 else _FS_TABLE - 0.5)
    table.scale(1.0, 1.22)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#EDEDED")
            cell.set_text_props(weight="bold")
        if col >= 0 and is_missing[col]:
            cell.set_facecolor("#F2F2F2")
        cell.set_edgecolor("#D9D9D9")
    ax_table.set_title("Bin detail (n = samples, bad = bad samples)",
                       fontsize=_FS_AXIS_LABEL, pad=3)
    return legend_handles, legend_labels


def _plot_woe_panel(ax, ordered, meta, short_labels, min_bin_n):
    count = len(ordered)
    x = np.arange(count)
    width = 0.38 if meta["oot_enabled"] else 0.68
    train_woe = ordered["woe"].to_numpy(dtype=float)
    bars_train = ax.bar(
        x - (width / 2.0 if meta["oot_enabled"] else 0.0), train_woe,
        width, color=_TRAIN_COLOR, alpha=0.82, label="Train WOE", zorder=2)
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
    ax.set_xticklabels(short_labels, rotation=20, ha="right",
                       fontsize=_FS_TICK)
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel("WOE", fontsize=_FS_AXIS_LABEL)
    ax.set_title("Bin WOE (positive = lower relative risk, negative = higher)",
                 fontsize=_FS_PANEL_TITLE, loc="left", pad=8)
    ax.text(0.01, 0.98, "WOE = ln(good share / bad share)",
            transform=ax.transAxes, fontsize=_FS_NOTE, color="#555555",
            va="top", bbox=dict(facecolor="white", alpha=0.86,
                                edgecolor="none", pad=1.2), zorder=6)
    if count <= 8:
        for index in range(count):
            value = train_woe[index]
            if np.isfinite(value):
                ax.text(index - (width / 2.0 if meta["oot_enabled"] else 0.0),
                        value, "{:.2f}".format(value), ha="center",
                        va="bottom" if value >= 0 else "top",
                        fontsize=_FS_TICK)
            if meta["oot_enabled"]:
                value = ordered["oot_woe"].to_numpy(dtype=float)[index]
                if np.isfinite(value):
                    ax.text(index + width / 2.0, value,
                            "{:.2f}".format(value), ha="center",
                            va="bottom" if value >= 0 else "top",
                            fontsize=_FS_TICK)
    values = np.r_[train_woe, ordered["oot_woe"].to_numpy(dtype=float)] \
        if meta["oot_enabled"] else train_woe
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(finite.max()) * 1.35, 0.5) if len(finite) else 1.0
    ax.set_ylim(-limit, limit)
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def _plot_iv_panel(ax, ordered, meta, info, short_labels, min_bin_n,
                   iv_floor):
    count = len(ordered)
    x = np.arange(count)
    width = 0.38 if meta["oot_enabled"] else 0.68
    train_iv = ordered["iv_part"].to_numpy(dtype=float)
    bars_train = ax.bar(
        x - (width / 2.0 if meta["oot_enabled"] else 0.0), train_iv,
        width, color=_TRAIN_COLOR, alpha=0.82, label="Train IV contribution",
        zorder=2)
    bars_oot = None
    if meta["oot_enabled"]:
        bars_oot = ax.bar(
            x + width / 2.0, ordered["oot_iv_part"].to_numpy(dtype=float),
            width, color=_OOT_COLOR, alpha=0.75, label="OOT IV contribution",
            zorder=2)
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
    ax.set_xticklabels(short_labels, rotation=20, ha="right",
                       fontsize=_FS_TICK)
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel("IV contribution", fontsize=_FS_AXIS_LABEL)
    ax.set_title("IV contribution (train vs OOT)" if meta["oot_enabled"]
                 else "IV contribution (train only, OOT disabled)",
                 fontsize=_FS_PANEL_TITLE, loc="left", pad=8)
    caption = "Train IV={}; OOT IV={}; missing-bin share {}".format(
        _fmt_float(info.get("iv", np.nan)),
        _fmt_float(info.get("oot_iv", np.nan)),
        _fmt_pct(info.get("iv_missing_share", np.nan)))
    if np.isfinite(iv_train) and iv_train > iv_floor:
        max_index = int(np.nanargmax(train_iv))
        caption += "\nTop bin {} (n={}, {:.1%} of train IV)".format(
            short_labels[max_index], int(train_n[max_index]),
            train_iv[max_index] / iv_train)
        if 0 < train_n[max_index] < min_bin_n:
            caption += "; small bin, concentrated contribution"
    ax.text(0.01, 0.98, caption, transform=ax.transAxes,
            fontsize=_FS_NOTE, color="#555555", va="top", linespacing=1.35,
            bbox=dict(facecolor="white", alpha=0.86,
                      edgecolor="none", pad=1.2), zorder=6)
    if np.isfinite(iv_train) and iv_train > iv_floor:
        if count <= 8:
            for index in range(count):
                value = train_iv[index]
                share = value / iv_train
                ax.text(
                    index - (width / 2.0 if meta["oot_enabled"] else 0.0),
                    value, "{:.1%}".format(share), ha="center",
                    va="bottom", fontsize=_FS_TICK)
    else:
        ax.text(0.5, 0.55, "IV~0: contribution share unstable, hidden",
                transform=ax.transAxes, ha="center",
                fontsize=_FS_AXIS_LABEL, color=_THRESHOLD_COLOR)
    values = np.r_[train_iv, ordered["oot_iv_part"].to_numpy(dtype=float)] \
        if meta["oot_enabled"] else train_iv
    finite = values[np.isfinite(values)]
    upper = float(finite.max()) * 1.35 if len(finite) else 1.0
    ax.set_ylim(0, max(upper, 1e-6))
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def _trend_legend_handles(oot_enabled, has_low_sample, has_single_class,
                          has_iv_na):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_TRAIN_COLOR, alpha=0.82,
                     label="Train samples")]
    if oot_enabled:
        handles.append(Patch(facecolor=_OOT_COLOR, alpha=0.80,
                             label="OOT samples"))
    handles.extend([
        Line2D([], [], color="#4B4B4B", marker="o", lw=1.4,
               label="Period bad rate"),
        Line2D([], [], color=_TRAIN_COLOR, ls=":", lw=1.2,
               label="Train overall bad rate"),
    ])
    if oot_enabled:
        handles.append(Line2D([], [], color=_OOT_COLOR, ls=":", lw=1.2,
                              label="OOT overall bad rate"))
    handles.extend([
        Patch(facecolor="#6A51A3", alpha=0.78, label="Period IV"),
        Line2D([], [], color=_TRAIN_COLOR, ls="--", lw=1.2,
               label="Train overall IV"),
    ])
    if has_low_sample:
        handles.append(Line2D([], [], color=_THRESHOLD_COLOR, marker="v",
                              ls="none", label="Low-sample period"))
    if has_single_class:
        handles.append(Line2D([], [], color=_THRESHOLD_COLOR, marker="x",
                              ls="none", label="Single-class period"))
    if has_iv_na:
        handles.append(Line2D([], [], color=_THRESHOLD_COLOR, marker="x",
                              ls="none", label="IV = NA"))
    return handles


def _plot_time_trend(axes, periods, matrices, info, period_context,
                     min_period_n, tick_labels, limits=None):
    n_periods = len(periods)
    x = np.arange(n_periods)
    ax_n, ax_bad, ax_iv = axes
    oot_enabled = period_context["first_oot"] is not None
    for axis in axes:
        _draw_period_overlay(axis, periods, period_context["first_oot"])
    ax_n.bar(x, periods["n_train"].to_numpy(dtype=float), width=0.84,
             color=_TRAIN_COLOR, alpha=0.82, label="Train", zorder=2)
    if oot_enabled:
        ax_n.bar(x, periods["n_oot"].to_numpy(dtype=float), width=0.84,
                 bottom=periods["n_train"].to_numpy(dtype=float),
                 color=_OOT_COLOR, alpha=0.8, label="OOT", zorder=2)
    low_period = periods["n_total"].to_numpy(dtype=float) < min_period_n
    if low_period.any():
        ax_n.scatter(x[low_period],
                     periods["n_total"].to_numpy(dtype=float)[low_period],
                     marker="v", s=26, color=_THRESHOLD_COLOR, zorder=4,
                     label="Low-sample period")
    ax_n.set_ylabel("Samples", fontsize=_FS_AXIS_LABEL)
    ax_n.tick_params(axis="y", labelsize=_FS_TICK)
    sample_upper = limits["samples"] if limits else None
    if sample_upper is not None:
        ax_n.set_ylim(0, sample_upper)
    if period_context["first_oot"] is not None:
        ax_n.text(period_context["first_oot"] - 0.5,
                  ax_n.get_ylim()[1] * 0.96, " Train/OOT split",
                  fontsize=_FS_NOTE, va="top", color="#333333",
                  bbox=dict(facecolor="white", alpha=0.86,
                            edgecolor="none", pad=1.0), zorder=5)

    bad_rate = matrices["bad_rate"] * 100.0
    mapped_n = matrices["mapped_n"]
    mapped_bad = matrices["bin_bad"].sum(axis=1)
    low, high = _wilson_interval(mapped_bad, mapped_n)
    valid = periods["both_classes"].to_numpy(dtype=bool) & np.isfinite(bad_rate)
    ax_bad.plot(x[valid], bad_rate[valid], color="#4B4B4B",
                marker="o", lw=1.4, label="Period bad rate", zorder=3)
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
                       marker="x", s=30, color=_THRESHOLD_COLOR, zorder=4,
                       label="Single-class period")
    ax_bad.axhline(info.get("_train_bad_rate", np.nan) * 100.0,
                   color=_TRAIN_COLOR, ls=":", lw=1.0, label="Train overall")
    if period_context["first_oot"] is not None:
        oot_mask = periods["split"].to_numpy() != "训练"
        if oot_mask.any():
            oot_bad = (matrices["bin_bad"].sum(axis=1)[oot_mask].sum()
                       / max(matrices["mapped_n"][oot_mask].sum(), 1))
            ax_bad.axhline(oot_bad * 100.0, color=_OOT_COLOR,
                           ls=":", lw=1.0, label="OOT overall")
    ax_bad.set_ylabel("Bad rate (%)", fontsize=_FS_AXIS_LABEL)
    ax_bad.tick_params(axis="y", labelsize=_FS_TICK)
    if limits:
        ax_bad.set_ylim(0, limits["bad_rate"])

    period_iv = matrices["period_iv"]
    ax_iv.bar(x, np.where(np.isfinite(period_iv), period_iv, 0.0),
              width=0.84, color="#6A51A3", alpha=0.78, label="Period IV",
              zorder=2)
    if np.isfinite(info.get("iv", np.nan)):
        ax_iv.axhline(info["iv"], color=_TRAIN_COLOR, ls="--", lw=1.0,
                      label="Train overall IV")
    invalid_iv = ~np.isfinite(period_iv)
    if invalid_iv.any():
        ax_iv.scatter(x[invalid_iv], np.zeros(int(invalid_iv.sum())),
                      marker="x", s=26, color=_THRESHOLD_COLOR, zorder=4,
                      label="IV=NA")
    ax_iv.set_ylabel("IV", fontsize=_FS_AXIS_LABEL)
    ax_iv.tick_params(axis="y", labelsize=_FS_TICK)
    if limits:
        ax_iv.set_ylim(0, limits["iv"])

    positions = _period_tick_positions(n_periods, 12)
    ax_iv.set_xticks(x[positions])
    ax_iv.set_xticklabels([tick_labels[index] for index in positions],
                          rotation=45, ha="right", fontsize=_FS_TICK)
    ax_iv.tick_params(axis="x", pad=6)
    for axis in axes[:-1]:
        axis.set_xticks([])
    for axis in axes:
        axis.set_xlim(-0.6, n_periods - 0.4)


def _masked_cmap(plt, name, bad_color):
    cmap = plt.get_cmap(name).copy()
    if hasattr(cmap, "with_extremes"):
        return cmap.with_extremes(bad=bad_color)
    cmap.set_bad(bad_color)
    return cmap


def _plot_woe_heatmap(ax, ordered, matrices, periods, period_context,
                      short_labels, min_bin_n, plt, vlim=None):
    from matplotlib.patches import Rectangle
    woe = matrices["woe"].T
    masked = np.ma.masked_invalid(woe)
    finite = np.abs(woe[np.isfinite(woe)])
    limit = (float(vlim) if vlim is not None
             else max(float(finite.max()) * 1.05, 1e-3) if len(finite) else 1.0)
    cmap = _masked_cmap(plt, "RdBu", "#D9D9D9")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-limit,
                      vmax=limit, interpolation="nearest", zorder=1)
    bins, n_periods = woe.shape
    ax.set_yticks(np.arange(bins))
    ax.set_yticklabels(short_labels, fontsize=_FS_TICK)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    positions = _period_tick_positions(n_periods, 12)
    labels = []
    for index in range(n_periods):
        label = str(periods.loc[index, "period_label"])
        if periods.loc[index, "split"] == "跨界":
            label += "*"
        labels.append(label)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[index] for index in positions],
                       rotation=45, ha="right", fontsize=_FS_TICK)
    ax.tick_params(axis="x", pad=5)
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
    cell_w, cell_h = _heatmap_cell_inches(ax, bins, n_periods)
    show_values = (cell_w >= 0.30 and cell_h >= 0.16
                   and bins * n_periods <= 400)
    if show_values:
        value_size = _heatmap_value_fontsize(cell_h)
        for row in range(bins):
            for column in range(n_periods):
                value = woe[row, column]
                if np.isfinite(value):
                    ax.text(column, row, "{:.2f}".format(value),
                            ha="center", va="center", fontsize=value_size,
                            color="white" if abs(value) > 0.62 * limit
                            else "#222222", zorder=4)
    ax.set_title("Period WOE heatmap (fixed train bins)",
                 fontsize=_FS_PANEL_TITLE, loc="left", pad=8)
    note = _wrap_text_for_axes(
        ax, "Positive = lower risk (blue); negative = higher risk (red); "
            "grey = empty/single-class; hatched = low sample or zero good/bad.",
        _FS_NOTE)
    ax.text(0.0, -0.28, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=_FS_NOTE, color="#555555", clip_on=False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label("WOE", fontsize=_FS_AXIS_LABEL)
    colorbar.ax.tick_params(labelsize=_FS_TICK)


def _plot_share_heatmap(ax, matrices, periods, period_context, short_labels,
                        plt, vmax=None):
    share = matrices["sample_pct"].T
    masked = np.ma.masked_where(matrices["bin_n"].T == 0, share)
    finite = share[np.isfinite(share)]
    upper = (float(vmax) if vmax is not None
             else max(float(finite.max()) * 1.05, 1e-3) if len(finite) else 1.0)
    cmap = _masked_cmap(plt, "Blues", "#D9D9D9")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0,
                      vmax=upper, interpolation="nearest", zorder=1)
    bins, n_periods = share.shape
    ax.set_yticks(np.arange(bins))
    ax.set_yticklabels(short_labels, fontsize=_FS_TICK)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    positions = _period_tick_positions(n_periods, 12)
    labels = []
    for index in range(n_periods):
        label = str(periods.loc[index, "period_label"])
        if periods.loc[index, "split"] == "跨界":
            label += "*"
        labels.append(label)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[index] for index in positions],
                       rotation=45, ha="right", fontsize=_FS_TICK)
    ax.tick_params(axis="x", pad=5)
    _draw_period_overlay(ax, periods, period_context["first_oot"])
    cell_w, cell_h = _heatmap_cell_inches(ax, bins, n_periods)
    show_values = (cell_w >= 0.30 and cell_h >= 0.16
                   and bins * n_periods <= 400)
    if show_values:
        value_size = _heatmap_value_fontsize(cell_h)
        for row in range(bins):
            for column in range(n_periods):
                value = share[row, column]
                if np.isfinite(value):
                    ax.text(column, row, "{:.0%}".format(value),
                            ha="center", va="center", fontsize=value_size,
                            color="white" if value > 0.55 * upper
                            else "#222222", zorder=4)
    ax.set_title("Period sample share (denominator = mapped samples in period)",
                 fontsize=_FS_PANEL_TITLE, loc="left", pad=8)
    note = _wrap_text_for_axes(
        ax, "Grey = empty bin; * = period crossing train/OOT split; "
            "coverage gaps: see reason and detail CSV.", _FS_NOTE)
    ax.text(0.0, -0.28, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=_FS_NOTE, color="#555555", clip_on=False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label("Sample share", fontsize=_FS_AXIS_LABEL)
    colorbar.ax.tick_params(labelsize=_FS_TICK)


def _plot_psi_panel(ax, ordered, matrices, periods, period_context,
                    short_labels, max_psi, ylim=None):
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
            label="Overall PSI (reliable periods)", zorder=4)
    low_finite = low_period & np.isfinite(period_psi)
    if low_finite.any():
        ax.scatter(x[low_finite], period_psi[low_finite], marker="^", s=42,
                   facecolors="none", edgecolors=_THRESHOLD_COLOR,
                   linewidths=1.2, zorder=6,
                   label="Low-sample PSI (not used)")
    ax.axhline(max_psi, color=_THRESHOLD_COLOR, ls="--", lw=1.0,
               label="Threshold {:g}".format(max_psi), zorder=3)
    if np.isfinite(reliable_psi).any():
        index = int(np.nanargmax(reliable_psi))
        ax.scatter([x[index]], [reliable_psi[index]], s=48,
                   facecolors="none", edgecolors="#222222", zorder=5)
        if n_periods <= 12:
            ax.text(x[index], reliable_psi[index],
                    " max reliable {}".format(
                        _fmt_float(reliable_psi[index], 4)),
                    fontsize=_FS_NOTE, va="bottom")
        local_upper = max(float(np.nanmax(reliable_psi)) * 1.35,
                          max_psi * 1.2, 1e-6)
    else:
        finite_all = period_psi[np.isfinite(period_psi)]
        local_upper = (max(float(finite_all.max()) * 1.35, max_psi * 1.2, 1e-6)
                       if len(finite_all) else 1.0)
        ax.text(0.5, 0.55,
                "No period reaches the min sample size\n"
                "PSI shown for reference only",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_FS_NOTE, color=_THRESHOLD_COLOR)
    upper = float(ylim) if ylim is not None else local_upper
    if n_periods <= 12:
        for index in range(n_periods):
            if np.isfinite(period_psi[index]) and not low_period[index]:
                ax.text(index, bottom[index],
                        _fmt_float(period_psi[index], 4),
                        ha="center", va="bottom",
                        fontsize=_FS_CELL_MIN, color="#333333")
    for index in np.flatnonzero(low_finite):
        value = period_psi[index]
        if value > upper:
            ax.annotate(_fmt_float(value, 4),
                        xy=(x[index], upper * 0.96),
                        xytext=(x[index], upper * 0.82), ha="center",
                        fontsize=_FS_CELL_MIN, color=_THRESHOLD_COLOR,
                        arrowprops=dict(arrowstyle="-|>",
                                        color=_THRESHOLD_COLOR, lw=0.6))
    positions = _period_tick_positions(n_periods, 14)
    labels = [str(periods.loc[index, "period_label"]) for index in positions]
    ax.set_xticks(x[positions])
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=_FS_TICK)
    ax.tick_params(axis="x", pad=5)
    _draw_period_overlay(ax, periods, period_context["first_oot"])
    ax.set_ylabel("PSI", fontsize=_FS_AXIS_LABEL)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    psi_contrib = matrices["psi_contrib"]
    reliable_contrib = np.where(low_period[:, None], np.nan, psi_contrib)
    source = (reliable_contrib if np.isfinite(reliable_contrib).any()
              else psi_contrib)
    if np.isfinite(source).any():
        flat_index = int(np.nanargmax(source))
        period_index, bin_index = divmod(flat_index, len(ordered))
        prefix = ("Max reliable contribution"
                  if np.isfinite(reliable_contrib).any()
                  else "Max contribution, all periods")
        ax.text(0.01, 0.98,
                "{}: {} / {} ({})".format(
                    prefix, periods.loc[period_index, "period_label"],
                    short_labels[bin_index],
                    _fmt_float(source[period_index, bin_index], 4)),
                transform=ax.transAxes, fontsize=_FS_NOTE, color="#333333",
                va="top", bbox=dict(facecolor="white", alpha=0.86,
                                    edgecolor="none", pad=1.2), zorder=6)
    ax.set_xlim(-0.6, n_periods - 0.4)
    ax.set_ylim(0, upper)
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def _feature_plot_limits(analysis, max_psi):
    """统一的纵轴和色标范围，保证各面板可直接比较。"""
    matrices = analysis["matrices"]
    meta = analysis["overall_meta"]
    reason = analysis["reason"]

    def finite_max(values, default=0.0, absolute=False):
        array = np.asarray(values, dtype=float)
        if absolute:
            array = np.abs(array)
        array = array[np.isfinite(array)]
        return float(array.max()) if array.size else float(default)

    samples = max(finite_max(matrices["total_n"], 1.0) * 1.12, 1.0)

    mapped_bad = matrices["bin_bad"].sum(axis=1)
    _, ci_high = _wilson_interval(mapped_bad, matrices["mapped_n"])
    bad_points = finite_max(matrices["bad_rate"], 0.0) * 100.0
    bad_ci = finite_max(ci_high, 0.0) * 100.0
    bad_overall = [meta["train_bad_rate"] * 100.0]
    if meta["oot_enabled"]:
        bad_overall.append(meta["oot_bad_rate"] * 100.0)
    bad_rate = max([bad_points, bad_ci] + bad_overall + [1.0]) * 1.12

    iv_values = [finite_max(matrices["period_iv"], 0.0),
                 float(reason["iv_train"])
                 if np.isfinite(reason["iv_train"]) else 0.0]
    iv = max(max(iv_values) * 1.30, 1e-6)

    reliable_psi = np.asarray(matrices["period_psi"], dtype=float)
    low_period = np.asarray(matrices["low_period"], dtype=bool)
    reliable_psi = np.where(low_period, np.nan, reliable_psi)
    psi_max = finite_max(reliable_psi, 0.0)
    if psi_max <= 0:
        psi_max = finite_max(matrices["period_psi"], 0.0)
    psi = max(max(psi_max, float(max_psi)) * 1.35, 1e-6)

    woe = max(finite_max(matrices["woe"], 0.0, absolute=True) * 1.05, 1e-3)
    share = max(finite_max(matrices["sample_pct"], 0.0) * 1.05, 1e-3)
    return {"samples": samples, "bad_rate": bad_rate, "iv": iv,
            "psi": psi, "woe": woe, "share": share}


def _period_tick_labels(periods, matrices):
    labels = []
    for index in range(len(periods)):
        label = str(periods.loc[index, "period_label"])
        coverage = matrices["coverage"][index]
        if np.isfinite(coverage) and coverage < 0.999:
            label += " (cov {:.0%})".format(coverage)
        if periods.loc[index, "split"] == "跨界":
            label += "*"
        labels.append(label)
    return labels


def _build_date_meta(train, oot):
    """提取训练/OOT 时间范围，供图头和进程池任务复用。"""
    def bounds(frame):
        if frame is None or frame.empty:
            return "", ""
        values = frame["sample_datetime"]
        return (values.min().strftime("%Y-%m-%d"),
                values.max().strftime("%Y-%m-%d"))
    train_start, train_end = bounds(train)
    oot_start, oot_end = bounds(oot)
    return {
        "train_start": train_start, "train_end": train_end,
        "oot_start": oot_start, "oot_end": oot_end,
    }


def _draw_figure_header(fig, analysis, info, params):
    meta = analysis["overall_meta"]
    reason = analysis["reason"]
    feature = analysis["feature"]
    export_name = analysis["export_name"]
    date_meta = analysis["date_meta"]
    title = "Feature analysis: {}".format(feature)
    if export_name != feature:
        title += " (export name: {})".format(export_name)
    train_text = "{} ~ {}".format(
        date_meta["train_start"], date_meta["train_end"])
    if meta["oot_enabled"]:
        oot_text = "{} ~ {} (n={}, bad rate {})".format(
            date_meta["oot_start"], date_meta["oot_end"],
            _fmt_n(meta["oot_total"]),
            _fmt_pct(meta["oot_bad_rate"], digits=2))
    else:
        oot_text = "disabled"
    header_1 = (
        "Target: {} | Train: {} (n={}, bad rate {}) | OOT: {}".format(
            params["target"], train_text, _fmt_n(meta["train_total"]),
            _fmt_pct(meta["train_bad_rate"], digits=2), oot_text))
    header_2 = (
        "Train IV={} | OOT IV={} | iv_min={} | PSI={} | Missing rate={} | "
        "Ordinary bins={} | Train direction={}".format(
            _fmt_float(reason["iv_train"]),
            _fmt_float(reason["iv_oot"]),
            _fmt_float(reason["iv_min"]),
            _fmt_float(info.get("psi", np.nan)),
            _fmt_pct(info.get("missing_rate", np.nan)),
            reason["n_normal_bins"],
            _direction_en(reason["train_direction"])))
    note = ("Note: OOT is used for bin selection, not an independent validation "
            "set; OOT uncovered samples={}.".format(
                _fmt_n(meta["oot_uncovered"]))
            if meta["oot_enabled"] else
            "Note: OOT disabled; OOT metrics are NA.")
    lines = [
        (title, _FS_TITLE, "#111111", "bold"),
        (header_1, _FS_HEADER, "#111111", "normal"),
        (header_2, _FS_HEADER, "#111111", "normal"),
        (note, _FS_NOTE, "#555555", "normal"),
    ]
    y = 0.992
    for text, fontsize, color, weight in lines:
        y = _add_wrapped_fig_text(fig, 0.012, y, text, fontsize,
                                  color=color, weight=weight)
        y -= 0.006
    return max(0.60, min(0.95, y - 0.008))


def _layout_feature_figure(fig, stacked_heatmaps, header_bottom):
    rows = [("overall", 4.8)]
    rows.append(("trend_legend", 0.58))
    rows.append(("trend", 2.95))
    if stacked_heatmaps:
        rows.append(("woe_heat", 2.35))
        rows.append(("share_heat", 2.35))
    else:
        rows.append(("heat", 2.95))
    rows.append(("psi_legend", 0.58))
    rows.append(("psi", 2.35))
    rows.append(("reason", 1.05))
    heights = [row[1] for row in rows]
    grid = fig.add_gridspec(
        len(rows), 6, height_ratios=heights,
        hspace=0.56, wspace=0.36, left=0.055, right=0.965,
        top=header_bottom, bottom=0.025)
    axes = {}
    for index, (name, _) in enumerate(rows):
        if name == "overall":
            overall_wrap = grid[index, :].subgridspec(
                2, 1, height_ratios=[5.2, 0.38], hspace=0.16)
            overall_grid = overall_wrap[0].subgridspec(1, 3, wspace=0.38)
            left_grid = overall_grid[0].subgridspec(
                3, 1, height_ratios=[3.5, 0.45, 1.30], hspace=0.18)
            axes["share"] = fig.add_subplot(left_grid[0])
            axes["share_legend"] = fig.add_subplot(left_grid[1])
            axes["table"] = fig.add_subplot(left_grid[2])
            axes["woe"] = fig.add_subplot(overall_grid[1])
            axes["iv"] = fig.add_subplot(overall_grid[2])
            axes["overall_legend"] = fig.add_subplot(overall_wrap[1])
        elif name == "trend_legend":
            axes["trend_legend"] = fig.add_subplot(grid[index, :])
        elif name == "trend":
            trend_grid = grid[index, :].subgridspec(3, 1, hspace=0.14)
            axes["trend"] = [fig.add_subplot(trend_grid[i])
                             for i in range(3)]
        elif name == "heat":
            axes["woe_heat"] = fig.add_subplot(grid[index, 0:3])
            axes["share_heat"] = fig.add_subplot(grid[index, 3:6])
        elif name == "woe_heat":
            axes["woe_heat"] = fig.add_subplot(grid[index, :])
        elif name == "share_heat":
            axes["share_heat"] = fig.add_subplot(grid[index, :])
        elif name == "psi_legend":
            axes["psi_legend"] = fig.add_subplot(grid[index, :])
        elif name == "psi":
            axes["psi"] = fig.add_subplot(grid[index, :])
        elif name == "reason":
            axes["reason"] = fig.add_subplot(grid[index, :])
    return axes


def _render_feature_figure(fig, analysis, info, params, plt, limits,
                           png_path, pdf_path=None):
    overall = analysis["overall"]
    meta = analysis["overall_meta"]
    reason = analysis["reason"]
    short_labels = [_short_bin_label(row.lower, row.upper,
                                     bool(row.is_missing))
                    for row in overall.itertuples()]
    period_context = analysis["period_context"]
    periods = period_context["periods"].reset_index(drop=True)
    matrices = analysis["matrices"]
    tick_labels = _period_tick_labels(periods, matrices)
    header_bottom = _draw_figure_header(fig, analysis, info, params)
    stacked = int(period_context["n_periods"]) > _STACK_HEATMAPS_AFTER
    axes = _layout_feature_figure(fig, stacked, header_bottom)

    handles, labels = _plot_share_panel(
        axes["share"], axes["table"], overall, meta, info,
        short_labels, params["plot_min_bin_n"])
    axes["share_legend"].axis("off")
    axes["share_legend"].legend(
        handles, labels, loc="center", ncol=3, frameon=False,
        fontsize=_FS_LEGEND, handlelength=1.5,
        columnspacing=1.0, handletextpad=0.5)
    _plot_woe_panel(axes["woe"], overall, meta, short_labels,
                    params["plot_min_bin_n"])
    _plot_iv_panel(axes["iv"], overall, meta, info, short_labels,
                   params["plot_min_bin_n"], params["plot_iv_floor"])
    axes["overall_legend"].axis("off")
    axes["overall_legend"].text(
        0.5, 0.5,
        "WOE / IV panels: blue = train, orange = OOT; grey hatch = "
        "missing bin; red edge = low sample; bars = contribution.",
        ha="center", va="center", fontsize=_FS_NOTE, color="#555555")

    trend_info = dict(info)
    trend_info["_train_bad_rate"] = meta["train_bad_rate"]
    _plot_time_trend(axes["trend"], periods, matrices, trend_info,
                     period_context, params["plot_min_period_n"],
                     tick_labels, limits)
    trend_handles = _trend_legend_handles(
        period_context["first_oot"] is not None,
        bool((matrices["total_n"] < params["plot_min_period_n"]).any()),
        bool((~periods["both_classes"].to_numpy(dtype=bool)).any()),
        bool((~np.isfinite(matrices["period_iv"])).any()))
    axes["trend_legend"].axis("off")
    axes["trend_legend"].text(
        0.0, 0.5, "Period trend (fixed train bins): samples / bad rate / IV",
        fontsize=_FS_PANEL_TITLE, weight="bold", va="center")
    axes["trend_legend"].legend(
        handles=trend_handles, loc="center right",
        ncol=5 if len(trend_handles) > 6 else 3, frameon=False,
        fontsize=_FS_LEGEND, handlelength=1.6,
        columnspacing=0.9, handletextpad=0.5)

    _plot_woe_heatmap(axes["woe_heat"], overall, matrices, periods,
                      period_context, short_labels,
                      params["plot_min_bin_n"], plt, vlim=limits["woe"])
    _plot_share_heatmap(axes["share_heat"], matrices, periods,
                        period_context, short_labels, plt,
                        vmax=limits["share"])
    psi_handles, psi_labels = _plot_psi_panel(
        axes["psi"], overall, matrices, periods, period_context,
        short_labels, params["max_psi"], ylim=limits["psi"])
    axes["psi_legend"].axis("off")
    axes["psi_legend"].text(
        0.0, 0.5, "Period PSI and bin contributions (vs train)",
        fontsize=_FS_PANEL_TITLE, weight="bold", va="center")
    axes["psi_legend"].legend(
        psi_handles, psi_labels, loc="center right",
        ncol=min(5, max(2, len(psi_handles))), frameon=False,
        fontsize=_FS_LEGEND, handlelength=1.6,
        columnspacing=0.9, handletextpad=0.5)

    axes["reason"].axis("off")
    reason_text = _wrap_text_for_axes(
        axes["reason"], reason["candidate_reason"], _FS_FOOTER)
    axes["reason"].text(
        0.0, 0.95, reason_text, fontsize=_FS_FOOTER, weight="bold",
        color="#1B7F3A", va="top", linespacing=1.35)

    fig.savefig(str(png_path), dpi=params["plot_dpi"],
                bbox_inches="tight", facecolor="white")
    if pdf_path is not None:
        fig.savefig(str(pdf_path), format="pdf",
                    bbox_inches="tight", facecolor="white")


def _feature_figure_size(stacked):
    return (17, 26) if stacked else (17, 23)


def _render_feature_plot(analysis, info, params, plt, plots_dir, file_stem):
    png_path = plots_dir / (file_stem + ".png")
    pdf_path = (plots_dir / (file_stem + ".pdf")
                if params["plot_pdf"] else None)
    stacked = (int(analysis["period_context"]["n_periods"])
               > _STACK_HEATMAPS_AFTER)
    fig = plt.figure(figsize=_feature_figure_size(stacked))
    try:
        _render_feature_figure(
            fig, analysis, info, params, plt,
            _feature_plot_limits(analysis, params["max_psi"]),
            png_path, pdf_path)
        status, error = "成功", ""
    except Exception as exc:
        status = "失败"
        error = "{}: {}".format(type(exc).__name__, exc)
        for path in (png_path, pdf_path):
            if path is not None and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
    finally:
        plt.close(fig)
    return {"png": png_path, "pdf": pdf_path,
            "status": status, "error": error}


_INDEX_COLUMNS = [
    "original_feature", "export_name", "plot_png", "plot_pdf",
    "plot_status", "plot_error", "candidate_reason",
    "name_collision", "collision_group",
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
        "DejaVu Sans", "Arial", "Liberation Sans",
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei",
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = set(font.name for font in font_manager.fontManager.ttflist)
    chosen = [name for name in candidates if name in available]
    if not chosen:
        chosen = ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": chosen + ["DejaVu Sans"],
        "font.size": _FS_BASE,
        "axes.titlesize": _FS_PANEL_TITLE,
        "axes.labelsize": _FS_AXIS_LABEL,
        "xtick.labelsize": _FS_TICK,
        "ytick.labelsize": _FS_TICK,
        "legend.fontsize": _FS_LEGEND,
        "figure.titlesize": _FS_TITLE,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
    })
    return plt


def _fallback_plot_result(params, plots_dir, file_stem, error):
    png_path = plots_dir / (file_stem + ".png")
    pdf_path = (plots_dir / (file_stem + ".pdf")
                if params["plot_pdf"] else None)
    return {"png": png_path, "pdf": pdf_path,
            "status": "失败", "error": error}


def _render_feature_task(payload):
    """进程池任务：只接收单个特征的小结果，不复制原始宽表。"""
    analysis, info, params, plots_dir, file_stem = payload
    try:
        plt = _setup_matplotlib()
    except Exception as exc:
        return _fallback_plot_result(
            params, plots_dir, file_stem,
            "{}: {}".format(type(exc).__name__, exc))
    return _render_feature_plot(
        analysis, info, params, plt, plots_dir, file_stem)


def _plots_archive_path(run_dir):
    return run_dir / (run_dir.name + "_plots.zip")


def _zip_plots(plots_dir, run_dir):
    archive = _plots_archive_path(run_dir)
    files = sorted(path for path in plots_dir.rglob("*") if path.is_file())
    if not files:
        if archive.exists():
            archive.unlink()
        return None
    with zipfile.ZipFile(str(archive), "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(str(path), path.relative_to(plots_dir).as_posix())
    return archive


def generate_feature_analysis_plots(data, train, oot, target, summary,
                                    bin_detail, oot_mask, run_dir,
                                    params):
    """在指定结果目录生成分析图、图片索引和分期明细。"""
    manual_features = params.get("plot_features") or None
    plot_targets = _resolve_plot_targets(
        summary, manual_features, params.get("plot_max_features"))
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    index_path = run_dir / "image_index.csv"
    time_path = run_dir / "time_bin_detail.csv"
    empty_index = pd.DataFrame(columns=_INDEX_COLUMNS)
    empty_time = pd.DataFrame(columns=_TIME_COLUMNS)
    previous_plots = set()
    if index_path.exists():
        try:
            old_index = pd.read_csv(index_path, dtype=str)
        except Exception:
            old_index = pd.DataFrame()
        for column in ("plot_png", "plot_pdf"):
            if column not in old_index:
                continue
            for value in old_index[column].dropna():
                path = Path(str(value))
                if (path.is_file()
                        and plots_dir.resolve() in path.resolve().parents):
                    previous_plots.add(path.resolve())

    def remove_plots(paths):
        for path in paths:
            try:
                path.unlink()
            except OSError:
                pass

    if not plot_targets:
        empty_index.to_csv(index_path, index=False, encoding="utf-8-sig")
        empty_time.to_csv(time_path, index=False, encoding="utf-8-sig")
        remove_plots(previous_plots)
        archive = _plots_archive_path(run_dir)
        if archive.exists():
            remove_plots([archive])
        print("没有可绘图的变量，未生成分析图；空结果目录：{}".format(run_dir))
        return {"run_dir": run_dir, "success": 0, "failed": 0,
                "index": empty_index, "time_detail": empty_time}
    try:
        plt = _setup_matplotlib()
    except RuntimeError as exc:
        print("绘图初始化失败：{}；本次仅输出筛选结果。".format(exc))
        return {"run_dir": run_dir, "success": 0, "failed": len(plot_targets),
                "index": empty_index, "time_detail": empty_time}

    labels = data[target].to_numpy(dtype=float)
    if oot is not None and oot_mask is not None:
        oot_array = np.asarray(oot_mask, dtype=bool)
    else:
        oot_array = np.zeros(len(data), dtype=bool)
    period_context = _build_period_context(
        data["sample_datetime"].to_numpy(), labels, oot_array,
        params["time_freq"])
    date_meta = _build_date_meta(train, oot)
    target_kind = "指定变量" if manual_features else "候选特征"
    print("分析图：{} 个{}；时间粒度 {}；时间段 {} 个；输出 {}"
          .format(len(plot_targets), target_kind, params["time_freq"],
                  period_context["n_periods"], run_dir))
    if period_context["first_oot"] is not None:
        print("  训练/OOT 切点位于第 {} 个时间段（{}）".format(
            period_context["first_oot"] + 1,
            period_context["periods"].loc[
                period_context["first_oot"], "period_label"]))
    if period_context["gaps"]:
        print("  时间缺口 {} 个，明细见 time_bin_detail.csv".format(
            len(period_context["gaps"])))

    n_periods = int(period_context["n_periods"])
    print("  单张图输出（全部 {} 个周期）".format(n_periods))

    export_names = {
        feature: str(feature).replace("_woe", "") for feature in plot_targets}
    name_counts = Counter(export_names.values())
    collision_groups = {name for name, count in name_counts.items()
                        if count > 1}
    detail_groups = {feature: group
                     for feature, group in bin_detail.groupby("feature")}
    info_map = summary.set_index("feature").to_dict("index")
    index_rows, time_rows = [], []
    failures = []
    items = []
    for position, feature in enumerate(plot_targets, 1):
        export_name = export_names[feature]
        file_stem = _safe_filename(
            feature if export_name in collision_groups else export_name)
        item = {
            "position": position, "feature": feature,
            "export_name": export_name, "file_stem": file_stem,
            "info": info_map.get(feature, {}),
            "analysis": None, "reason": None, "plot_result": None,
        }
        try:
            detail = detail_groups.get(feature)
            if detail is None:
                raise ValueError("bin_detail 中缺少该特征的分箱结果")
            analysis = _prepare_feature_analysis(
                feature, export_name, detail.copy(), item["info"], data,
                target, train, oot, period_context, params)
            analysis["date_meta"] = date_meta
            if manual_features and not bool(item["info"].get("selected", False)):
                analysis["reason"]["candidate_reason"] = (
                    "Requested by user; screening status: {}.".format(
                        _screen_reason_en(item["info"].get("reason", ""))))
            item["analysis"] = analysis
            item["reason"] = dict(analysis["reason"])
            time_detail = analysis["time_long"]
            time_detail.insert(0, "export_name", export_name)
            time_detail.insert(0, "feature", feature)
            item["time_detail"] = time_detail
        except Exception as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
            item["reason"] = {
                "candidate_reason": "Analysis failed: {}".format(error),
            }
            item["plot_result"] = _fallback_plot_result(
                params, plots_dir, file_stem, error)
        items.append(item)

    plot_jobs = int(params.get("plot_n_jobs", 1) or 1)
    if plot_jobs == -1:
        plot_jobs = os.cpu_count() or 1
    workers = max(1, min(plot_jobs, len(items)))
    rendered_serially = False
    if workers > 1:
        print("并行绘图：{} 个进程".format(workers))
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pending = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for index, item in enumerate(items):
                    if item["analysis"] is None:
                        continue
                    payload = (item["analysis"], item["info"], params,
                               plots_dir, item["file_stem"])
                    pending[executor.submit(
                        _render_feature_task, payload)] = index
                for done, future in enumerate(as_completed(pending), 1):
                    index = pending[future]
                    item = items[index]
                    try:
                        item["plot_result"] = future.result()
                    except Exception as exc:
                        item["plot_result"] = _fallback_plot_result(
                            params, plots_dir, item["file_stem"],
                            "{}: {}".format(type(exc).__name__, exc))
                    if done % 10 == 0 or done == len(pending):
                        print("分析绘图：{}/{}".format(done, len(pending)))
        except Exception as exc:
            print("并行绘图不可用，改为单进程：{}".format(exc))
            rendered_serially = True
    if workers == 1 or rendered_serially:
        done = 0
        total_analyses = sum(
            entry["analysis"] is not None for entry in items)
        for item in items:
            if item["analysis"] is None or item["plot_result"] is not None:
                continue
            try:
                item["plot_result"] = _render_feature_plot(
                    item["analysis"], item["info"], params, plt, plots_dir,
                    item["file_stem"])
            except Exception as exc:
                item["plot_result"] = _fallback_plot_result(
                    params, plots_dir, item["file_stem"],
                    "{}: {}".format(type(exc).__name__, exc))
            done += 1
            if done % 10 == 0 or done == total_analyses:
                print("分析绘图：{}/{}".format(done, len(items)))

    for item in items:
        if item.get("time_detail") is not None:
            time_rows.append(item["time_detail"])
        reason = item["reason"] or {"candidate_reason": ""}
        plot_result = item["plot_result"]
        if plot_result is None:
            continue
        if plot_result["status"] != "成功":
            failures.append((item["feature"], plot_result["error"]))
            print("警告：特征 {} 绘图失败：{}".format(
                item["feature"], plot_result["error"]))
        pdf_path = plot_result["pdf"]
        index_rows.append({
            "original_feature": item["feature"],
            "export_name": item["export_name"],
            "plot_png": str(plot_result["png"]),
            "plot_pdf": str(pdf_path) if pdf_path is not None else "",
            "plot_status": plot_result["status"],
            "plot_error": plot_result["error"],
            "candidate_reason": reason["candidate_reason"],
            "name_collision": item["export_name"] in collision_groups,
            "collision_group": (item["export_name"]
                                if item["export_name"] in collision_groups
                                else ""),
        })
    index_frame = pd.DataFrame(index_rows, columns=_INDEX_COLUMNS)
    time_frame = (pd.concat(time_rows, ignore_index=True)
                  if time_rows else empty_time)
    index_frame.to_csv(index_path, index=False, encoding="utf-8-sig")
    time_frame.to_csv(time_path, index=False, encoding="utf-8-sig")
    current_plots = set()
    for row in index_rows:
        if row["plot_status"] != "成功":
            continue
        for column in ("plot_png", "plot_pdf"):
            if row[column]:
                current_plots.add(Path(row[column]).resolve())
    remove_plots(previous_plots - current_plots)
    archive = None
    try:
        archive = _zip_plots(plots_dir, run_dir)
    except Exception as exc:
        print("警告：绘图压缩失败：{}".format(exc))
    if index_frame.empty:
        success, failed = 0, 0
    else:
        status_by_feature = (index_frame.groupby("original_feature")
                             ["plot_status"]
                             .apply(lambda values: bool((values == "成功").all())))
        success = int(status_by_feature.sum())
        failed = int(len(status_by_feature) - success)
    if failures:
        print("失败清单：")
        for feature, error in failures:
            print("  - {}：{}".format(feature, error))
    print("分析图完成：成功 {} 个特征，失败 {} 个特征，共 {} 张图；目录：{}"
          .format(success, failed, len(index_frame), run_dir))
    if archive is not None:
        print("绘图压缩包：{}".format(archive))
    result = {
        "run_dir": run_dir, "success": success, "failed": failed,
        "index": index_frame, "time_detail": time_frame,
        "archive": archive,
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
                          plot_results=True, plot_features=None,
                          plot_pdf=False, plot_dpi=200,
                          time_freq="week",
                          plot_min_bin_n=30, plot_min_bin_good=5,
                          plot_min_bin_bad=5, plot_min_period_n=100,
                          plot_iv_floor=1e-6,
                          n_jobs=1, feature_chunk_size=None,
                          dedup_max_features=None, dedup_sample_rows=None,
                          plot_max_features=None, plot_n_jobs=1):
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

    结果写入
    {根目录}/feature_screening_reports/{样本文件名}_{变量表文件名}_{标签名}/，
    文件名不含日期；同一组合重复运行覆盖同名文件，分析图按特征名命名；
    绘图完成后 plots 目录自动打包为同目录下的
    {样本文件名}_{变量表文件名}_{标签名}_plots.zip。

    分析绘图（默认对去重后的最终候选特征执行，不改变筛选结果）：
    plot_results: 是否生成分析图、候选原因和明细；默认开启。
    plot_features: 指定要出图的变量清单；非空时只画这些变量（不论是否入选），
        None 或空列表表示画最终候选。未进入筛选结果的变量会跳过并提示。
    plot_pdf: 是否同时导出 PDF；默认只导出 PNG。
    plot_dpi: PNG 分辨率，默认 200，支持 200~300。
    每个特征固定输出一张图，包含全部时间段。
    time_freq: 分期粒度 day/week/month，默认 week；一次运行内所有特征一致。
    plot_min_bin_n / plot_min_bin_good / plot_min_bin_bad: 图中标记
        低样本箱的阈值（样本量、好样本、坏样本），默认 30/5/5。
    plot_min_period_n: 图中标记低样本期间的阈值，默认 100。
    plot_iv_floor: 判断 IV≈0 的下限，默认 1e-6。
    以上阈值只影响图中的斜线标注和 PSI 参考线，不是筛选标准。

    大数据运行参数：
    feature_chunk_size: 每次读取/筛选的变量列数；None 表示一次性载入。
        分块模式返回的 data 只含主键、标签、时间列，以及实际绘制时的变量列。
    n_jobs: 分箱并行进程数，-1 表示全部 CPU；无 fork 的平台自动退回单进程。
    dedup_max_features: 相关性去重只比较 IV 最高的前 N 个变量；
        None 表示全部入选变量。
    dedup_sample_rows: 相关性去重最多使用的训练样本行数（固定随机种子）；
        None 表示使用全部训练样本，分块模式默认最多 20 万行。
    plot_max_features: 只绘制 IV 最高的前 N 个候选；None 表示全部。
    plot_n_jobs: 并行绘图进程数，-1 表示全部 CPU。
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
    for name, value, allow_none in [
            ("n_jobs", n_jobs, False),
            ("plot_n_jobs", plot_n_jobs, False),
            ("feature_chunk_size", feature_chunk_size, True),
            ("dedup_max_features", dedup_max_features, True),
            ("dedup_sample_rows", dedup_sample_rows, True),
            ("plot_max_features", plot_max_features, True)]:
        if value is None and allow_none:
            continue
        allow_minus_one = name in ("n_jobs", "plot_n_jobs")
        if (not isinstance(value, (int, np.integer))
                or value == 0 or (value < 1 and not (allow_minus_one and value == -1))):
            raise ValueError("{} 必须是正整数{}。".format(
                name, "或 -1" if allow_minus_one else ""))
    if n_jobs > 1 and not hasattr(os, "fork"):
        print("当前平台不支持 fork，分箱改为单进程；绘图仍可设置 plot_n_jobs。")
        n_jobs = 1
    for name, value in [
            ("plot_min_bin_n", plot_min_bin_n),
            ("plot_min_bin_good", plot_min_bin_good),
            ("plot_min_bin_bad", plot_min_bin_bad),
            ("plot_min_period_n", plot_min_period_n),
            ("plot_iv_floor", plot_iv_floor)]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(name + " 必须为非负有限数值。")
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
    if not set(keys).issubset(samples.columns):
        raise ValueError("样本表 缺少主键列。")
    _clean_variable_keys(samples, keys, "样本表")
    samples[target] = pd.to_numeric(samples[target], errors="coerce")
    if obs is not None and obs in samples:
        samples[obs] = pd.to_numeric(samples[obs], errors="coerce")

    variable_columns = pd.read_csv(root / variable_file, nrows=0).columns
    if not set(keys).issubset(variable_columns):
        raise ValueError("变量表 缺少主键列。")

    # 变量表可能混入样本表的列（标签、ID 等），一律不作为特征。
    excluded = [c for c in variable_columns
                if c in sample_columns and c not in keys]
    if excluded:
        print("已排除变量表中与样本表同名的列：{}".format("、".join(excluded)))
    if feature_prefixes is None:
        candidates = [c for c in variable_columns
                      if c not in keys and c not in sample_columns]
        strict = False
    else:
        candidates = [c for c in variable_columns
                      if c.startswith(tuple(feature_prefixes))
                      and c not in sample_columns]
        strict = True
    if not candidates:
        raise ValueError("变量表没有可用的数值特征列。")

    screen_params = {
        "n_bins": n_bins, "n_prebins": n_prebins,
        "min_bin_pct": min_bin_pct, "max_psi": max_psi,
        "min_iv": min_iv,
        "max_missing_iv_share": max_missing_iv_share,
    }

    if feature_chunk_size is None:
        variables = _read_variable_columns(
            root / variable_file, keys, candidates)
        _clean_variable_keys(variables, keys)
        features, skipped = _numeric_feature_block(
            variables, candidates, strict)
        if skipped:
            print("已跳过非数值列：{}".format("、".join(skipped)))
        if not features:
            raise ValueError("变量表没有可用的数值特征列。")

        joined = samples.merge(
            variables[keys + features], on=keys, how="left",
            validate="one_to_one", indicator=True,
        )
        matched = joined["_merge"].eq("both")
        eligible = matched & joined[target].isin([0, 1])
        if obs is not None and obs in joined:
            eligible &= joined[obs].eq(1)
        else:
            print("未发现成熟标记列，未按成熟度过滤；请确认未成熟标签没有被填成 0。")
        data = joined.loc[eligible, keys + [target] + features].copy()
        print("样本数：{:,}；匹配数：{:,}；匹配率：{:.2%}".format(
            len(samples), int(matched.sum()), matched.mean()))
        print("分析样本：{:,}；逾期率：{:.2%}；数值特征：{:,}".format(
            len(data), data[target].mean(), len(features)))
        if data[target].nunique() != 2:
            raise ValueError("分析样本必须同时包含标签 0 和 1，请检查匹配及成熟度。")
        train, oot, oot_mask, _ = _split_train_oot(
            data, target, oot_start, oot_ratio)

        rows, details = _screen_feature_set(
            features, train, oot, target, screen_params, n_jobs=n_jobs)
        summary, bin_detail = _build_summary_frames(rows, details)
        _apply_dedup(
            summary, bin_detail, train, dedup_corr,
            max_features=dedup_max_features, sample_rows=dedup_sample_rows)
        data_mode = "full"
        data_return = data
        plot_data = data
        train_for_plot, oot_for_plot = train, oot
    else:
        blocks = _column_blocks(candidates, feature_chunk_size)
        first = _read_variable_columns(root / variable_file, keys, blocks[0])
        _clean_variable_keys(first, keys)
        probe = samples.merge(
            first[keys], on=keys, how="left", validate="one_to_one",
            indicator=True)
        matched = probe["_merge"].eq("both").to_numpy()
        eligible = matched & samples[target].isin([0, 1]).to_numpy()
        if obs is not None and obs in samples:
            eligible &= samples[obs].eq(1).to_numpy()
        else:
            print("未发现成熟标记列，未按成熟度过滤；请确认未成熟标签没有被填成 0。")
        sample_meta = samples.loc[eligible, keys + [target]].copy()
        sample_meta[target] = samples.loc[eligible, target].to_numpy()
        print("样本数：{:,}；匹配数：{:,}；匹配率：{:.2%}".format(
            len(samples), int(matched.sum()), matched.mean()))
        print("分析样本：{:,}；逾期率：{:.2%}；数值特征：{:,}".format(
            len(sample_meta), sample_meta[target].mean(), len(candidates)))
        if sample_meta[target].nunique() != 2:
            raise ValueError("分析样本必须同时包含标签 0 和 1，请检查匹配及成熟度。")
        train_meta, oot_meta, oot_mask, _ = _split_train_oot(
            sample_meta, target, oot_start, oot_ratio)

        features, skipped = [], []
        rows, details = [], []
        for block_index, block in enumerate(blocks):
            frame = (first if block_index == 0
                     else _read_variable_columns(
                         root / variable_file, keys, block))
            if block_index:
                _clean_variable_keys(frame, keys)
            block_features, block_skipped = _numeric_feature_block(
                frame, block, strict)
            skipped.extend(block_skipped)
            if not block_features:
                continue
            features.extend(block_features)
            merged = sample_meta[keys + [target]].merge(
                frame[keys + block_features], on=keys, how="left",
                validate="one_to_one")
            if oot_mask is None:
                train_block = merged[[target] + block_features]
                oot_block = None
            else:
                train_block = merged.loc[
                    ~oot_mask, [target] + block_features]
                oot_block = merged.loc[oot_mask, [target] + block_features]
            block_rows, block_details = _screen_feature_set(
                block_features, train_block, oot_block, target,
                screen_params, n_jobs=n_jobs, progress=False)
            rows.extend(block_rows)
            details.extend(block_details)
            print("分块筛选：{}/{} 列（块大小 {}）".format(
                len(features), len(candidates), feature_chunk_size))
        if skipped:
            print("已跳过非数值列：{}".format("、".join(skipped)))
        if not features:
            raise ValueError("变量表没有可用的数值特征列。")

        summary, bin_detail = _build_summary_frames(rows, details)
        if dedup_corr is not None:
            selected_features = summary.loc[
                summary["selected"], "feature"].tolist()
            if selected_features:
                dedup_features = _dedup_feature_list(
                    selected_features, dedup_max_features)
                effective_rows = dedup_sample_rows
                if effective_rows is None:
                    effective_rows = min(len(train_meta), 200000)
                    print("分块模式默认使用最多 {:,} 行训练样本做相关性去重"
                          .format(effective_rows))
                train_positions = (np.arange(len(sample_meta))
                                   if oot_mask is None
                                   else np.flatnonzero(~oot_mask))
                if effective_rows < len(train_positions):
                    rng = np.random.RandomState(20240101)
                    picked = np.sort(rng.choice(
                        len(train_positions), size=int(effective_rows),
                        replace=False))
                    train_positions = train_positions[picked]
                dedup_base = sample_meta.iloc[train_positions][
                    keys].reset_index(drop=True)
                detail_groups = {
                    feature: group for feature, group in
                    bin_detail.loc[bin_detail["feature"].isin(dedup_features)]
                    .groupby("feature")}
                woe_matrix = _load_woe_blocks(
                    root / variable_file, keys, dedup_features,
                    detail_groups, dedup_base, feature_chunk_size)
                used, kept = _dedup_from_woe(
                    summary, woe_matrix, dedup_corr)
                print("相关性去重：参与 {} 个 -> 保留 {} 个（阈值 {:g}）".format(
                    used, kept, dedup_corr))
        data_mode = "chunk"
        data_return = sample_meta
        plot_data = None
        train_for_plot = (train_meta[[target, "sample_datetime"]]
                          if oot_mask is not None
                          else sample_meta[[target, "sample_datetime"]])
        oot_for_plot = (oot_meta[[target, "sample_datetime"]]
                        if oot_mask is not None else None)
    candidates = summary.loc[summary["selected"], ["feature"]].reset_index(drop=True)
    candidates["feature"] = candidates["feature"].str.replace("_woe", "", regex=False)
    candidates = candidates.drop_duplicates().reset_index(drop=True)
    if plot_results and plot_features is not None:
        if isinstance(plot_features, str):
            plot_features = [plot_features]
        requested = []
        for name in plot_features:
            name = str(name).strip()
            if name and name not in requested:
                requested.append(name)
        known = set(summary["feature"])
        unknown = [name for name in requested if name not in known]
        if unknown:
            print("警告：以下变量不在筛选结果中，已跳过：{}".format(
                "、".join(unknown)))
        plot_features = [name for name in requested if name in known] or None
        if requested and plot_features is None:
            raise ValueError("plot_features 中没有可绘制的变量。")
    run_name = "{}_{}_{}".format(
        _safe_filename(Path(sample_file).stem, max_len=40),
        _safe_filename(Path(variable_file).stem, max_len=40),
        _safe_filename(target, max_len=40))
    run_dir = (root / "feature_screening_reports" / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("特征筛选结果", summary), ("特征分箱结果", bin_detail),
                        ("候选变量清单", candidates)]:
        frame.to_csv(run_dir / "{}_{}.csv".format(run_name, name),
                     index=False, encoding="utf-8-sig")
    print("入选特征：{}；结果目录：{}".format(len(candidates), run_dir))
    if plot_results:
        if data_mode == "chunk":
            plot_targets = _resolve_plot_targets(
                summary, plot_features, plot_max_features, verbose=False)
            plot_data = _load_feature_blocks(
                root / variable_file, keys, plot_targets,
                sample_meta, feature_chunk_size, strict)
            data_return = plot_data
        analysis_params = {
            "target": target,
            "min_iv": float(min_iv),
            "max_psi": float(max_psi),
            "max_missing_iv_share": max_missing_iv_share,
            "dedup_corr": dedup_corr,
            "plot_features": plot_features,
            "plot_max_features": plot_max_features,
            "plot_n_jobs": int(plot_n_jobs),
            "plot_pdf": bool(plot_pdf),
            "plot_dpi": int(plot_dpi),
            "time_freq": time_freq,
            "plot_min_bin_n": int(plot_min_bin_n),
            "plot_min_bin_good": int(plot_min_bin_good),
            "plot_min_bin_bad": int(plot_min_bin_bad),
            "plot_min_period_n": int(plot_min_period_n),
            "plot_iv_floor": float(plot_iv_floor),
        }
        LAST_ANALYSIS = generate_feature_analysis_plots(
            plot_data, train_for_plot, oot_for_plot, target, summary,
            bin_detail, oot_mask, run_dir, analysis_params)
    else:
        print("分析绘图已关闭（plot_results=False）。")
    return data_return, summary, bin_detail, candidates


if __name__ == "__main__":
    # 固定数据根目录，与脚本所在位置无关。
    data, summary, bin_detail, candidates = run_feature_screening(DEFAULT_ROOT_DIR)
    try:
        from IPython.display import display
        display(summary.head(30))
    except ImportError:
        print(summary.head(30).to_string(index=False))
