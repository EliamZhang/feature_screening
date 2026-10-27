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
"""
from pathlib import Path
from datetime import datetime
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


def run_feature_screening(root_dir, min_iv=0.02, n_bins=5,
                          n_prebins=20, min_bin_pct=0.05,
                          sample_file="samples.csv",
                          variable_file="variable_bank_statement_surplus.csv",
                          target="y_fpd15_ovd", obs_col=None,
                          feature_prefixes=FEATURE_PREFIXES,
                          oot_start=None, oot_ratio=0.3,
                          max_psi=0.25, max_missing_iv_share=0.5,
                          dedup_corr=0.8):
    """读取指定的样本表/变量表，按指定标签筛选特征。

    返回 data, summary, bin_detail, candidates。
    sample_file / variable_file: 根目录下的样本表、变量表文件名。
    target: 标签列；obs_col 为成熟标记列，None 时按 target 自动匹配。
    feature_prefixes: 参与筛选的列前缀；None 表示变量表除主键外的全部列。
    oot_start / oot_ratio: 仅按 sample_datetime 时间切分，oot_start 优先；
        oot_ratio 取最近的样本占比，切点落在时间点边界上；
        两者都为 None 时不做 OOT 评估。切点只在训练期拟合。
    max_psi: OOT 与训练集分箱占比的最大允许 PSI。
    max_missing_iv_share: 缺失箱 IV 占整体 IV 的上限；None 不限制。
    dedup_corr: 入选变量间 WOE 相关去重阈值（按 iv_min 降序贪心保留）；
        None 表示不去重。
    min_iv 是 IV 阈值（有 OOT 时取训练与 OOT 的较小值）；
    n_bins/n_prebins/min_bin_pct 控制分箱。
    """
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

    if feature_prefixes is None:
        features = [c for c in variables.columns if c not in keys]
    else:
        features = [c for c in variables.columns
                    if c.startswith(tuple(feature_prefixes))]
    if not features:
        raise ValueError("变量表没有可用的特征列。")
    for col in features:
        try:
            numeric = pd.to_numeric(variables[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("特征 {} 含非数值内容，请检查。".format(col)) from exc
        default = -1 if col.startswith("bank_txn_lender") else -999999.99999
        variables[col] = numeric.replace([default, np.inf, -np.inf], np.nan)

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
    return data, summary, bin_detail, candidates


if __name__ == "__main__":
    # 固定数据根目录，与脚本所在位置无关。
    data, summary, bin_detail, candidates = run_feature_screening(DEFAULT_ROOT_DIR)
    try:
        from IPython.display import display
        display(summary.head(30))
    except ImportError:
        print(summary.head(30).to_string(index=False))
