import os
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_log(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'log csv not found: {csv_path}')

    df = pd.read_csv(csv_path)

    # 基本字段检查
    required_cols = [
        'update',
        'avg_episode_reward',
        'eval_rule_win', 'eval_rule_p1', 'eval_rule_p2', 'eval_rule_p3',
        'eval_rand_win', 'eval_rand_p1', 'eval_rand_p2', 'eval_rand_p3',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f'CSV missing columns: {missing}\n'
            f'found columns: {list(df.columns)}'
        )

    # 排序、去重（如断点续训重复写某个 update）
    df = df.sort_values('update').drop_duplicates(subset=['update'], keep='last').reset_index(drop=True)

    # 强制数值类型（None / 空会变成 NaN）
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """
    简单滑动平均：输出与输入等长（边界用 NaN 填充以避免误导）。
    window <= 1: 原样返回
    """
    if window is None or window <= 1:
        return y.astype(float)

    y = y.astype(float)
    n = y.shape[0]
    out = np.full((n,), np.nan, dtype=float)

    # 只对非 NaN 段做平滑
    # 这里用卷积的方式，要求窗口内全有效才给值；否则 NaN
    valid = np.isfinite(y).astype(float)
    kernel = np.ones(window, dtype=float)

    y_sum = np.convolve(np.nan_to_num(y, nan=0.0), kernel, mode='same')
    v_sum = np.convolve(valid, kernel, mode='same')

    # v_sum == window 表示窗口内没有 NaN
    mask = v_sum >= window - 1e-9
    out[mask] = y_sum[mask] / v_sum[mask]

    return out


def _plot_avg_reward(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    smooth_window: int,
    save_path: Optional[str] = None
) -> None:
    x_a = df_a['update'].to_numpy()
    y_a = df_a['avg_episode_reward'].to_numpy()

    x_b = df_b['update'].to_numpy()
    y_b = df_b['avg_episode_reward'].to_numpy()

    y_a_s = _moving_average(y_a, smooth_window)
    y_b_s = _moving_average(y_b, smooth_window)

    plt.figure()
    plt.plot(x_a, y_a, alpha=0.25, label=f'{label_a} (raw)')
    plt.plot(x_b, y_b, alpha=0.25, label=f'{label_b} (raw)')

    if smooth_window and smooth_window > 1:
        plt.plot(x_a, y_a_s, linewidth=2.0, label=f'{label_a} (smooth={smooth_window})')
        plt.plot(x_b, y_b_s, linewidth=2.0, label=f'{label_b} (smooth={smooth_window})')

    plt.xlabel('update')
    plt.ylabel('avg_episode_reward')
    plt.title('Avg episode reward comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')


def _ema(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    指数滑动平均（EMA），alpha 越大越跟随当前点（更抖），越小越平滑（更慢）。
    输入 y 不应含 NaN（调用前先 dropna）。
    """
    y = y.astype(float)
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def _plot_eval_group(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    prefix: str,
    title: str,
    save_path: Optional[str] = None,
    *,
    smooth_kind: str = 'ema',   # 'none' | 'ma' | 'ema'
    ma_window: int = 5,         # 用于 smooth_kind='ma'
    ema_alpha: float = 0.35,    # 用于 smooth_kind='ema'
) -> None:
    """
    prefix: 'eval_rule' 或 'eval_rand'
    在“有评估值”的 update 上画折线（不是散点云），并可选平滑以呈现趋势。
    """
    cols = [f'{prefix}_win', f'{prefix}_p1', f'{prefix}_p2', f'{prefix}_p3']
    names = ['win', '+1', '+2', '+3']

    def _extract_series(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
        sub = df[['update', col]].dropna().sort_values('update')
        x = sub['update'].to_numpy()
        y = sub[col].to_numpy().astype(float)
        return x, y

    def _smooth(y: np.ndarray) -> np.ndarray:
        if smooth_kind == 'none':
            return y
        if smooth_kind == 'ma':
            # 移动平均：这里用简单卷积，输出长度不变（边界用 same）
            if ma_window <= 1:
                return y
            k = np.ones(ma_window, dtype=float)
            return np.convolve(y, k / k.sum(), mode='same')
        if smooth_kind == 'ema':
            return _ema(y, ema_alpha)
        raise ValueError(f'unknown smooth_kind: {smooth_kind}')

    plt.figure()

    # 画法：每个指标（win/+1/+2/+3）分别画两条线（run1/run2）
    for col, name in zip(cols, names):
        x1, y1 = _extract_series(df_a, col)
        x2, y2 = _extract_series(df_b, col)

        if len(x1) > 0:
            y1s = _smooth(y1)
            plt.plot(
                x1, y1s,
                linewidth=2.0,
                label=f'{label_a} {name}'
            )

        if len(x2) > 0:
            y2s = _smooth(y2)
            plt.plot(
                x2, y2s,
                linewidth=2.0,
                label=f'{label_b} {name}'
            )

    plt.xlabel('update')
    plt.ylabel('rate')
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)

    # 图例放到图外，避免遮挡曲线
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_a', type=str, default='results/1', help='dir containing train_log.csv')
    parser.add_argument('--exp_b', type=str, default='results/2', help='dir containing train_log.csv')
    parser.add_argument('--label_a', type=str, default='exp_a')
    parser.add_argument('--label_b', type=str, default='exp_b')
    parser.add_argument('--smooth', type=int, default=25, help='moving average window for avg reward (<=1 means off)')
    parser.add_argument('--save_dir', type=str, default=None, help='if set, save figures to this dir')
    parser.add_argument('--show', action='store_true', help='show plots interactively')
    args = parser.parse_args()

    csv_a = os.path.join(args.exp_a, 'train_log.csv')
    csv_b = os.path.join(args.exp_b, 'train_log.csv')

    df_a = _read_log(csv_a)
    df_b = _read_log(csv_b)

    save_dir = args.save_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 1) avg_reward comparison (with smoothing)
    _plot_avg_reward(
        df_a, df_b,
        label_a=args.label_a,
        label_b=args.label_b,
        smooth_window=args.smooth,
        save_path=None if save_dir is None else os.path.join(save_dir, 'avg_reward.png')
    )

    # 2) vs rule-based
    _plot_eval_group(
        df_a, df_b,
        label_a=args.label_a,
        label_b=args.label_b,
        prefix='eval_rule',
        title='Evaluation vs RuleBasedAgent (win / +1 / +2 / +3)',
        save_path=None if save_dir is None else os.path.join(save_dir, 'eval_vs_rule.png')
    )

    # 3) vs random
    _plot_eval_group(
        df_a, df_b,
        label_a=args.label_a,
        label_b=args.label_b,
        prefix='eval_rand',
        title='Evaluation vs RandomAgent (win / +1 / +2 / +3)',
        save_path=None if save_dir is None else os.path.join(save_dir, 'eval_vs_random.png')
    )

    if args.show or save_dir is None:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
