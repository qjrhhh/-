# -*- coding: utf-8 -*-
"""
開源證券《分鐘資金流因子的構建方法——市場微觀結構系列(31)》 聚寬(JoinQuant)真實數據版
=====================================================================================
運行環境：聚寬「研究環境」(Python 3.6.7 / numpy 1.19.5 / pandas 0.23.4 / scipy 1.2.0)
股票池  ：沪深300（按月取成分股快照，逐日只在當日成分股內做橫截面）
區間    ：2024-01-01 ~ 2026-12-20（超過今天的部分自動截斷到「昨天」）

用法：整份貼進研究環境的一個 cell 運行；或存成 .py 後 `%run minute_flow_jq.py`。
每次運行都要逐隻股票下載 1 分鐘行情，較慢（約 20~60 分鐘）；數據與結果均不寫磁盤，
數據取到後直接在內存中使用，結果只 print / plt.show() 顯示，中斷後需重新開始。

----------------------------------------------------------------------------
【重要：與研報不同的地方，請務必閱讀】
1. 聚寬沒有「逐筆成交 + 大/小單」數據，無法像研報那樣用逐筆合成分鐘大單/小單淨流入。
   本代碼用 1 分鐘 K 線做「代理指標」：
       分鐘淨流入 = 分鐘成交額 × 方向(CLV 或 sign)
       大單 = 當天成交額最大的前 20% 分鐘（BIG_Q=0.8）的淨流入
       小單 = 其餘分鐘的淨流入
   所以「大單/小單」只是放量/縮量分鐘的資金流向，絕對數值與研報不可直接比較。
   若你有真實逐筆數據，只要替換 process_stock() 裡 big / small 兩個 (天,240) 數組即可，其餘流程不用動。
2. 「原始資金流殘差因子」(對照基準) 優先用聚寬日度 get_money_flow（大單=超大單+大單，小單=小單）；
   取不到時退回用分鐘代理流入的全天加總。
3. 研報只寫了「標準化 + 動量中性化」，沒寫是否除以成交額。真實數據裡不同股票成交額差異極大，
   不處理的話因子會變成「成交額大小因子」，所以默認用 過去20日流入 / 過去20日成交額（NORMALIZE_BY_AMOUNT）。
4. 情景因子的大盤指標用上證指數（研報原文），想換成沪深300 把 MARKET_INDEX 改成 '000300.XSHG'。
5. 日度因子對應「當日收盤後」，forward_ret = 次日 close / 當日 close - 1，因子窗口含當日（過去20個交易日）。
6. 樣本只有約 300 隻股票、不到 3 年，ICIR / 信息比率的波動會很大，不要期望複現研報 3~4 的量級。
"""
import time
import warnings
import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

from jqdata import *   # 聚寬研究環境：get_price / get_trade_days / get_index_stocks / get_money_flow

mpl.rcParams['font.family'] = 'serif'          # 聚寬研究環境下 serif 可正常顯示中文
mpl.rcParams['axes.unicode_minus'] = False

# ============================================================
# 參數
# ============================================================
START_DATE = '2024-01-01'
END_DATE = '2026-12-20'
STOCK_INDEX = '000300.XSHG'      # 股票池
MARKET_INDEX = '000001.XSHG'     # 情景因子使用的大盤指數（研報：上證指數）

WARMUP_DAYS = 30                 # 起始日前多取的交易日，用於讓第一天就有完整的20日窗口
ROLL_WINDOW = 20                 # 過去20日
ROLL_MIN = 15                    # 窗口內至少有效天數（停牌等缺失天數會被跳過）
MIN_PER_DAY = 240
MIN_BARS = 200                   # 一天至少有這麼多根分鐘線才算有效交易日

BIG_Q = 0.8                      # 大單代理：當天成交額分位數 >= BIG_Q 的分鐘
FLOW_DIRECTION = 'clv'           # 'clv'：(2C-H-L)/(H-L)；'sign'：sign(C-前收)
NORMALIZE_BY_AMOUNT = True       # 因子 = 20日流入 / 20日成交額
CUT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]   # 價格切割比例（對應研報圖1、圖2）
SCENARIO_RATIO = 0.5
SCENARIO_MINUTES = 60            # 情景：用第1小時

# 時段（分鐘序號，0 = 9:31 那根，239 = 15:00 那根）
PERIOD_IDX = {
    '第1小時': np.arange(0, 60),
    '第2小時': np.arange(60, 120),
    '第3小時': np.arange(120, 180),
    '第4小時': np.arange(180, 240),
    '前半小時': np.arange(0, 30),
    # 整點：9:30/10:00/10:30/11:00/13:00/13:30/14:00/14:30 之後5分鐘，共40分鐘
    '整點時段': np.concatenate([np.arange(m, m + 5) for m in [0, 30, 60, 90, 120, 150, 180, 210]]),
}
PERIOD_NAMES = ['第1小時', '第2小時', '第3小時', '第4小時', '前半小時', '整點時段']
FLOW_NAMES = [('大單', 'big'), ('小單', 'small')]

# 研報給出的參考值（僅用於最後對照打印）
REPORT_REF = {
    '切割-振幅-大單': (0.041, 3.23, 0.045, 3.34, 15.9, 3.39),
    '切割-振幅-小單': (-0.047, -2.96, -0.054, -3.32, 17.7, 3.02),
    '時段-前半小時-大單': (0.043, 3.41, 0.049, 3.47, 16.8, 3.64),
    '時段-前半小時-小單': (-0.061, -3.47, -0.070, -3.82, 23.3, 3.65),
    '情景-5min漲跌幅-大單': (0.037, 3.88, 0.040, 3.86, 14.1, 4.04),
    '情景-5min漲跌幅-小單': (-0.054, -3.99, -0.059, -4.15, 21.2, 4.22),
}


# ============================================================
# 1. 交易日 / 股票池 / 日線
# ============================================================
def build_calendar():
    today = dt.date.today()
    end = min(pd.to_datetime(END_DATE).date(), today - dt.timedelta(days=1))
    ev = [pd.Timestamp(d) for d in get_trade_days(start_date=START_DATE, end_date=end)]
    pre = [pd.Timestamp(d) for d in get_trade_days(end_date=START_DATE, count=WARMUP_DAYS + 1)]
    pre = [d for d in pre if d < ev[0]][-WARMUP_DAYS:]
    days = pd.DatetimeIndex(pre + ev)
    return days, ev[0]


def build_universe(days):
    """每月第一個交易日取一次指數成分股快照，逐日 ffill 成「是否為成分股」矩陣"""
    snaps = []
    last_key = None
    for d in days:
        key = (d.year, d.month)
        if key != last_key:
            snaps.append((d, list(get_index_stocks(STOCK_INDEX, date=d.date()))))
            last_key = key
    stocks = sorted(set(c for _, codes in snaps for c in codes))
    member = pd.DataFrame(False, index=days, columns=stocks)
    for d, codes in snaps:
        member.loc[d:, :] = False
        member.loc[d:, codes] = True
    return stocks, member


def fetch_daily_close(stocks, days):
    out = []
    for i in range(0, len(stocks), 100):
        sub = stocks[i:i + 100]
        p = get_price(sub, start_date=days[0].date(), end_date=days[-1].date(),
                      frequency='daily', fields=['close'], skip_paused=False,
                      fq='pre', panel=False)
        out.append(p.pivot(index='time', columns='code', values='close'))
    close = pd.concat(out, axis=1)
    close.index = pd.to_datetime(close.index)
    return close.reindex(index=days, columns=stocks)


def fetch_daily_flow(stocks, days):
    """聚寬日度資金流（單位：萬元 -> 元）。取不到就返回 None。"""
    try:
        bigs, smalls = [], []
        for i in range(0, len(stocks), 50):
            sub = stocks[i:i + 50]
            mf = get_money_flow(sub, start_date=days[0].date(), end_date=days[-1].date(),
                                fields=['date', 'sec_code', 'net_amount_xl', 'net_amount_l', 'net_amount_s'])
            mf['date'] = pd.to_datetime(mf['date'])
            mf['big'] = (mf['net_amount_xl'] + mf['net_amount_l']) * 1e4
            mf['small'] = mf['net_amount_s'] * 1e4
            bigs.append(mf.pivot(index='date', columns='sec_code', values='big'))
            smalls.append(mf.pivot(index='date', columns='sec_code', values='small'))
        big = pd.concat(bigs, axis=1).reindex(index=days, columns=stocks)
        small = pd.concat(smalls, axis=1).reindex(index=days, columns=stocks)
        return big, small
    except Exception as ex:
        print('get_money_flow 取數失敗，原始基準改用分鐘代理流入：', ex)
        return None


# ============================================================
# 2. 分鐘數據 -> (天, 240) 網格
# ============================================================
FIELDS = ['open', 'high', 'low', 'close', 'money']


def fetch_minute(code, s, e, depth=0):
    """取 [s, e] 的 1 分鐘行情；超過單次上限時自動二分區間重取"""
    try:
        df = get_price(code,
                       start_date=s.strftime('%Y-%m-%d 00:00:00'),
                       end_date=e.strftime('%Y-%m-%d 23:59:59'),
                       frequency='1m', fields=FIELDS, skip_paused=False,
                       fq='pre', panel=False)
        if df is not None and 'time' in df.columns:
            df = df.set_index('time')
        return df
    except Exception:
        if (e - s).days < 20 or depth > 4:
            raise
        mid = s + (e - s) // 2
        a = fetch_minute(code, s, mid, depth + 1)
        b = fetch_minute(code, mid + pd.Timedelta(days=1), e, depth + 1)
        return pd.concat([a, b])


def to_grid(df, days):
    """時間戳 -> (交易日, 分鐘序號) ；聚寬 1m 的時間標籤是K線結束時刻：9:31..11:30, 13:01..15:00"""
    idx = pd.DatetimeIndex(df.index)
    t = np.asarray(idx.hour * 60 + idx.minute)
    slot = np.where(t <= 690, t - 571, t - 661)
    didx = days.get_indexer(idx.normalize())
    ok = (slot >= 0) & (slot < MIN_PER_DAY) & (didx >= 0) & ((t <= 690) | (t >= 781))
    g = {}
    for f in FIELDS:
        a = np.full((len(days), MIN_PER_DAY), np.nan)
        a[didx[ok], slot[ok]] = df[f].values[ok]
        g[f] = a
    return g


def clean_grid(g):
    C = g['close']
    n_bar = np.sum(~np.isnan(C), axis=1)
    day_ok = n_bar >= MIN_BARS
    Cf = pd.DataFrame(C).ffill(axis=1).bfill(axis=1).values
    O = np.where(np.isnan(g['open']), Cf, g['open'])
    H = np.where(np.isnan(g['high']), Cf, g['high'])
    L = np.where(np.isnan(g['low']), Cf, g['low'])
    M = np.nan_to_num(g['money'])
    return O, H, L, Cf, M, day_ok


def minute_features(O, H, L, C):
    prev = np.concatenate([O[:, :1], C[:, :-1]], axis=1)       # 前一分鐘收盤；每天第一分鐘用當日開盤
    with np.errstate(all='ignore'):
        ret = C / prev - 1.0
        amp = (H - L) / prev
        spread = H - L
        clv = np.where(spread > 0, (2 * C - H - L) / spread, np.sign(C - prev))
    return prev, ret, amp, clv


def process_stock(code, days):
    """單隻股票：下載1分鐘線 -> 全部需要的「每日聚合量」，只保留小數組，省內存"""
    df = fetch_minute(code, days[0], days[-1])
    if df is None or len(df) == 0:
        return None
    g = to_grid(df, days)
    O, H, L, C, M, day_ok = clean_grid(g)
    with np.errstate(all='ignore'):
        prev, ret, amp, clv = minute_features(O, H, L, C)
        direction = np.sign(C - prev) if FLOW_DIRECTION == 'sign' else clv
        sflow = M * np.nan_to_num(direction)                       # 分鐘淨流入(元)
        thr = np.percentile(M, BIG_Q * 100, axis=1)[:, None]
        big_mask = (M >= thr) & (M > 0)
        big = np.where(big_mask, sflow, 0.0)                       # ===== 若有真實逐筆數據，替換這兩行 =====
        small = np.where(big_mask, 0.0, sflow)
    money = M.sum(axis=1)
    valid = day_ok & (money > 0)

    out = {}

    def put(key, arr):
        a = np.asarray(arr, dtype=np.float64).copy()
        a[~valid] = np.nan
        out[key] = a

    flows = (('big', big), ('small', small))
    put('money', money)
    put('tot|big', big.sum(axis=1))
    put('tot|small', small.sum(axis=1))

    # ---- 切割：price / ret / amp ----
    inds = {'price': C, 'ret': ret, 'amp': amp}
    for name, ind in inds.items():
        rank = np.nan_to_num(ind).argsort(axis=1, kind='mergesort').argsort(axis=1)
        ratios = CUT_RATIOS if name == 'price' else [0.5]
        keeps = ['high', 'low'] if name == 'price' else ['high']
        for r in ratios:
            k = int(round(MIN_PER_DAY * r))
            for keep in keeps:
                mask = (rank >= MIN_PER_DAY - k) if keep == 'high' else (rank < k)
                for fn, fl in flows:
                    put('cut|%s|%.1f|%s|%s' % (name, r, keep, fn), np.where(mask, fl, 0.0).sum(axis=1))

    # ---- 時段 ----
    for pname, idx in PERIOD_IDX.items():
        for fn, fl in flows:
            put('per|%s|%s' % (pname, fn), fl[:, idx].sum(axis=1))

    # ---- 情景：只留第1小時的分鐘流入 ----
    out['f60_big'] = np.where(valid[:, None], big[:, :SCENARIO_MINUTES], 0.0).astype(np.float32)
    out['f60_small'] = np.where(valid[:, None], small[:, :SCENARIO_MINUTES], 0.0).astype(np.float32)
    return out


def load_all(stocks, days):
    res = {}
    t0 = time.time()
    for i, code in enumerate(stocks):
        try:
            r = process_stock(code, days)
        except Exception as ex:
            print('  [失敗] %s : %s' % (code, ex))
            continue
        if r is None:
            continue
        res[code] = r
        if (i + 1) % 10 == 0:
            print('  已處理 %d/%d 隻，用時 %.0f 秒' % (i + 1, len(stocks), time.time() - t0))
    return res


def build_market_indicators(days):
    """大盤分鐘指標（只取第1小時）：振幅、1min漲跌幅、5min漲跌幅、符號成交額 -> (天, 60)"""
    df = fetch_minute(MARKET_INDEX, days[0], days[-1])
    g = to_grid(df, days)
    O, H, L, C, M, day_ok = clean_grid(g)
    prev, ret1, amp, _ = minute_features(O, H, L, C)
    pad = np.repeat(O[:, :1], 5, axis=1)
    ext = np.concatenate([pad, C], axis=1)
    with np.errstate(all='ignore'):
        ret5 = C / ext[:, :MIN_PER_DAY] - 1.0
        signed_amt = M * np.sign(ret1)
    n = SCENARIO_MINUTES
    res = {'振幅': amp[:, :n], '1min漲跌幅': ret1[:, :n], '5min漲跌幅': ret5[:, :n], '符號成交額': signed_amt[:, :n]}
    for k in res:
        a = res[k].copy()
        a[~day_ok] = np.nan
        res[k] = a
    return res


# ============================================================
# 3. 因子構造（切割 / 時段 / 情景）
# ============================================================
def roll_sum(df):
    return df.rolling(ROLL_WINDOW, min_periods=ROLL_MIN).sum()


def finish_raw(num_df, den_df):
    """20日滾動流入；可選除以20日成交額"""
    num = roll_sum(num_df)
    den = roll_sum(den_df)
    raw = num / den if NORMALIZE_BY_AMOUNT else num
    return raw.where(den.notnull())


def cutting_factor_raw(agg, money):
    return finish_raw(agg, money)


def period_factor_raw(agg, money):
    return finish_raw(agg, money)


def scenario_factor_raw(F, idx_ind, money, days, stocks, ratio=SCENARIO_RATIO, win=ROLL_WINDOW):
    """過去20日第1小時共 20*60 個大盤分鐘樣本，取指標靠前 ratio 的分鐘（對所有股票是同一批分鐘），
    再加總個股在這些分鐘的資金流。F: (天, 股票, 60)"""
    nd, ns, nm = F.shape
    ind = np.where(np.isnan(idx_ind), -np.inf, idx_ind)
    num = np.full((nd, ns), np.nan)
    for d in range(win - 1, nd):
        sl = ind[d - win + 1:d + 1]
        flat = sl[np.isfinite(sl)]
        if flat.size < win * nm * 0.5:
            continue
        thr = np.percentile(flat, (1 - ratio) * 100)
        mask = sl >= thr                                   # (win, nm)
        num[d] = (F[d - win + 1:d + 1] * mask[:, None, :]).sum(axis=(0, 2))
    num_df = pd.DataFrame(num, index=days, columns=stocks)
    den = roll_sum(money)
    raw = num_df / den if NORMALIZE_BY_AMOUNT else num_df
    return raw.where(den.notnull())


# ============================================================
# 4. 標準化 + 動量中性化 -> 殘差因子
# ============================================================
def zscore_cross_section(df):
    v = df.values
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mu = np.nanmean(v, axis=1)[:, None]
        sd = np.nanstd(v, axis=1)[:, None]
    sd = np.where(sd == 0, np.nan, sd)
    return pd.DataFrame((v - mu) / sd, index=df.index, columns=df.columns)


def neutralize_momentum(fac, mom):
    F = fac.values
    M = mom.values
    out = np.full(F.shape, np.nan)
    for d in range(F.shape[0]):
        m = ~(np.isnan(F[d]) | np.isnan(M[d]))
        if m.sum() < 10:
            continue
        X = np.vstack([np.ones(m.sum()), M[d][m]]).T
        y = F[d][m]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        out[d, m] = y - X.dot(beta)
    return pd.DataFrame(out, index=fac.index, columns=fac.columns)


def make_residual(raw, mom, member):
    raw = raw.where(member)                 # 只保留當日沪深300成分股
    z = zscore_cross_section(raw)
    return neutralize_momentum(z, mom)


# ============================================================
# 5. 因子評價
# ============================================================
def evaluate_factor(fac_df, fwd_df, ev_start, n_groups=5, ann=252):
    fac = fac_df[fac_df.index >= ev_start]
    fwd = fwd_df.reindex(fac.index)
    F, R = fac.values, fwd.values
    n = len(F)
    ic = np.full(n, np.nan)
    ric = np.full(n, np.nan)
    grp = np.full((n, n_groups), np.nan)
    for i in range(n):
        m = ~(np.isnan(F[i]) | np.isnan(R[i]))
        if m.sum() < n_groups * 5:
            continue
        f, r = F[i][m], R[i][m]
        if f.std() == 0:
            continue
        ic[i] = np.corrcoef(f, r)[0, 1]
        ric[i] = np.corrcoef(stats.rankdata(f), stats.rankdata(r))[0, 1]
        order = np.argsort(f, kind='mergesort')
        for g, ix in enumerate(np.array_split(order, n_groups)):
            grp[i, g] = r[ix].mean()
    ok = ~np.isnan(ic)
    if ok.sum() < 5:
        return None
    # 多空方向用全樣本平均IC的符號一次性決定（逐日翻轉會引入前視偏差）
    sign = 1.0 if np.nanmean(ic) >= 0 else -1.0
    ls = sign * (grp[:, -1] - grp[:, 0])
    lsv = ls[ok]
    ic_v, ric_v = ic[ok], ric[ok]
    icir = ic_v.mean() / ic_v.std() if ic_v.std() > 0 else np.nan
    ricir = ric_v.mean() / ric_v.std() if ric_v.std() > 0 else np.nan
    group_ret = pd.DataFrame(grp, index=fac.index, columns=['Group%d' % (g + 1) for g in range(n_groups)])
    ls_ret = pd.Series(ls, index=fac.index)
    return dict(
        ic_mean=ic_v.mean(), icir=icir, icir_ann=icir * np.sqrt(ann),
        rank_ic_mean=ric_v.mean(), rank_icir=ricir, rank_icir_ann=ricir * np.sqrt(ann),
        ls_ann_ret=lsv.mean() * ann,
        ls_ir=(lsv.mean() / lsv.std() * np.sqrt(ann)) if lsv.std() > 0 else np.nan,
        n_days=int(ok.sum()),
        group_ret=group_ret, ls_ret=ls_ret,
        group_cum=(1 + group_ret.fillna(0)).cumprod(),
        ls_cum=(1 + ls_ret.fillna(0)).cumprod(),
    )


def metrics_table(results, keys):
    rows = []
    for k in keys:
        m = results.get(k)
        if m is None:
            continue
        rows.append([k, m['ic_mean'], m['icir'], m['icir_ann'], m['rank_ic_mean'],
                     m['rank_icir'], m['rank_icir_ann'], m['ls_ann_ret'] * 100, m['ls_ir']])
    t = pd.DataFrame(rows, columns=['因子', 'IC均值', 'ICIR(日)', 'ICIR(年化)', 'RankIC均值',
                                    'RankICIR(日)', 'RankICIR(年化)', '多空年化收益%', '信息比率'])
    return t.set_index('因子').round(3)


def avg_cross_corr(a, b, ev_start):
    A = a[a.index >= ev_start].values
    B = b[b.index >= ev_start].values
    cs = []
    for i in range(len(A)):
        m = ~(np.isnan(A[i]) | np.isnan(B[i]))
        if m.sum() > 10 and A[i][m].std() > 0 and B[i][m].std() > 0:
            cs.append(np.corrcoef(A[i][m], B[i][m])[0, 1])
    return np.mean(cs) if cs else np.nan


# ============================================================
# 6. 畫圖（對應研報各圖）
# ============================================================
def _save(fig, name):
    fig.tight_layout()
    plt.show()                                   # 改動: 不再 savefig，只顯示


def plot_cut_ratio_sweep(results):
    """圖1/圖2：不同價格切割比例的信息比率（head=較高價格，tail=較低價格）"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (fname, fk) in zip(axes, FLOW_NAMES):
        for keep, lab, c in (('high', 'head', '#c0392b'), ('low', 'tail', '#27ae60')):
            xs, ys = [], []
            for r in CUT_RATIOS:
                m = results.get('cutsweep-%.1f-%s-%s' % (r, keep, fk))
                if m is not None:
                    xs.append('%d%%' % int(r * 100))
                    ys.append(m['ls_ir'])
            ax.plot(xs, ys, marker='o', color=c, label='%s-%s' % (fname, lab))
        ax.set_title('不同價格%s分鐘資金流切割：信息比率IR' % fname)
        ax.legend()
    _save(fig, '1_切割比例_價格.png')


def plot_bars(results, labels, key_fn, title, fname, highlight=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (flow_cn, fk) in zip(axes, FLOW_NAMES):
        vals = []
        for lab in labels:
            m = results.get(key_fn(lab, flow_cn))
            vals.append(m['ls_ir'] if m is not None else np.nan)
        cols = ['#c0392b' if lab == highlight else '#2980b9' for lab in labels]
        ax.bar(range(len(labels)), vals, color=cols)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title('%s：%s' % (flow_cn, title))
        ax.set_ylabel('信息比率 IR')
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v, '%.2f' % v, ha='center', va='bottom', fontsize=8)
    _save(fig, fname)


def plot_group_curves(results, key_pattern, title, fname):
    """五分組淨值 + 多空對沖(右軸)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (flow_cn, fk) in zip(axes, FLOW_NAMES):
        m = results.get(key_pattern % flow_cn)
        if m is None:
            continue
        for c in m['group_cum'].columns:
            ax.plot(m['group_cum'].index, m['group_cum'][c], label=c, linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(m['ls_cum'].index, m['ls_cum'], color='green', linewidth=1.8, label='多空對沖(右軸)')
        ax.set_title('%s%s 5分組曲線' % (flow_cn, title))
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='lower right', fontsize=7)
        for lab in ax.get_xticklabels():
            lab.set_rotation(25)
    _save(fig, fname)


def plot_period_ls(results):
    """圖11/圖12：不同時段多空對沖曲線"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (flow_cn, fk) in zip(axes, FLOW_NAMES):
        for p in PERIOD_NAMES:
            m = results.get('時段-%s-%s' % (p, flow_cn))
            if m is not None:
                ax.plot(m['ls_cum'].index, m['ls_cum'], label=p)
        ax.set_title('不同時段%s分鐘資金流殘差因子多空對沖曲線' % flow_cn)
        ax.legend(fontsize=8)
        for lab in ax.get_xticklabels():
            lab.set_rotation(25)
    _save(fig, '11_時段多空曲線.png')


# ============================================================
# 7. 主流程
# ============================================================
def main():
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)

    print('=' * 90)
    print('1) 交易日、股票池、日線')
    days, ev_start = build_calendar()
    print('   取數區間 %s ~ %s（評價從 %s 開始），共 %d 個交易日（含 %d 日預熱）'
          % (days[0].date(), days[-1].date(), ev_start.date(), len(days), WARMUP_DAYS))
    stocks, member = build_universe(days)
    print('   %s 歷史成分股並集：%d 隻' % (STOCK_INDEX, len(stocks)))
    close = fetch_daily_close(stocks, days)
    fwd_ret = close.shift(-1) / close - 1.0                 # 次日收益
    mom = close / close.shift(ROLL_WINDOW) - 1.0            # 過去20日收益（含當日）

    print('=' * 90)
    print('2) 逐隻下載 1 分鐘行情並聚合（有緩存，可中斷續跑）')
    res = load_all(stocks, days)
    ok_stocks = [c for c in stocks if c in res]
    print('   成功 %d / %d 隻' % (len(ok_stocks), len(stocks)))
    member = member[ok_stocks]
    close = close[ok_stocks]
    fwd_ret = fwd_ret[ok_stocks]
    mom = mom[ok_stocks]

    def agg(key):
        return pd.DataFrame(dict((c, res[c][key]) for c in ok_stocks), index=days)[ok_stocks]

    money = agg('money')
    F60 = {'big': np.stack([res[c]['f60_big'] for c in ok_stocks], axis=1),
           'small': np.stack([res[c]['f60_small'] for c in ok_stocks], axis=1)}
    mkt = build_market_indicators(days)

    results = {}
    resid_store = {}

    def run(name, raw):
        resid = make_residual(raw, mom, member)
        resid_store[name] = resid
        results[name] = evaluate_factor(resid, fwd_ret, ev_start)

    # ---------- 【1】切割 ----------
    print('=' * 90)
    print('【1】分鐘資金流切割殘差因子（切割比例50%，取較高部分）')
    for cut_cn, cut_en in (('價格', 'price'), ('漲跌幅', 'ret'), ('振幅', 'amp')):
        for flow_cn, fk in FLOW_NAMES:
            run('切割-%s-%s' % (cut_cn, flow_cn),
                cutting_factor_raw(agg('cut|%s|0.5|high|%s' % (cut_en, fk)), money))
    print(metrics_table(results, ['切割-%s-%s' % (c, f) for c in ('價格', '漲跌幅', '振幅') for f, _ in FLOW_NAMES]))

    # 圖1/圖2：價格切割比例掃描
    for r in CUT_RATIOS:
        for keep in ('high', 'low'):
            for flow_cn, fk in FLOW_NAMES:
                raw = cutting_factor_raw(agg('cut|price|%.1f|%s|%s' % (r, keep, fk)), money)
                results['cutsweep-%.1f-%s-%s' % (r, keep, fk)] = evaluate_factor(
                    make_residual(raw, mom, member), fwd_ret, ev_start)

    # ---------- 【2】時段 ----------
    print('=' * 90)
    print('【2】分鐘資金流時段殘差因子')
    for p in PERIOD_NAMES:
        for flow_cn, fk in FLOW_NAMES:
            run('時段-%s-%s' % (p, flow_cn), period_factor_raw(agg('per|%s|%s' % (p, fk)), money))
    print(metrics_table(results, ['時段-%s-%s' % (p, f) for p in PERIOD_NAMES for f, _ in FLOW_NAMES]))

    # ---------- 【3】情景 ----------
    print('=' * 90)
    print('【3】分鐘資金流情景殘差因子（大盤第1小時分鐘指標，取靠前50%分鐘）')
    scen_names = ['振幅', '1min漲跌幅', '5min漲跌幅', '符號成交額']
    for s in scen_names:
        for flow_cn, fk in FLOW_NAMES:
            raw = scenario_factor_raw(F60[fk], mkt[s], money, days, ok_stocks)
            run('情景-%s-%s' % (s, flow_cn), raw)
    print(metrics_table(results, ['情景-%s-%s' % (s, f) for s in scen_names for f, _ in FLOW_NAMES]))

    # ---------- 【4】原始（日度）資金流殘差因子 ----------
    print('=' * 90)
    print('【4】原始資金流殘差因子（對照基準）')
    daily_flow = fetch_daily_flow(ok_stocks, days)
    for flow_cn, fk in FLOW_NAMES:
        if daily_flow is not None:
            num = daily_flow[0 if fk == 'big' else 1]
        else:
            num = agg('tot|%s' % fk)
        run('原始-%s' % flow_cn, finish_raw(num, money))
    print(metrics_table(results, ['原始-大單', '原始-小單']))

    # ---------- 與研報對照 ----------
    print('=' * 90)
    print('與研報(全A、2013-2025)的對照（只看方向/相對強弱，絕對值不可比）')
    rows = []
    for k, v in REPORT_REF.items():
        m = results.get(k)
        rows.append([k, v[0], m['ic_mean'] if m else np.nan, v[1], m['icir_ann'] if m else np.nan,
                     v[4], m['ls_ann_ret'] * 100 if m else np.nan, v[5], m['ls_ir'] if m else np.nan])
    print(pd.DataFrame(rows, columns=['因子', '研報IC', '本次IC', '研報ICIR', '本次ICIR(年化)',
                                      '研報多空年化%', '本次多空年化%', '研報IR', '本次IR']).set_index('因子').round(3))

    # ---------- 因子相關性（表6/表7） ----------
    print('=' * 90)
    print('因子相關性矩陣（逐日橫截面相關係數的均值）')
    for flow_cn, fk in FLOW_NAMES:
        names = ['原始-%s' % flow_cn, '切割-振幅-%s' % flow_cn, '時段-前半小時-%s' % flow_cn, '情景-5min漲跌幅-%s' % flow_cn]
        mat = pd.DataFrame(np.nan, index=names, columns=names)
        for a in names:
            for b in names:
                mat.loc[a, b] = avg_cross_corr(resid_store[a], resid_store[b], ev_start)
        print('\n%s殘差因子相關性：' % flow_cn)
        print(mat.round(2))

    # ---------- 圖 ----------
    plot_cut_ratio_sweep(results)
    plot_bars(results, ['價格', '漲跌幅', '振幅'], lambda lab, f: '切割-%s-%s' % (lab, f),
              '切割因子信息比率', '2_切割指標對比.png', highlight='振幅')
    plot_bars(results, PERIOD_NAMES, lambda lab, f: '時段-%s-%s' % (lab, f),
              '時段因子信息比率', '3_時段因子對比.png', highlight='前半小時')
    plot_bars(results, scen_names, lambda lab, f: '情景-%s-%s' % (lab, f),
              '情景因子信息比率', '4_情景因子對比.png', highlight='5min漲跌幅')
    plot_bars(results, ['原始', '切割', '時段', '情景'],
              lambda lab, f: {'原始': '原始-%s' % f, '切割': '切割-振幅-%s' % f,
                              '時段': '時段-前半小時-%s' % f, '情景': '情景-5min漲跌幅-%s' % f}[lab],
              '四類方案信息比率對比', '5_四類方案對比.png', highlight='情景')
    plot_period_ls(results)
    plot_group_curves(results, '切割-價格-%s', '分鐘資金流切割殘差因子(價格)', '6_五分組_切割價格.png')
    plot_group_curves(results, '切割-漲跌幅-%s', '分鐘資金流切割殘差因子(漲跌幅)', '6_五分組_切割漲跌幅.png')
    plot_group_curves(results, '切割-振幅-%s', '分鐘資金流切割殘差因子(振幅)', '6_五分組_切割振幅.png')
    plot_group_curves(results, '時段-前半小時-%s', '分鐘資金流時段殘差因子', '7_五分組_時段.png')
    plot_group_curves(results, '情景-5min漲跌幅-%s', '分鐘資金流情景殘差因子', '8_五分組_情景.png')
    print('\n圖表已顯示（未寫入磁盤）')
    return results, resid_store


if __name__ == '__main__' or True:
    results, resid_store = main()
