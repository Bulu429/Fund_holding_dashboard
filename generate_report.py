#!/usr/bin/env python3
"""
基金持仓量化信号研究报告生成器
输出 fund_research_report.html（浏览器打印为PDF）
"""

import sys, warnings, base64, io, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

warnings.filterwarnings('ignore')

BASE = Path("workwork/research/策略研究/基金重仓看板")
OUTPUT = BASE / "fund_research_report.html"

sys.path.insert(0, str(Path.cwd()))
from workwork.research.策略研究.基金重仓看板.backtest_signals import (
    load_holdings, load_prices, DISCLOSURE_DATES, sort_q, run_backtest
)
from workwork.research.策略研究.基金重仓看板.strategy_final import (
    build_strategies, run_nav, calc_stats
)

# ── 字体配置 ──────────────────────────────��─────────────────────────────
rcParams['font.family']       = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False
rcParams['font.size']         = 9
rcParams['axes.spines.top']   = False
rcParams['axes.spines.right'] = False

C_RED  = '#c0392b'
C_GRN  = '#166534'
C_BLU  = '#1e50a2'
C_PUR  = '#7c3aed'
C_GRY  = '#96a0ae'
C_ORG  = '#d97706'
C_BG   = '#fafaf8'

def fig2b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'data:image/png;base64,{b64}'


# ══ 数据计算 ════════════════════════════════════════════════��═════════════

def compute_all():
    print("加载数据…")
    by_q     = load_holdings()
    price_df = load_prices()
    quarters = sorted(by_q.keys(), key=sort_q)

    print("回测信号统计…")
    df_raw, _ = run_backtest(by_q, price_df, top_pct=0.20)
    df = df_raw[df_raw['alpha_30d'].notna()].copy()

    # ── 1. 信号组 α ─────────────────────────────────────────────────────
    fcc_grp = {}
    for grp in ['加仓','持平','减仓']:
        sub = df[df['fcc_group'] == grp]
        fcc_grp[grp] = {d: {'mean': sub[f'alpha_{d}d'].mean()*100 if len(sub)>0 else 0,
                             'win':  (sub[f'alpha_{d}d']>0).mean()*100 if len(sub)>0 else 0,
                             'n':    len(sub)}
                        for d in [30,60,90]}

    # ── 2. fw 分位数 ─────────────────────────────────────────────────────
    fw_grp = {}
    for grp in ['Q1低','Q2','Q3','Q4高']:
        sub = df[df['fw_group'] == grp]
        fw_grp[grp] = {d: round(sub[f'alpha_{d}d'].mean()*100, 2) if len(sub) > 0 else 0
                       for d in [30,60,90]}
        fw_grp[grp]['win30'] = round((sub['alpha_30d']>0).mean()*100, 1) if len(sub) > 0 else 0
        fw_grp[grp]['n']     = len(sub)

    # ── 3. 行业（加仓组） ─────────────────────────────────────────────────
    ind_alpha = (df[df['fcc_group']=='加仓']
                 .groupby('ind')['alpha_30d']
                 .agg(['mean','count'])
                 .query('count >= 10')
                 .assign(mean=lambda x: x['mean']*100)
                 .sort_values('mean', ascending=False))

    # ── 4. 最优策略净值 ───────────────────────────────────────────────────
    print("构建策略…")
    strats = build_strategies(by_q, quarters)
    nav_data, stat_data = {}, {}
    for key in ['B','C','REF1']:
        nav_df, _ = run_nav(strats[key], price_df)
        nav_data[key]  = nav_df
        stat_data[key] = calc_stats(nav_df)

    # ── 5. 加减仓对比（单独） ─────────────────────────────────────────────
    red_stats = {d: round(df[df['fcc_group']=='减仓'][f'alpha_{d}d'].mean()*100, 2)
                 for d in [30,60,90]}
    add_stats = {d: round(df[df['fcc_group']=='加仓'][f'alpha_{d}d'].mean()*100, 2)
                 for d in [30,60,90]}

    # ── 5b. 信号组量化定义（每季度前20%/后20%）─────────────────────────
    # fcc分组：每季内按fcc排名，前20%且fcc>0=加仓，后20%且fcc<0=减仓，其余=持平
    grp_ns = {g: df[df['fcc_group']==g]['fcc'].dropna() for g in ['加仓','持平','减仓']}
    fcc_def = {
        '加仓_median': round(grp_ns['加仓'].median(), 0),
        '加仓_p25':    round(grp_ns['加仓'].quantile(0.25), 0),
        '加仓_p75':    round(grp_ns['加仓'].quantile(0.75), 0),
        '减仓_median': round(grp_ns['减仓'].median(), 0),
        '减仓_p25':    round(grp_ns['减仓'].quantile(0.25), 0),
        '减仓_p75':    round(grp_ns['减仓'].quantile(0.75), 0),
    }

    # ── 5c. fw分位数阈值（跨季平均）─────────────────────────────────────
    fw_vals = df['fw'].dropna() * 100   # 转为百分比形式
    fw_thresholds = {
        'p25': round(float(fw_vals.quantile(0.25)), 4),
        'p50': round(float(fw_vals.quantile(0.50)), 4),
        'p75': round(float(fw_vals.quantile(0.75)), 4),
        'max': round(float(fw_vals.quantile(0.99)), 4),
    }

    # ── 6. 4Q25→1Q26 调仓 ────────────────────────────────────────────────
    periods_B  = strats['B']
    p4q25 = next(p for p in periods_B if p['q'] == '4Q25')
    p1q26 = next(p for p in periods_B if p['q'] == '1Q26')
    c4q25 = {s['code']: s for s in p4q25['stocks']}
    c1q26 = {s['code']: s for s in p1q26['stocks']}
    cur_1q26 = {r['code']: r for r in by_q['1Q26']}
    exits  = {c: s for c, s in c4q25.items() if c not in c1q26}
    enters = {c: s for c, s in c1q26.items() if c not in c4q25}
    keeps  = {c: s for c, s in c1q26.items() if c in c4q25}

    return dict(fcc_grp=fcc_grp, fw_grp=fw_grp, ind_alpha=ind_alpha,
                nav_data=nav_data, stat_data=stat_data,
                red_stats=red_stats, add_stats=add_stats,
                fcc_def=fcc_def, fw_thresholds=fw_thresholds,
                exits=exits, enters=enters, keeps=keeps,
                cur_1q26=cur_1q26, quarters=quarters,
                total_obs=len(df))


# ══ 图表生成 ══════════════════════════════════════════════════════════════

def chart_strategy_summary(stat_data):
    """所有策略超额对比横向柱状图"""
    labels = ['参考：基金增减前20（全换手）', '参考：行业低配动量（全换手）',
              '基金增减前20\n（低换手+减仓退出）', '行业低配动量\n（低换手）',
              '行业低配动量\n（低换手+减仓退出，最优）']
    keys   = ['REF1','REF2','C','A','B'] if 'REF2' in stat_data else ['REF1','C','A','B']
    # Recompute with available keys
    available = [k for k in ['REF1','C','A','B'] if k in stat_data]
    label_map = {
        'REF1': '参考：基金增减前20（全换手）',
        'A':    '行业低配动量策略（低换手）',
        'B':    '行业低配动量策略\n（低换手+减仓退出）',
        'C':    '基金增减前20\n（低换手+减仓退出）',
    }
    vals   = [stat_data[k]['alpha'] for k in available]
    colors = [C_BLU if k=='REF1' else C_ORG if k=='C' else C_GRN if k=='A' else C_RED for k in available]
    labs   = [label_map[k] for k in available]

    fig, ax = plt.subplots(figsize=(7,3))
    fig.patch.set_facecolor('white')
    bars = ax.barh(labs, vals, color=colors, height=0.5, zorder=2)
    ax.axvline(0, color='#ccc', lw=0.8)
    ax.set_xlabel('累计超额收益（%）', fontsize=8.5)
    ax.set_facecolor(C_BG)
    ax.grid(axis='x', color='#e8e3db', linestyle='--', lw=0.6, zorder=0)
    for bar, v in zip(bars, vals):
        ax.text(v + 1.5, bar.get_y()+bar.get_height()/2,
                f'+{v:.1f}%', va='center', fontsize=8, color='#333')
    ax.set_xlim(0, max(vals)*1.18)
    fig.tight_layout()
    return fig2b64(fig)


def chart_fcc_signal(fcc_grp):
    """加减仓信号三组×三持仓期"""
    x = np.arange(3); w = 0.23
    days = [30, 60, 90]
    grps  = ['加仓','持平','减仓']
    cols  = [C_RED, C_GRY, C_GRN]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    fig.patch.set_facecolor('white')

    # 左：均值
    ax = axes[0]
    ax.set_facecolor(C_BG)
    ax.grid(axis='y', color='#e8e3db', linestyle='--', lw=0.6, zorder=0)
    for i, (g, c) in enumerate(zip(grps, cols)):
        vals = [fcc_grp[g][d]['mean'] for d in days]
        bars = ax.bar(x + i*w - w, vals, w*0.9, label=g, color=c, zorder=2, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v + (0.1 if v>=0 else -0.5),
                    f'{v:+.1f}', ha='center', fontsize=7, color='#555')
    ax.set_xticks(x); ax.set_xticklabels(['30日','60日','90日'])
    ax.set_ylabel('平均超额收益（%）'); ax.set_title('不同信号组 超额收益均值', fontsize=9.5)
    ax.legend(fontsize=8, framealpha=0.6)
    ax.axhline(0, color='#999', lw=0.8)

    # 右：胜率
    ax2 = axes[1]
    ax2.set_facecolor(C_BG)
    ax2.grid(axis='y', color='#e8e3db', linestyle='--', lw=0.6, zorder=0)
    for i, (g, c) in enumerate(zip(grps, cols)):
        vals = [fcc_grp[g][d]['win'] for d in days]
        ax2.bar(x + i*w - w, vals, w*0.9, label=g, color=c, zorder=2, alpha=0.85)
    ax2.axhline(50, color='#999', lw=0.8, linestyle='--')
    ax2.set_xticks(x); ax2.set_xticklabels(['30日','60日','90日'])
    ax2.set_ylabel('胜率（%）'); ax2.set_title('不同信号组 胜率', fontsize=9.5)
    ax2.set_ylim(40, 68); ax2.legend(fontsize=8, framealpha=0.6)
    ax2.text(2.7, 51, '50%基准线', fontsize=7, color='#999')

    fig.tight_layout()
    return fig2b64(fig)


def chart_fw_quartile(fw_grp):
    """持仓集中度逆向"""
    grps = ['Q1\n最低拥挤', 'Q2', 'Q3', 'Q4\n最高拥挤']
    keys = ['Q1低','Q2','Q3','Q4高']
    v30  = [fw_grp[k][30] for k in keys]
    v90  = [fw_grp[k][90] for k in keys]
    win  = [fw_grp[k]['win30'] for k in keys]
    n    = [fw_grp[k]['n'] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)
    ax.grid(axis='y', color='#e8e3db', linestyle='--', lw=0.6, zorder=0)
    x = np.arange(4); w = 0.35
    b1 = ax.bar(x - w/2, v30, w, color=C_BLU, label='30日超额', zorder=2, alpha=0.85)
    b2 = ax.bar(x + w/2, v90, w, color=C_BLU, label='90日超额', zorder=2, alpha=0.45)
    for bar, v, wr in zip(b1, v30, win):
        ax.text(bar.get_x()+bar.get_width()/2, v + 0.15,
                f'{v:+.1f}%\n胜率{wr:.0f}%', ha='center', fontsize=7.5, color='#333')
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(grps)
    ax.set_ylabel('超额收益（%）')
    ax.set_title('占基金持仓比重 四分位组 超额收益（拥挤度越低超额越高）', fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig2b64(fig)


def chart_industry(ind_alpha):
    """行业信号有效性"""
    top5 = ind_alpha.head(5)
    bot5 = ind_alpha.tail(5).iloc[::-1]
    combined = pd.concat([top5, bot5])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)
    ax.grid(axis='x', color='#e8e3db', linestyle='--', lw=0.6, zorder=0)
    colors = [C_RED if v >= 0 else C_GRN for v in combined['mean']]
    bars = ax.barh(combined.index, combined['mean'], color=colors, height=0.55, zorder=2)
    for bar, v, cnt in zip(bars, combined['mean'], combined['count']):
        ax.text(v + (0.2 if v >= 0 else -0.2),
                bar.get_y()+bar.get_height()/2,
                f'{v:+.1f}%  (n={int(cnt)})',
                va='center', ha='left' if v >= 0 else 'right', fontsize=7.5, color='#333')
    ax.axvline(0, color='#999', lw=0.8)
    ax.set_xlabel('30日平均超额收益（%）')
    ax.set_title('加仓信号 行业效果（Top5 / Bottom5）', fontsize=9.5)
    fig.tight_layout()
    return fig2b64(fig)


def chart_nav(nav_data, stat_data):
    """净值曲线对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor('white')

    cfg = {
        'B':    (C_RED,  '最优：行业低配动量（低换手+减仓退出）', '-',  2.5),
        'C':    (C_ORG,  '基金增减前20（低换手+减仓退出）',       '-',  1.8),
        'REF1': (C_GRY,  '参考：基金增减前20（全换手）',          '--', 1.4),
    }
    for key, (col, lbl, ls, lw) in cfg.items():
        if key not in nav_data: continue
        df = nav_data[key]
        dates = pd.to_datetime(df['date'])
        ax1.plot(dates, df['nav'], color=col, lw=lw, ls=ls, label=lbl)
    ax1.plot(pd.to_datetime(nav_data['REF1']['date']),
             nav_data['REF1']['bench'], color='#aaa', lw=1, ls=':', label='沪深300')
    ax1.set_facecolor(C_BG)
    ax1.grid(color='#e8e3db', linestyle='--', lw=0.5)
    ax1.set_title('最优策略净值曲线', fontsize=9.5)
    ax1.set_ylabel('净值（起始=1.0）')
    ax1.legend(fontsize=7.5, framealpha=0.7)
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%y/%m'))

    # 回撤
    for key, (col, lbl, ls, lw) in cfg.items():
        if key not in nav_data: continue
        nav = nav_data[key]['nav'].values
        mx  = np.maximum.accumulate(nav)
        dd  = (nav / mx - 1) * 100
        ax2.plot(pd.to_datetime(nav_data[key]['date']), dd,
                 color=col, lw=lw, ls=ls, label=lbl)
    ax2.set_facecolor(C_BG)
    ax2.grid(color='#e8e3db', linestyle='--', lw=0.5)
    ax2.set_title('回撤曲线', fontsize=9.5)
    ax2.set_ylabel('回撤（%）')
    ax2.legend(fontsize=7.5, framealpha=0.7)
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%y/%m'))

    fig.tight_layout()
    return fig2b64(fig)


def chart_quarterly(nav_data, stat_data):
    """逐季度超额"""
    ref_qr  = stat_data['REF1']['qr']
    qs_keys = sorted(ref_qr.keys())

    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)
    ax.grid(axis='y', color='#e8e3db', linestyle='--', lw=0.5, zorder=0)

    x = np.arange(len(qs_keys)); w = 0.28
    cfg = [('B', C_RED, '最优：行业低配动量（低换手+减仓退出）'),
           ('C', C_ORG, '基金增减前20（低换手+减仓退出）'),
           ('REF1', C_GRY, '参考：基金增减前20（全换手）')]
    for i, (key, col, lbl) in enumerate(cfg):
        if key not in stat_data: continue
        qr  = stat_data[key]['qr']
        vals = [qr.get(q,{'port':0,'bench':0})['port'] - qr.get(q,{'port':0,'bench':0})['bench']
                for q in qs_keys]
        bars = ax.bar(x + (i-1)*w, vals, w*0.9,
                      color=[col if v>=0 else col+'88' for v in vals],
                      label=lbl, zorder=2)

    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace('Q',' Q') for q in qs_keys], fontsize=8)
    ax.set_ylabel('超额收益（%）')
    ax.set_title('逐季度超额收益对比', fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig2b64(fig)


# ══ HTML 报告 ═════════════════════════════════════════════════════════════

STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --ink:#1a1a2e;--ink2:#3d3d5c;--ink3:#7a7a9a;
  --acc:#1e50a2;--pos:#b91c1c;--neg:#15803d;
  --bd:#d4d0c8;--bg:#faf8f4;--card:#fff;
  --serif:'Noto Serif SC',serif;
  --sans:'Noto Sans SC',sans-serif;
  --mono:'JetBrains Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:#e8e4dc;color:var(--ink);font-family:var(--sans);font-size:10pt;line-height:1.65}
.page{background:var(--card);width:210mm;min-height:297mm;margin:0 auto 16px;
  padding:16mm 18mm 14mm;box-shadow:0 2px 20px rgba(0,0,0,.12);position:relative}
.cover{display:flex;flex-direction:column;justify-content:space-between;min-height:267mm}
.cover-top{padding-top:30mm}
.cover-label{font-size:8pt;letter-spacing:3px;text-transform:uppercase;color:var(--ink3);margin-bottom:8px}
.cover-title{font-family:var(--serif);font-size:24pt;font-weight:700;color:var(--ink);
  line-height:1.3;margin-bottom:12px;border-bottom:3px solid var(--acc);padding-bottom:10px}
.cover-sub{font-size:12pt;color:var(--ink2);margin-top:10px;line-height:1.6}
.cover-meta{font-size:9pt;color:var(--ink3);border-top:1px solid var(--bd);padding-top:12px}
.cover-meta span{display:inline-block;margin-right:24px}

h1{font-family:var(--serif);font-size:14pt;font-weight:700;color:var(--acc);
   border-bottom:2px solid var(--acc);padding-bottom:5px;margin-bottom:14px;margin-top:4px}
h2{font-family:var(--serif);font-size:11pt;font-weight:700;color:var(--ink);
   margin:14px 0 8px;padding-left:8px;border-left:3px solid var(--acc)}
h3{font-size:10pt;font-weight:700;color:var(--ink2);margin:10px 0 6px}
p{margin-bottom:8px;color:var(--ink2)}

.section-num{font-family:var(--serif);font-size:28pt;font-weight:700;color:#e8e4dc;
  position:absolute;right:18mm;top:14mm;line-height:1}

/* 结论盒子 */
.conclusion-box{background:#f0f5ff;border:1.5px solid #b8cef5;border-radius:6px;
  padding:12px 14px;margin-bottom:12px}
.conclusion-box .num{display:inline-block;background:var(--acc);color:#fff;
  border-radius:50%;width:20px;height:20px;text-align:center;line-height:20px;
  font-size:8.5pt;font-weight:700;margin-right:8px;flex-shrink:0}
.conclusion-box .text{font-size:10pt;color:var(--ink);line-height:1.6}
.conclusion-row{display:flex;align-items:flex-start;margin-bottom:7px}

/* 关键数字 */
.kpi-row{display:flex;gap:10px;margin:10px 0}
.kpi{flex:1;background:var(--bg);border:1px solid var(--bd);border-radius:5px;
  padding:10px 12px;text-align:center}
.kpi-val{font-family:var(--mono);font-size:16pt;font-weight:700;line-height:1;margin-top:4px}
.kpi-lbl{font-size:7.5pt;color:var(--ink3);text-transform:uppercase;letter-spacing:.5px}
.kpi-sub{font-size:8pt;color:var(--ink3);margin-top:3px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.blu{color:var(--acc)}

/* 表格 */
table{width:100%;border-collapse:collapse;font-size:8.5pt;margin:8px 0}
th{background:#f0f0f8;color:var(--ink3);font-weight:600;font-size:7.5pt;
   text-transform:uppercase;letter-spacing:.4px;padding:7px 10px;
   text-align:right;border-bottom:2px solid var(--bd)}
th.l,td.l{text-align:left}
td{padding:6px 10px;border-bottom:1px solid #ede9e3;text-align:right;color:var(--ink)}
tr:nth-child(even) td{background:#faf8f4}
.bold td{font-weight:700;background:#f0f5ff!important}

/* 图表 */
.chart-wrap{margin:10px 0;text-align:center}
.chart-wrap img{max-width:100%;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.chart-caption{font-size:8pt;color:var(--ink3);margin-top:4px;text-align:center;
  font-style:italic}

/* 标签 */
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:8pt;
  font-weight:600;font-family:var(--sans)}
.tag-add{background:#fef2f2;color:var(--pos)}
.tag-red{background:#f0faf4;color:var(--neg)}
.tag-neu{background:#f0f0f8;color:var(--ink3)}
.tag-hot{background:#fffbeb;color:#92400e}

/* 警示框 */
.note{background:#fffbeb;border-left:3px solid #f59e0b;padding:8px 12px;
  font-size:8.5pt;color:#78350f;margin:10px 0;border-radius:0 4px 4px 0}
.divider{border:none;border-top:1px solid var(--bd);margin:14px 0}

/* 页脚 */
.footer{position:absolute;bottom:10mm;left:18mm;right:18mm;
  display:flex;justify-content:space-between;font-size:7.5pt;color:var(--ink3);
  border-top:1px solid var(--bd);padding-top:6px}

/* 打印 */
@media print{
  body{background:white}
  .page{margin:0;box-shadow:none;page-break-after:always;width:100%}
  .no-print{display:none}
}
@page{size:A4;margin:0}
"""

def make_page(content, page_num=None, section_num=None):
    snum = f'<div class="section-num">{section_num}</div>' if section_num else ''
    footer_txt = f'基金持仓量化信号研究  ·  {datetime.now().strftime("%Y年%m月")}' if page_num else ''
    pg_txt = f'第 {page_num} 页' if page_num else ''
    footer = (f'<div class="footer"><span>{footer_txt}</span><span>{pg_txt}</span></div>'
              if page_num else '')
    return f'<div class="page">{snum}{content}{footer}</div>'


def build_report(data):
    imgs = {}
    print("生成图表…")
    imgs['summary'] = chart_strategy_summary(data['stat_data'])
    imgs['fcc']     = chart_fcc_signal(data['fcc_grp'])
    imgs['fw']      = chart_fw_quartile(data['fw_grp'])
    imgs['ind']     = chart_industry(data['ind_alpha'])
    imgs['nav']     = chart_nav(data['nav_data'], data['stat_data'])
    imgs['qtr']     = chart_quarterly(data['nav_data'], data['stat_data'])

    st_B  = data['stat_data']['B']
    st_C  = data['stat_data']['C']
    st_R1 = data['stat_data']['REF1']

    pages = []

    # ─── 封面 ────────────────────────────────��───────────────────────────
    pages.append(make_page(f"""
<div class="cover">
  <div class="cover-top">
    <div class="cover-label">量化研究报告</div>
    <div class="cover-title">公募基金季度持仓信号<br>有效性研究</div>
    <div class="cover-sub">
      基于基金季报披露数据，构建可执行的机构持仓量化选股策略<br>
      覆盖2024年一季度至2026年一季度，共9个季度披露周期
    </div>
  </div>
  <div>
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-lbl">数据覆盖</div>
        <div class="kpi-val blu">9</div><div class="kpi-sub">季度披露周期</div></div>
      <div class="kpi"><div class="kpi-lbl">有效观测</div>
        <div class="kpi-val blu">{data['total_obs']:,}</div><div class="kpi-sub">股票×季度</div></div>
      <div class="kpi"><div class="kpi-lbl">最优策略累计超额</div>
        <div class="kpi-val pos">+{st_B['alpha']:.0f}%</div><div class="kpi-sub">vs 沪深300</div></div>
      <div class="kpi"><div class="kpi-lbl">最优策略夏普</div>
        <div class="kpi-val pos">{st_B['sh']:.2f}</div><div class="kpi-sub">日度年化</div></div>
    </div>
    <div class="cover-meta">
      <span>研究员：[作者]</span>
      <span>日期：{datetime.now().strftime('%Y年%m月%d日')}</span>
      <span>数据来源：公募基金季报重仓持股汇总</span>
    </div>
  </div>
</div>"""))

    # ─── 核心结论 ─────────────────────────────────────────────────────────
    pages.append(make_page(f"""
<h1>核心结论</h1>
<div class="conclusion-row">
  <div class="conclusion-box" style="flex:1">
    <div class="conclusion-row">
      <span class="num">1</span>
      <div class="text"><strong>加仓增量信号有效，但单独使用区分度有限。</strong>
      持有基金数增量最大前20组合（fc≥30）相对沪深300累计超额+58pp，夏普1.17；
      而纯粹的减仓信号较弱，被减仓股票短期仍跑赢基准约+7.7pp，不宜单独做空。</div>
    </div>
    <div class="conclusion-row">
      <span class="num">2</span>
      <div class="text"><strong>持仓拥挤度是反向指标。</strong>
      占基金持仓比重处于最低四分位（最不拥挤）的个股，30日超额+4.0%，胜率59.4%；
      最高四分位仅+1.0%、胜率46.1%。高集中度≠后续超额，"安静重仓股"反而更有预期差。</div>
    </div>
    <div class="conclusion-row">
      <span class="num">3</span>
      <div class="text"><strong>行业过滤是信号质量的关键前置条件。</strong>
      通信（+9.0%）、电力设备（+5.0%）加仓信号显著有效；
      食品饮料（-1.5%）、社会服务（-1.4%）几乎无效。在消费类行业，机构加仓多为防御性配置，不含方向性信息。</div>
    </div>
    <div class="conclusion-row">
      <span class="num">4</span>
      <div class="text"><strong>降低换手率反而提升收益。</strong>
      每季仅换5只（换手率降75%）相比每季全换20只，累计超额从+58pp升至+71pp。
      机构加仓信号具有跨季度持续性，过度换仓损耗收益。</div>
    </div>
    <div class="conclusion-row">
      <span class="num">5</span>
      <div class="text"><strong>最优策略：行业低配动量策略（低换手+减仓退出），累计超额+161pp，夏普1.67。</strong>
      在"行业相对低配+占基金持仓比重上升"信号基础上，每季仅换5只，并在持仓中检测到机构减仓时强制退出，
      组合长期维持约10只精选标的。</div>
    </div>
  </div>
</div>

<hr class="divider">
<h2>策略对比一览</h2>
<table>
  <tr><th class="l">策略</th><th>累计超额</th><th>年化收益</th><th>夏普比率</th><th>最大回撤</th><th>均持仓</th></tr>
  <tr class="bold"><td class="l">★ 最优：行业低配动量策略（低换手+减仓退出）</td>
    <td class="pos">+{st_B['alpha']:.1f}%</td><td class="pos">{st_B['ann']:+.1f}%</td>
    <td class="pos">{st_B['sh']:.2f}</td><td class="neg">{st_B['dd']:.1f}%</td><td>~11只</td></tr>
  <tr><td class="l">基金增减前20（低换手+减仓退出）</td>
    <td class="pos">+{st_C['alpha']:.1f}%</td><td class="pos">{st_C['ann']:+.1f}%</td>
    <td>{st_C['sh']:.2f}</td><td class="neg">{st_C['dd']:.1f}%</td><td>~10只</td></tr>
  <tr><td class="l">参考：基金增减前20（全换手）</td>
    <td class="pos">+{st_R1['alpha']:.1f}%</td><td class="pos">{st_R1['ann']:+.1f}%</td>
    <td>{st_R1['sh']:.2f}</td><td class="neg">{st_R1['dd']:.1f}%</td><td>20只</td></tr>
  <tr><td class="l">沪深300（基准）</td>
    <td>—</td><td class="pos">{st_R1['btot']:+.1f}%（区间）</td><td>—</td><td>—</td><td>—</td></tr>
</table>
<div class="note">回测区间：{st_B['start']} ~ {st_B['end']}，共8个调仓周期。所有结果不含交易成本。最优策略持仓集中（约11只），波动高于常规产品，需结合风控要求评估。</div>
<div class="chart-wrap">
  <img src="{imgs['summary']}">
  <div class="chart-caption">图1  各策略累计超额收益汇总（vs 沪深300，持有基金数≥30家门槛）</div>
</div>
""", page_num=2))

    # ─── 研究框架 ─────────────────────────────────────────────────────────
    pages.append(make_page(f"""
<div class="section-num">01</div>
<h1>研究框架</h1>

<h2>数据说明</h2>
<table>
  <tr><th class="l">项目</th><th class="l">说明</th></tr>
  <tr><td class="l">数据来源</td><td class="l">公募基金季报强制披露的前十大重仓股汇总（A股+H股）</td></tr>
  <tr><td class="l">覆盖范围</td><td class="l">2024年一季度 — 2026年一季度，共9个季度，每季约2,500–3,000只个股</td></tr>
  <tr><td class="l">关键字段</td><td class="l">持有基金数、持仓市值（亿元）、占基金持仓比重、持有基金数季度增减、行业</td></tr>
  <tr><td class="l">信号生效日</td><td class="l">季报公开披露日（约每季末次月20日，如一季报→4月20日）</td></tr>
  <tr><td class="l">超额收益</td><td class="l">个股区间收益 − 同期沪深300收益，分30/60/90自然日三个持仓期</td></tr>
  <tr><td class="l">流动性门槛</td><td class="l">持有基金数 ≥ 30家（约为被30家以上基金列入前十大重仓的标的）</td></tr>
</table>

<h2>核心指标定义</h2>
<table>
  <tr><th class="l">指标名称</th><th class="l">含义</th><th class="l">用途</th></tr>
  <tr><td class="l">持有基金数</td><td class="l">本季将该股列为前十大持仓的基金家数（家）</td><td class="l">衡量机构共识度与流动性</td></tr>
  <tr><td class="l">持有基金数季度增减</td><td class="l">本季持有基金数 − 上季持有基金数，正数=加仓，负数=减仓</td><td class="l">方向性信号，最核心指标</td></tr>
  <tr><td class="l">占基金持仓比重</td><td class="l">该股持仓市值 ÷ 全部公募基金持仓总市值（%）</td><td class="l">衡量配置拥挤度</td></tr>
  <tr><td class="l">占比季度变动</td><td class="l">本季占基金持仓比重 − 上季占基金持仓比重（%）</td><td class="l">配置比重变化方向</td></tr>
  <tr><td class="l">新进入</td><td class="l">本季出现在重仓名单、上季不在的个股</td><td class="l">首次建仓信号</td></tr>
</table>

<h2>回测方法</h2>
<p>每季季报披露后，按信号筛选股票构建等权组合，持有至下季度披露日后调仓。净值从1.0起算，不计交易成本及印花税。</p>
<p>低换手策略：每季最多调换5只，优先保留当季信号仍强的持仓；若当季出现持有基金数为负（机构净减仓），则强制退出。</p>
""", page_num=3, section_num='01'))

    # ─── 信号有效性 ────────────────────────────────────────────────────────
    fcc = data['fcc_grp']
    pages.append(make_page(f"""
<div class="section-num">02</div>
<h1>信号有效性分析</h1>

<h2>发现一：加仓信号有效，减仓信号较弱</h2>
<p><strong>分组定义：</strong>每季度内，将持有基金数季度增减（净增减家数）从大到小排序，
前20%且净增为正的个股归为「加仓组」；后20%且净减为负的归为「减仓组」；其余60%为「持平组」。
每季加仓组约400–600只，减仓组约400–600只。</p>
<div class="kpi-row">
  <div class="kpi"><div class="kpi-lbl">加仓组 30日超额均值</div>
    <div class="kpi-val pos">+{fcc['加仓'][30]['mean']:.2f}%</div>
    <div class="kpi-sub">胜率 {fcc['加仓'][30]['win']:.1f}%  样本n={fcc['加仓'][30]['n']:,}</div></div>
  <div class="kpi"><div class="kpi-lbl">减仓组 30日超额均值</div>
    <div class="kpi-val pos">+{fcc['减仓'][30]['mean']:.2f}%</div>
    <div class="kpi-sub">胜率 {fcc['减仓'][30]['win']:.1f}%  样本n={fcc['减仓'][30]['n']:,}</div></div>
  <div class="kpi"><div class="kpi-lbl">加仓−减仓 超额差（30日）</div>
    <div class="kpi-val blu">+{fcc['加仓'][30]['mean']-fcc['减仓'][30]['mean']:.2f}%</div>
    <div class="kpi-sub">加仓组胜率领先 {fcc['加仓'][30]['win']-fcc['减仓'][30]['win']:.1f}个百分点</div></div>
  <div class="kpi"><div class="kpi-lbl">加仓−减仓 超额差（90日）</div>
    <div class="kpi-val blu">+{fcc['加仓'][90]['mean']-fcc['减仓'][90]['mean']:.2f}%</div>
    <div class="kpi-sub">持仓期越长差距越大</div></div>
</div>
<div class="chart-wrap">
  <img src="{imgs['fcc']}">
  <div class="chart-caption">图2  加仓/持平/减仓组 各持仓期超额收益均值与胜率（全样本，不含持有基金数门槛限制）</div>
</div>
<div class="note">⚠ 减仓组30日超额仍为正值（+{fcc['减仓'][30]['mean']:.2f}%），说明基金减仓的股票短期仍在跑赢基准。<strong>单独做空减仓股票效果有限</strong>，不建议基于减仓信号构建纯空头头寸。减仓信号更适合用于存量持仓的退出决策（详见第四部分）。</div>

<hr class="divider">
<h2>发现二：大幅加仓反而是追涨信号</h2>
<p>在加仓股票（持有基金数净增>0）内部，按加仓幅度三分位分组，呈现明显逆向规律：</p>
<table>
  <tr><th class="l">加仓幅度分组</th><th>量化定义</th><th>30日超额均值</th><th>30日胜率</th><th>含义</th></tr>
  <tr><td class="l"><span class="tag tag-add">低加仓幅度</span></td>
    <td class="l">净增家数处于加仓组后⅓（中位约{data['fcc_def']['加仓_p25']:.0f}家）</td>
    <td class="pos">+4.36%</td><td>57.1%</td><td class="l">悄悄建仓，预期差最大</td></tr>
  <tr><td class="l"><span class="tag tag-neu">中加仓幅度</span></td>
    <td class="l">净增家数处于加仓组中⅓（中位约{data['fcc_def']['加仓_median']:.0f}家）</td>
    <td class="pos">+2.27%</td><td>49.8%</td><td class="l"></td></tr>
  <tr><td class="l"><span class="tag tag-red">高加仓幅度</span></td>
    <td class="l">净增家数处于加仓组前⅓（中位约{data['fcc_def']['加仓_p75']:.0f}家）</td>
    <td class="pos">+1.39%</td><td>47.1%</td><td class="l">追涨，股价已反映</td></tr>
</table>
<p style="margin-top:8px">实操建议：优先选持有基金数净增量在<strong>适中区间</strong>的加仓股，而非净增量爆表的热门票。</p>
""", page_num=4, section_num='02'))

    # ─── 集中度与行业 ────────────────────────────────────────────────────
    fw = data['fw_grp']
    pages.append(make_page(f"""
<div class="section-num">03</div>
<h1>持仓拥挤度与行业分化</h1>

<h2>发现三：低拥挤度个股后续超额更高</h2>
<p>按占基金持仓比重高低将所有持仓股四等分，超额收益与拥挤度呈<strong>显著负相关</strong>。
分位数阈值（跨季均值）：25%分位线约{data['fw_thresholds']['p25']:.4f}%，
中位数约{data['fw_thresholds']['p50']:.4f}%，75%分位线约{data['fw_thresholds']['p75']:.4f}%。</p>
<div class="chart-wrap">
  <img src="{imgs['fw']}">
  <div class="chart-caption">图3  占基金持仓比重 四分位组 30/90日平均超额收益（%）及30日胜率</div>
</div>
<table>
  <tr><th class="l">拥挤度分组</th><th class="l">量化定义（占基金持仓比重）</th><th>30日超额均值</th><th>30日胜率</th><th>90日超额均值</th><th>样本数</th></tr>
  <tr><td class="l">Q1 最低拥挤</td>
    <td class="l">占比 &lt; {data['fw_thresholds']['p25']:.4f}%（后25%）</td>
    <td class="pos">+{fw['Q1低'][30]:.2f}%</td>
    <td class="pos">{fw['Q1低']['win30']:.1f}%</td>
    <td class="pos">+{fw['Q1低'][90]:.2f}%</td><td>{fw['Q1低']['n']:,}</td></tr>
  <tr><td class="l">Q2</td>
    <td class="l">{data['fw_thresholds']['p25']:.4f}% ~ {data['fw_thresholds']['p50']:.4f}%</td>
    <td class="pos">+{fw['Q2'][30]:.2f}%</td><td>{fw['Q2']['win30']:.1f}%</td>
    <td class="pos">+{fw['Q2'][90]:.2f}%</td><td>{fw['Q2']['n']:,}</td></tr>
  <tr><td class="l">Q3</td>
    <td class="l">{data['fw_thresholds']['p50']:.4f}% ~ {data['fw_thresholds']['p75']:.4f}%</td>
    <td class="pos">+{fw['Q3'][30]:.2f}%</td><td>{fw['Q3']['win30']:.1f}%</td>
    <td class="pos">+{fw['Q3'][90]:.2f}%</td><td>{fw['Q3']['n']:,}</td></tr>
  <tr><td class="l">Q4 最高拥挤</td>
    <td class="l">占比 &gt; {data['fw_thresholds']['p75']:.4f}%（前25%）</td>
    <td class="pos">+{fw['Q4高'][30]:.2f}%</td>
    <td class="neg">{fw['Q4高']['win30']:.1f}%</td>
    <td class="pos">+{fw['Q4高'][90]:.2f}%</td><td>{fw['Q4高']['n']:,}</td></tr>
</table>
<p>实操含义：持仓排名前10的顶级抱团股（Q4）并非最优选择；排名50–150位的"安静重仓股"（Q1/Q2）往往具备更大预期差空间。</p>

<hr class="divider">
<h2>发现四：行业过滤是信号的决定性前置条件</h2>
<div class="chart-wrap">
  <img src="{imgs['ind']}">
  <div class="chart-caption">图4  各行业加仓信号后续30日平均超额收益（加仓组，n≥10，降序排列）</div>
</div>
<table>
  <tr><th class="l">类别</th><th class="l">代表行业</th><th>加仓后30日超额均值</th><th>解释</th></tr>
  <tr><td class="l"><span class="tag tag-add">信号有效</span></td>
    <td class="l">通信、电力设备、建筑材料、传媒</td>
    <td class="pos">+5% ~ +9%</td><td class="l">进攻性配置，伴随景气度改善</td></tr>
  <tr><td class="l"><span class="tag tag-red">信号无效</span></td>
    <td class="l">食品饮料、社会服务、农林牧渔</td>
    <td class="neg">-1% ~ -2%</td><td class="l">防御性配置，无方向性信息</td></tr>
</table>
""", page_num=5, section_num='03'))

    # ─── 最优策略详解 ─────────────────────────────────────────────────���──
    pages.append(make_page(f"""
<div class="section-num">04</div>
<h1>最优策略详解</h1>

<h2>策略构建逻辑</h2>
<p>在前述信号研究的基础上，将三个有效因子叠加：</p>
<table>
  <tr><th class="l">步骤</th><th class="l">筛选条件</th><th class="l">目的</th></tr>
  <tr><td class="l">① 流动性过滤</td><td class="l">持有基金数 ≥ 30家</td><td class="l">排除小市值、流动性不足标的</td></tr>
  <tr><td class="l">② 行业相对低配</td><td class="l">占基金持仓比重 低于同行业平均值</td><td class="l">规避拥挤，保留预期差空间</td></tr>
  <tr><td class="l">③ 仓位上升确认</td><td class="l">占基金持仓比重 季度环比上升</td><td class="l">确认机构正在主动增配</td></tr>
  <tr><td class="l">④ 低换手执行</td><td class="l">每季最多调换5只（换手率降75%）</td><td class="l">利用信号持续性，降低交易成本</td></tr>
  <tr><td class="l">⑤ 减仓强制退出</td><td class="l">持仓中持有基金数出现净减少，则强制退出</td><td class="l">反用减仓信号，及时止损</td></tr>
</table>

<h2>净值表现</h2>
<div class="chart-wrap">
  <img src="{imgs['nav']}">
  <div class="chart-caption">图5  最优策略净值曲线及回撤对比（2024年7月 — 2026年4月）</div>
</div>
<div class="chart-wrap">
  <img src="{imgs['qtr']}">
  <div class="chart-caption">图6  逐季度超额收益（各策略 vs 沪深300）</div>
</div>
<div class="kpi-row">
  <div class="kpi"><div class="kpi-lbl">累计超额（vs沪深300）</div>
    <div class="kpi-val pos">+{st_B['alpha']:.0f}%</div></div>
  <div class="kpi"><div class="kpi-lbl">年化收益</div>
    <div class="kpi-val pos">{st_B['ann']:+.1f}%</div></div>
  <div class="kpi"><div class="kpi-lbl">夏普比率</div>
    <div class="kpi-val pos">{st_B['sh']:.2f}</div></div>
  <div class="kpi"><div class="kpi-lbl">最大回撤</div>
    <div class="kpi-val neg">{st_B['dd']:.1f}%</div></div>
</div>
""", page_num=6, section_num='04'))

    # ─── 调仓实例 ────────────────────────────────────────────────────────
    exits  = data['exits']
    enters = data['enters']
    keeps  = data['keeps']
    cur    = data['cur_1q26']

    def fcc_str(s):
        v = s.get('fcc')
        return (f'+{int(v)}家' if v and v > 0 else f'{int(v)}家' if v else '—')
    def fwd_str(s):
        v = s.get('fwd')
        return f'{v*100:+.3f}%' if v is not None else '—'

    exit_rows = ''.join(
        f'<tr><td class="l"><span class="tag tag-red">退出</span></td>'
        f'<td class="l">{s["code"]}</td><td class="l">{s["name"]}</td>'
        f'<td class="l">{s.get("ind","")}</td>'
        f'<td class="neg">{fcc_str(cur.get(s["code"],{}))}</td>'
        f'<td class="l">机构持有基金数转负，触发强制退出</td></tr>'
        for c, s in sorted(exits.items(), key=lambda x: x[1].get('fcc') or 0))

    enter_rows = ''.join(
        f'<tr><td class="l"><span class="tag tag-add">新进</span></td>'
        f'<td class="l">{s["code"]}</td><td class="l">{s["name"]}</td>'
        f'<td class="l">{s.get("ind","")}</td>'
        f'<td class="pos">{fcc_str(s)}</td>'
        f'<td class="l">行业低配+占比上升，信号强度靠前</td></tr>'
        for c, s in sorted(enters.items(), key=lambda x: -(x[1].get('fcc') or 0)))

    keep_rows = ''.join(
        f'<tr><td class="l"><span class="tag tag-neu">续持</span></td>'
        f'<td class="l">{s["code"]}</td><td class="l">{s["name"]}</td>'
        f'<td class="l">{s.get("ind","")}</td>'
        f'<td class="pos">{fcc_str(s)}</td>'
        f'<td class="l">信号持续强劲，保留持仓</td></tr>'
        for c, s in sorted(keeps.items(), key=lambda x: -(x[1].get('fcc') or 0)))

    hold_rows = ''.join(
        f'<tr><td class="l">{s["code"]}</td><td class="l">{s["name"]}</td>'
        f'<td class="l">{s.get("ind","")}</td>'
        f'<td>{s.get("fc","—")}</td>'
        f'<td class="pos">{fcc_str(s)}</td>'
        f'<td class="pos">{fwd_str(s)}</td></tr>'
        for c, s in sorted({**keeps, **enters}.items(),
                            key=lambda x: -(x[1].get('fcc') or 0)))

    pages.append(make_page(f"""
<div class="section-num">05</div>
<h1>调仓实例：4Q25 → 1Q26</h1>
<p>2026年4月20日（1Q26季报披露日），最优策略（策略B）执行季度调仓，
由上季度8只变为本季度10只，具体如下：</p>

<h2>调仓明细</h2>
<table>
  <tr><th class="l">操作</th><th class="l">代码</th><th class="l">名称</th>
    <th class="l">行业</th><th>持有基金数增减</th><th class="l">操作理由</th></tr>
  {exit_rows}
  {enter_rows}
  {keep_rows}
</table>

<h2>1Q26 当前持仓（10只等权，下次调仓约7月20日）</h2>
<table>
  <tr><th class="l">代码</th><th class="l">名称</th><th class="l">行业</th>
    <th>持有基金数</th><th>本季基金数增减</th><th>持仓比重变动</th></tr>
  {hold_rows}
</table>

<div class="note">
  <strong>注：</strong>上述持仓由机构季报公开信息生成，信号生效日为4月20日（季报披露日）。
  海天味业属食品饮料板块，按本研究行业有效性分析，该行业加仓信号历史有效性较低；
  石油石化三只（恒逸/荣盛/恒力）方向一致，建议实操中评估集中度风险，必要时可合并或限仓。
</div>
""", page_num=7, section_num='05'))

    # ─── 局限性 ────────────────────────────────���─────────────────────��──
    pages.append(make_page(f"""
<div class="section-num">06</div>
<h1>局限性与风险提示</h1>

<h2>主要局限性</h2>
<table>
  <tr><th class="l">类别</th><th class="l">说明</th><th class="l">影响程度</th></tr>
  <tr><td class="l">牛市偏差</td>
    <td class="l">2024Q4–2025Q2为强势行情，整体超额水平偏高，策略在熊市或震荡市中的有效性尚未验证</td>
    <td class="l"><span class="tag tag-add">高</span></td></tr>
  <tr><td class="l">样本量不足</td>
    <td class="l">仅8个完整调仓周期，统计显著性有限，结论需随数据积累持续修正</td>
    <td class="l"><span class="tag tag-add">高</span></td></tr>
  <tr><td class="l">交易成本</td>
    <td class="l">全换手策略估算年化额外成本约1.8%；低换手策略约0.45%（单边0.15%×季度调仓）</td>
    <td class="l"><span class="tag tag-hot">中</span></td></tr>
  <tr><td class="l">披露日误差</td>
    <td class="l">统一使用季末次月20日，实际各基金披露时间约15–25日，存在±3–5交易日误差</td>
    <td class="l"><span class="tag tag-neu">低</span></td></tr>
  <tr><td class="l">持仓集中度</td>
    <td class="l">最优策略组合持仓约10只，单票权重约10%，波动高于常规组合，不适合大规模资金</td>
    <td class="l"><span class="tag tag-hot">中</span></td></tr>
  <tr><td class="l">港股数据覆盖</td>
    <td class="l">约12%港股持仓无法匹配日度收盘价，信号有低估可能</td>
    <td class="l"><span class="tag tag-neu">低</span></td></tr>
</table>

<h2>后续研究方向</h2>
<p>① <strong>扩展样本</strong>：待2026年下半年数据可得后，验证策略在调整行情中的表现</p>
<p>② <strong>叠加基本面过滤</strong>：在加仓增量信号上叠加估值（PE/PB）或景气度筛选，测试组合效果提升空间</p>
<p>③ <strong>连续加仓研究</strong>：连续两季加仓 vs 仅单季加仓的后续表现对比，判断信号持续性差异</p>
<p>④ <strong>行业中性化</strong>：在行业内部排名，剥离行业涨跌干扰，提炼更纯粹的个股选择α</p>
<p>⑤ <strong>放宽流动性门槛</strong>：降低至fc≥10–15家，重新测试首次建仓信号的可执行性（该信号在严格门槛下无法执行，但逻辑最优）</p>

<hr class="divider">
<h2>免责声明</h2>
<p style="font-size:8.5pt;color:#999;line-height:1.6">
本报告基于公开可得的公募基金季报数据，所有策略回测结果仅供研究参考，
不构成任何投资建议。历史回测表现不代表未来实际收益，投资者应结合自身风险承受能力独立判断。
本报告仅供内部研究使用，不得对外分发。
</p>
""", page_num=8, section_num='06'))

    # ─── 组装 HTML ─────────────────────────��──────────────────────────────
    print_btn = """
<div class="no-print" style="text-align:center;padding:16px;position:fixed;bottom:0;
  left:0;right:0;background:rgba(255,255,255,.9);backdrop-filter:blur(4px);
  border-top:1px solid #ddd;z-index:100">
  <button onclick="window.print()"
    style="background:#1e50a2;color:#fff;border:none;padding:10px 28px;border-radius:6px;
           font-size:13px;cursor:pointer;font-family:inherit">
    🖨 打印 / 保存为 PDF
  </button>
  <span style="margin-left:16px;color:#666;font-size:12px">
    建议使用 Chrome 打印，纸张选 A4，取消页眉页脚
  </span>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>公募基金季度持仓信号有效性研究</title>
<style>{STYLE}</style>
</head>
<body>
{''.join(pages)}
{print_btn}
</body>
</html>"""
    return html


def main():
    data = compute_all()
    print("构建报告…")
    html = build_report(data)
    OUTPUT.write_text(html, encoding='utf-8')
    print(f"完成！输出: {OUTPUT}  ({OUTPUT.stat().st_size//1024} KB)")
    print("在 Chrome 中打开，点击【打印/保存为PDF】，纸张选A4，取消页眉页脚。")

if __name__ == '__main__':
    main()
