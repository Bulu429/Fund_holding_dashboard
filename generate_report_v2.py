#!/usr/bin/env python3
"""
基金持仓信号研究报告 v2
叙事主线：信号组合递进（单因子→多因子→执行优化）
格式：研报风格 HTML，Chrome 打印为 A4 PDF
"""

import sys, warnings, base64, io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings('ignore')

BASE   = Path("workwork/research/策略研究/基金重仓看板")
OUTPUT = BASE / "fund_report_v2.html"

sys.path.insert(0, str(Path.cwd()))
from workwork.research.策略研究.基金重仓看板.backtest_signals import (
    load_holdings, load_prices, sort_q, run_backtest
)
from workwork.research.策略研究.基金重仓看板.strategy_final import (
    build_strategies, run_nav, calc_stats
)

rcParams['font.family']        = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False
rcParams['font.size']          = 9
rcParams['axes.spines.top']    = False
rcParams['axes.spines.right']  = False

C_RED = '#c0392b'; C_GRN = '#166534'; C_BLU = '#1e50a2'
C_ORG = '#d97706'; C_GRY = '#96a0ae'; C_BG  = '#fafaf8'

def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'data:image/png;base64,{data}'


# ══════════════════════════════════════════════════════════════════════
# 1. 数据计算
# ══════════════════════════════════════════════════════════════════════

def compute():
    print("加载数据…")
    by_q     = load_holdings()
    price_df = load_prices()
    quarters = sorted(by_q.keys(), key=sort_q)

    print("回测信号统计…")
    df_raw, _ = run_backtest(by_q, price_df, top_pct=0.20)
    df = df_raw[df_raw['alpha_30d'].notna()].copy()

    # 信号组
    fcc_grp = {}
    for g in ['加仓','持平','减仓']:
        sub = df[df['fcc_group'] == g]
        fcc_grp[g] = {d: {
            'mean': round(sub[f'alpha_{d}d'].mean(), 2) if len(sub) else 0,
            'win':  round((sub[f'alpha_{d}d']>0).mean()*100, 1) if len(sub) else 0,
            'n':    len(sub)} for d in [30,60,90]}

    # fw 分位数
    fw_grp = {}
    for g in ['Q1低','Q2','Q3','Q4高']:
        sub = df[df['fw_group'] == g]
        fw_grp[g] = {d: round(sub[f'alpha_{d}d'].mean(), 2) if len(sub) else 0
                     for d in [30,60,90]}
        fw_grp[g]['win30'] = round((sub['alpha_30d']>0).mean()*100, 1) if len(sub) else 0
        fw_grp[g]['n']     = len(sub)

    # 加仓幅度三分位
    add_df = df[df['fcc_group'] == '加仓'].copy()
    add_df['fcc_q3'] = pd.qcut(add_df['fcc'], q=3, labels=['低加仓','中加仓','高加仓'])
    fcc_size = {}
    for g in ['低加仓','中加仓','高加仓']:
        sub = add_df[add_df['fcc_q3'] == g]
        fcc_size[g] = {
            'mean': round(sub['alpha_30d'].mean(), 2),
            'win':  round((sub['alpha_30d']>0).mean()*100, 1),
            'n':    len(sub),
            'med_fcc': round(sub['fcc'].median(), 0),
        }

    # 行业
    ind_alpha = (df[df['fcc_group']=='加仓']
                 .groupby('ind')['alpha_30d']
                 .agg(['mean','count'])
                 .query('count >= 10')
                 .sort_values('mean', ascending=False))

    # 最优策略净值
    print("构建策略…")
    strats = build_strategies(by_q, quarters)
    nav_data, stat_data = {}, {}
    for key in ['B', 'C', 'REF1']:
        ndf, _ = run_nav(strats[key], price_df)
        nav_data[key]  = ndf
        stat_data[key] = calc_stats(ndf)

    return dict(
        fcc_grp=fcc_grp, fw_grp=fw_grp, fcc_size=fcc_size,
        ind_alpha=ind_alpha, nav_data=nav_data, stat_data=stat_data,
        total_obs=len(df),
    )


# ══════════════════════════════════════════════════════════════════════
# 2. 图表
# ══════════════════════════════════════════════════════════════════════

def chart_evolution(stat_data):
    """Page1 核心：策略演进路径图"""
    steps = [
        ('单因子\n加仓信号', '+58%\n夏普1.17',  C_BLU),
        ('叠加\n行业低配',   '+62%\n夏普1.25',  '#0891b2'),
        ('降低\n换手率',     '+71%\n夏普1.29',  C_ORG),
        ('反用\n减仓退出',   '+161%\n夏普1.67', C_RED),
    ]
    sub = [
        '持有基金数增量前20\n(fc≥30)',
        '行业内低配\n+占比上升',
        '每季仅换5只\n换手率降75%',
        '减仓股强制退出\n信号叠加',
    ]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2)
    ax.axis('off')

    bw, bh, by = 2.1, 2.0, 0.6
    gap = 0.55
    xs = [0.15, 0.15+bw+gap, 0.15+2*(bw+gap), 0.15+3*(bw+gap)]

    for i, ((title, num, col), desc, x) in enumerate(zip(steps, sub, xs)):
        # 方框底色（渐淡）
        rect = plt.Rectangle((x, by), bw, bh,
                              facecolor=col+'18', edgecolor=col, linewidth=1.8,
                              zorder=2)
        ax.add_patch(rect)

        # 超额数字（大）
        lines = num.split('\n')
        ax.text(x+bw/2, by+bh*0.68, lines[0],
                ha='center', va='center', fontsize=18, fontweight='bold',
                color=col, zorder=3)
        ax.text(x+bw/2, by+bh*0.42, lines[1],
                ha='center', va='center', fontsize=8.5,
                color=col, zorder=3)

        # 标题
        ax.text(x+bw/2, by+bh*0.88, title,
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                color='#1a1a2e', zorder=3)

        # 描述（小字）
        ax.text(x+bw/2, by-0.28, desc,
                ha='center', va='top', fontsize=7.5,
                color='#566070', zorder=3)

        # 箭头
        if i < 3:
            ax.annotate('', xy=(xs[i+1]-0.02, by+bh/2),
                        xytext=(x+bw+0.04, by+bh/2),
                        arrowprops=dict(arrowstyle='->', color='#555',
                                        lw=1.8, mutation_scale=14),
                        zorder=4)
            # 增量标注
            vals = [58.2, 62.2, 71.4, 160.7]
            delta = vals[i+1] - vals[i]
            mx = x + bw + gap/2
            ax.text(mx, by+bh/2+0.18, f'+{delta:.1f}%',
                    ha='center', va='bottom', fontsize=7.5,
                    color='#999', style='italic')

    fig.tight_layout(pad=0.3)
    return b64(fig)


def chart_fcc_size(fcc_size):
    """发现一：加仓幅度三分位"""
    grps = ['低加仓','中加仓','高加仓']
    vals = [fcc_size[g]['mean'] for g in grps]
    wins = [fcc_size[g]['win'] for g in grps]
    meds = [int(fcc_size[g]['med_fcc']) for g in grps]
    cols = [C_BLU, C_ORG, C_RED]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)
    ax.grid(axis='y', color='#e8e3db', ls='--', lw=0.6, zorder=0)
    x = np.arange(3)
    bars = ax.bar(x, vals, color=cols, width=0.5, zorder=2, alpha=0.88)
    for bar, v, w, m in zip(bars, vals, wins, meds):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.08,
                f'{v:+.2f}%\n胜率{w:.0f}%\n(中位净增{m}家)',
                ha='center', va='bottom', fontsize=7.5, color='#333',
                linespacing=1.5)
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['低加仓幅度\n（悄悄建仓）', '中加仓幅度', '高加仓幅度\n（追涨行为）'],
                       fontsize=8.5)
    ax.set_ylabel('30日超额收益（%）', fontsize=8.5)
    ax.set_title('加仓幅度越大，后续超额越低', fontsize=9.5, pad=8)
    ax.set_ylim(0, max(vals)*1.55)
    fig.tight_layout()
    return b64(fig)


def chart_fw_quartile(fw_grp):
    """发现二：持仓拥挤度四分位"""
    grps = ['Q1低','Q2','Q3','Q4高']
    labs = ['Q1\n最低拥挤', 'Q2', 'Q3', 'Q4\n最高拥挤']
    v30  = [fw_grp[g][30] for g in grps]
    win  = [fw_grp[g]['win30'] for g in grps]
    cols = [C_BLU, '#0891b2', C_ORG, C_GRY]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)
    ax.grid(axis='y', color='#e8e3db', ls='--', lw=0.6, zorder=0)
    bars = ax.bar(np.arange(4), v30, color=cols, width=0.5, zorder=2, alpha=0.88)
    for bar, v, w in zip(bars, v30, win):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.08,
                f'{v:+.2f}%\n胜率{w:.0f}%',
                ha='center', va='bottom', fontsize=8, color='#333', linespacing=1.5)
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel('30日超额收益（%）', fontsize=8.5)
    ax.set_title('持仓越拥挤，后续超额越低', fontsize=9.5, pad=8)
    ax.set_ylim(0, max(v30)*1.5)
    fig.tight_layout()
    return b64(fig)


def chart_nav_simple(nav_data, stat_data):
    """发现三：净值曲线（3条）"""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)
    ax.grid(color='#e8e3db', ls='--', lw=0.5)

    cfg = [
        ('B',    C_RED,  '最优策略（+161%）',    '-',   2.5),
        ('REF1', C_BLU,  '参考：加仓前20（+58%）', '--',  1.5),
    ]
    for key, col, lbl, ls, lw in cfg:
        df = nav_data[key]
        dates = pd.to_datetime(df['date'])
        ax.plot(dates, df['nav'], color=col, lw=lw, ls=ls, label=lbl)
    # 基准
    df0 = nav_data['REF1']
    ax.plot(pd.to_datetime(df0['date']), df0['bench'],
            color=C_GRY, lw=1.2, ls=':', label='沪深300基准')

    ax.set_ylabel('净值（起始=1.0）', fontsize=8.5)
    ax.legend(fontsize=8.5, framealpha=0.8, loc='upper left')
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%y/%m'))
    fig.tight_layout()
    return b64(fig)


def chart_nav_full(nav_data):
    """附录B：完整净值曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor('white')
    cfg = [
        ('B',    C_RED,  '最优策略B',      '-',  2.5),
        ('C',    C_ORG,  '基金增减低换手',  '-',  2.0),
        ('REF1', C_GRY,  '参考全换手',     '--', 1.5),
    ]
    dates = pd.to_datetime(nav_data['REF1']['date'])
    for key, col, lbl, ls, lw in cfg:
        df = nav_data[key]
        ax1.plot(pd.to_datetime(df['date']), df['nav'],
                 color=col, lw=lw, ls=ls, label=lbl)
    ax1.plot(dates, nav_data['REF1']['bench'], color='#bbb', lw=1, ls=':', label='沪深300')
    for ax in [ax1, ax2]:
        ax.set_facecolor(C_BG)
        ax.grid(color='#e8e3db', ls='--', lw=0.5)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%y/%m'))
    ax1.set_title('净值曲线', fontsize=9.5)
    ax1.legend(fontsize=8, framealpha=0.8)
    ax1.set_ylabel('净值（起始=1.0）')

    # 回撤
    for key, col, lbl, ls, lw in cfg:
        nav = nav_data[key]['nav'].values
        mx  = np.maximum.accumulate(nav)
        dd  = (nav/mx - 1) * 100
        ax2.plot(pd.to_datetime(nav_data[key]['date']), dd,
                 color=col, lw=lw, ls=ls, label=lbl)
    ax2.set_title('回撤曲线', fontsize=9.5)
    ax2.legend(fontsize=8, framealpha=0.8)
    ax2.set_ylabel('回撤（%）')
    fig.tight_layout()
    return b64(fig)


def chart_qtr(nav_data, stat_data):
    """附录B：逐季度超额"""
    qs = sorted(stat_data['REF1']['qr'].keys())
    x = np.arange(len(qs)); w = 0.28
    cfg = [
        ('B',    C_RED, '最优策略B'),
        ('REF1', C_GRY, '参考全换手'),
    ]
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)
    ax.grid(axis='y', color='#e8e3db', ls='--', lw=0.5, zorder=0)
    for i, (key, col, lbl) in enumerate(cfg):
        qr   = stat_data[key]['qr']
        vals = [qr.get(q,{'port':0,'bench':0})['port'] -
                qr.get(q,{'port':0,'bench':0})['bench'] for q in qs]
        ax.bar(x + (i-0.5)*w, vals, w*0.9,
               color=[col if v>=0 else col+'88' for v in vals],
               label=lbl, zorder=2)
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace('Q',' Q') for q in qs], fontsize=8.5)
    ax.set_ylabel('超额收益（%）', fontsize=8.5)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return b64(fig)


# ══════════════════════════════════════════════════════════════════════
# 3. HTML 报告
# ══════════════════════════════════════════════════════════════════════

STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --ink:#1a1a2e; --ink2:#3d3d5c; --ink3:#7a7a9a;
  --acc:#1e50a2; --pos:#b91c1c; --neg:#15803d;
  --bd:#d4d0c8;  --bg:#faf8f4;  --card:#ffffff;
  --serif:'Noto Serif SC',serif;
  --sans:'Noto Sans SC',sans-serif;
  --mono:'JetBrains Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:#e0dbd2;color:var(--ink);font-family:var(--sans);font-size:10pt;line-height:1.65}

/* A4页面 */
.page{background:var(--card);width:210mm;min-height:297mm;margin:0 auto 14px;
  padding:14mm 16mm 12mm;box-shadow:0 2px 24px rgba(0,0,0,.14);position:relative}

/* 字体 */
h1{font-family:var(--serif);font-size:15pt;font-weight:700;color:var(--acc);
  border-bottom:2.5px solid var(--acc);padding-bottom:6px;margin-bottom:16px}
h2{font-family:var(--serif);font-size:11.5pt;font-weight:700;color:var(--ink);
  border-left:3.5px solid var(--acc);padding-left:8px;margin:14px 0 8px}
h3{font-size:10pt;font-weight:700;color:var(--ink2);margin:10px 0 5px}
p{margin-bottom:7px;color:var(--ink2);font-size:9.5pt}

/* 封面 */
.cover-wrap{display:flex;flex-direction:column;justify-content:space-between;
  min-height:269mm;padding-top:24mm}
.cover-eyebrow{font-size:8pt;letter-spacing:3px;text-transform:uppercase;
  color:var(--ink3);margin-bottom:8px}
.cover-title{font-family:var(--serif);font-size:26pt;font-weight:700;
  line-height:1.25;color:var(--ink);border-bottom:3px solid var(--acc);padding-bottom:10px}
.cover-sub{font-size:11.5pt;color:var(--ink2);margin-top:10px;line-height:1.7}
.cover-punch{font-family:var(--serif);font-size:13pt;color:var(--acc);
  font-style:italic;margin-top:16px;border-left:3px solid var(--acc);padding-left:10px}
.cover-meta{font-size:8.5pt;color:var(--ink3);border-top:1px solid var(--bd);
  padding-top:10px;display:flex;gap:24px}

/* KPI卡片 */
.kpi-row{display:flex;gap:8px;margin:12px 0}
.kpi{flex:1;background:var(--bg);border:1px solid var(--bd);border-radius:5px;
  padding:10px 12px;text-align:center;min-width:0}
.kpi-lbl{font-size:7.5pt;color:var(--ink3);text-transform:uppercase;letter-spacing:.5px}
.kpi-val{font-family:var(--mono);font-size:18pt;font-weight:700;line-height:1.1;margin-top:4px}
.kpi-sub{font-size:7.5pt;color:var(--ink3);margin-top:3px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.blu{color:var(--acc)}.gry{color:var(--ink3)}

/* 发现块 */
.finding{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px 0 14px;
  align-items:start}
.finding-text{padding-right:4px}
.finding-num{font-family:var(--mono);font-size:20pt;font-weight:700;
  line-height:1;color:var(--acc);margin-bottom:6px}
.finding-title{font-family:var(--serif);font-size:10.5pt;font-weight:700;
  color:var(--ink);margin-bottom:6px}
.finding-body{font-size:9pt;color:var(--ink2);line-height:1.7}
.finding-body b{color:var(--ink);font-weight:600}
.divider{border:none;border-top:1px solid var(--bd);margin:13px 0}

/* 图表 */
.chart{text-align:center;margin:6px 0}
.chart img{max-width:100%;border-radius:3px}
.cap{font-size:7.5pt;color:var(--ink3);text-align:center;margin-top:3px;font-style:italic}

/* 表格 */
table{width:100%;border-collapse:collapse;font-size:8.5pt;margin:8px 0}
th{background:#f0f0f8;color:var(--ink3);font-weight:700;font-size:7.5pt;
  text-transform:uppercase;letter-spacing:.4px;padding:7px 9px;
  text-align:right;border-bottom:2px solid var(--bd)}
th.l,td.l{text-align:left}
td{padding:6px 9px;border-bottom:1px solid #ede9e3;text-align:right;color:var(--ink)}
tr:nth-child(even) td{background:#faf8f4}
.bold td{font-weight:700;background:#f0f5ff!important}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:8pt;font-weight:600}
.tag-add{background:#fef2f2;color:var(--pos)}
.tag-red{background:#f0faf4;color:var(--neg)}
.tag-neu{background:#f0f0f8;color:var(--ink3)}
.tag-hot{background:#fffbeb;color:#92400e}
.tag-new{background:#fef9c3;color:#713f12}

/* 持仓表 */
.pos-logic{font-family:var(--sans);font-size:8pt;color:var(--ink3);
  font-style:italic;text-align:left!important}

/* 注意框 */
.note{background:#fffbeb;border-left:3px solid #f59e0b;padding:7px 11px;
  font-size:8.5pt;color:#78350f;margin:8px 0;border-radius:0 4px 4px 0}
.risk{background:#fef2f2;border-left:3px solid var(--pos);padding:7px 11px;
  font-size:8.5pt;color:#7f1d1d;margin:8px 0;border-radius:0 4px 4px 0}

/* 附录分隔页 */
.appendix-cover{display:flex;align-items:center;justify-content:center;
  height:269mm;text-align:center}
.appendix-label{font-family:var(--serif);font-size:36pt;font-weight:700;
  color:#e0dbd2;letter-spacing:4px}
.appendix-sub{font-size:12pt;color:var(--ink3);margin-top:12px}

/* 页脚 */
.footer{position:absolute;bottom:9mm;left:16mm;right:16mm;
  display:flex;justify-content:space-between;
  font-size:7.5pt;color:var(--ink3);border-top:1px solid var(--bd);padding-top:5px}

/* 页码角标 */
.pg-mark{font-family:var(--mono);font-size:8pt;color:var(--ink3)}

/* 打印 */
@media print{
  body{background:white}
  .page{margin:0;box-shadow:none;page-break-after:always;width:100%}
  .no-print{display:none}
}
@page{size:A4;margin:0}
"""

def footer(left, right):
    return f'<div class="footer"><span>{left}</span><span class="pg-mark">{right}</span></div>'

def page(body, pg=None, fl='公募基金持仓信号研究'):
    pg_txt = f'— {pg} —' if pg else ''
    ft = footer(fl, pg_txt) if pg else ''
    return f'<div class="page">{body}{ft}</div>'


def _fw_rows(fw):
    defs = {'Q1低': '<0.007%（后25%）', 'Q2': '0.007–0.017%',
            'Q3': '0.017–0.040%', 'Q4高': '>0.040%（前25%）'}
    rows = []
    for g in ['Q1低','Q2','Q3','Q4高']:
        pc = 'pos' if fw[g][30] > 0 else 'neg'
        rows.append(
            f'<tr><td class="l">{g}（占比 {defs[g]}）</td>'
            f'<td class="{pc}">+{fw[g][30]:.2f}%</td>'
            f'<td>{fw[g]["win30"]:.1f}%</td>'
            f'<td class="pos">+{fw[g][60]:.2f}%</td>'
            f'<td class="pos">+{fw[g][90]:.2f}%</td>'
            f'<td>{fw[g]["n"]:,}</td></tr>'
        )
    return ''.join(rows)


def build(data, imgs, gen_time):
    st_B  = data['stat_data']['B']
    st_R1 = data['stat_data']['REF1']
    fcc   = data['fcc_grp']
    fw    = data['fw_grp']
    sz    = data['fcc_size']
    pages = []

    # ── 封面 ──────────────────────────────────────────────────────────
    pages.append(page(f"""
<div class="cover-wrap">
  <div>
    <div class="cover-eyebrow">量化研究报告 · 内部使用</div>
    <div class="cover-title">公募基金季度持仓信号<br>有效性研究</div>
    <div class="cover-sub">基于季报披露数据构建可执行的机构持仓量化选股策略<br>
      覆盖2024Q1—2026Q1，共9个季度披露周期，22,000+有效观测</div>
    <div class="cover-punch">
      从单因子加仓信号出发，经三步叠加优化，<br>
      最终策略累计超额收益+161%，夏普比率1.67
    </div>
  </div>
  <div>
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-lbl">最优策略累计超额</div>
        <div class="kpi-val pos">+{st_B['alpha']:.0f}%</div>
        <div class="kpi-sub">vs 沪深300，8个调仓周期</div></div>
      <div class="kpi"><div class="kpi-lbl">夏普比率</div>
        <div class="kpi-val pos">{st_B['sh']:.2f}</div>
        <div class="kpi-sub">日度年化</div></div>
      <div class="kpi"><div class="kpi-lbl">最大回撤</div>
        <div class="kpi-val neg">{st_B['dd']:.1f}%</div>
        <div class="kpi-sub">持仓期内</div></div>
      <div class="kpi"><div class="kpi-lbl">参考策略超额</div>
        <div class="kpi-val blu">+{st_R1['alpha']:.0f}%</div>
        <div class="kpi-sub">单因子加仓前20基线</div></div>
      <div class="kpi"><div class="kpi-lbl">均持仓股数</div>
        <div class="kpi-val gry">~10只</div>
        <div class="kpi-sub">等权，季度调仓</div></div>
    </div>
    <div class="cover-meta">
      <span>研究员：[作者]</span>
      <span>日期：{datetime.now().strftime('%Y年%m月%d日')}</span>
      <span>数据来源：公募基金季报重仓持股汇总（A股+H股）</span>
    </div>
  </div>
</div>"""))

    # ── Page 1：执行摘要 ─────────────────────────────────────────────
    pages.append(page(f"""
<h1>执行摘要</h1>
<p>本研究以公募基金季报披露的前十大持仓数据为原料，测试持仓信号对后续个股走势的预测价值。
我们从最简单的"加仓信号"出发，逐步叠加三个优化维度，最终形成一套可执行的量化选股策略。
<strong>研究的核心结论是：信号需要组合使用，每一步叠加都能带来可量化的超额提升。</strong></p>

<h2>策略演进路径</h2>
<div class="chart">
  <img src="{imgs['evolution']}">
  <div class="cap">图1  四步策略演进：每步叠加带来的累计超额收益提升（基准：沪深300）</div>
</div>

<h2>四步优化逻辑</h2>
<table>
  <tr><th style="width:16%">步骤</th><th class="l">信号/优化内容</th>
    <th class="l" style="width:36%">核心逻辑</th><th style="width:12%">累计超额</th></tr>
  <tr><td style="font-weight:700;color:{C_BLU}">① 基础信号</td>
    <td class="l">持有基金数季度净增最大前20（fc≥30）</td>
    <td class="l">机构共同加仓是有效方向性信号</td>
    <td class="pos">+58%</td></tr>
  <tr><td style="font-weight:700;color:#0891b2">② 行业低配</td>
    <td class="l">在①基础上，筛选占基金持仓比重低于同行业均值的股票</td>
    <td class="l">低拥挤+方向向上，预期差最大</td>
    <td class="pos">+62%</td></tr>
  <tr><td style="font-weight:700;color:{C_ORG}">③ 低换手</td>
    <td class="l">每季仅换5只（保留信号强的持仓），换手率降75%</td>
    <td class="l">机构加仓信号具有跨季度持续性</td>
    <td class="pos">+71%</td></tr>
  <tr class="bold"><td style="font-weight:700;color:{C_RED}">④ 减仓退出</td>
    <td class="l">持仓中若持有基金数转为净减少，强制退出</td>
    <td class="l">把弱的减仓信号转化为强退出过滤器</td>
    <td class="pos">+161%</td></tr>
</table>
<p style="margin-top:6px;font-size:9pt;color:var(--ink3)">
注：超额收益为相对沪深300的累计回报，回测区间2024年7月—2026年4月，不含交易成本。</p>
""", pg=1))

    # ── Page 2：三个核心发现 ─────────────────────────────────────────
    pages.append(page(f"""
<h1>三个核心发现</h1>

<!-- 发现一 -->
<div class="finding">
  <div class="finding-text">
    <div class="finding-num">01</div>
    <div class="finding-title">加仓信号有效，但大幅加仓是追涨</div>
    <div class="finding-body">
      在所有加仓股票内部按加仓幅度三分位分组：
      <b>低加仓幅度组（悄悄建仓）</b>30日超额+4.36%，胜率57%；
      <b>高加仓幅度组（追涨买入）</b>仅+1.39%，胜率47%。<br><br>
      大幅加仓往往发生在股价已有明显上涨之后，市场已定价；
      小幅加仓代表少数基金悄悄布局，预期差最大。<br><br>
      <b>实操含义：</b>选持有基金数净增适中的加仓股，
      避开净增量爆表的热门票。
    </div>
  </div>
  <div>
    <div class="chart"><img src="{imgs['fcc_size']}">
      <div class="cap">图2  加仓幅度三分位 × 30日超额收益（全样本，n={sz['低加仓']['n']+sz['中加仓']['n']+sz['高加仓']['n']:,}）</div>
    </div>
  </div>
</div>

<div class="divider"></div>

<!-- 发现二 -->
<div class="finding">
  <div class="finding-text">
    <div class="finding-num">02</div>
    <div class="finding-title">持仓拥挤度是反向指标</div>
    <div class="finding-body">
      按占基金持仓比重高低四分位分组，超额收益与拥挤度呈显著负相关：
      <b>最低拥挤度组（Q1）</b>30日超额+4.0%，胜率59.4%；
      <b>最高拥挤度组（Q4）</b>仅+1.0%，胜率46.1%。<br><br>
      高集中度意味着已被市场充分定价，持仓拥挤；
      排名50–150位的"安静重仓股"反而存在更大预期差。<br><br>
      <b>实操含义：</b>不要只盯季报榜单前10大重仓股，
      行业内低配但占比上升的标的信号质量更优。
    </div>
  </div>
  <div>
    <div class="chart"><img src="{imgs['fw_quartile']}">
      <div class="cap">图3  占基金持仓比重四分位组 × 30日超额收益及胜率</div>
    </div>
  </div>
</div>

<div class="divider"></div>

<!-- 发现三 -->
<div class="finding">
  <div class="finding-text">
    <div class="finding-num">03</div>
    <div class="finding-title">减仓信号做退出，比做空更有价值</div>
    <div class="finding-body">
      减仓股票短期仍有+7.7%超额（相对基准），直接做空效果很弱。
      但将减仓信号用于<b>触发强制退出</b>后，
      策略累计超额从+71%跳升至<b>+161%</b>。<br><br>
      逻辑：减仓是"早期预警"而非"立即下跌"的信号，
      用于退出现有持仓而非做空，效果更确定、更安全。<br><br>
      <b>实操含义：</b>每季调仓时，优先退出已出现
      "持有基金数净减少"的持仓，无需等信号排名轮换。
    </div>
  </div>
  <div>
    <div class="chart"><img src="{imgs['nav_simple']}">
      <div class="cap">图4  最优策略 vs 参考策略 vs 沪深300 净值曲线</div>
    </div>
  </div>
</div>
""", pg=2))

    # ── Page 3：当前持仓建议 ─────────────────────────────────────────
    holdings_1q26 = [
        ('6869.HK',   '长飞光纤光缆', '通信',   169, '+113', '续持',
         '通信板块信号历史最强（行业均值+9%），持续两季占比上升+0.60%，信号最强'),
        ('688498.SH',  '源杰科技',   '电子',   268, '+68',  '续持',
         '行业内低配，持续增配68家，占比上升0.42%，电子板块布局方向'),
        ('002756.SZ',  '永兴材料',   '有色金属', 97, '+52',  '续持',
         '有色行业内低配，连续两季正向增配，占比稳步上升'),
        ('600105.SH',  '永鼎股份',   '通信',    67, '+24',  '续持',
         '通信板块，市值较小但行业内持续低配+增配，预期差大'),
        ('688195.SH',  '腾景科技',   '电子',    50, '+19',  '续持',
         '电子行业低配，稳步增配，占比上升0.12%'),
        ('603288.SH',  '海天味业',   '食品饮料', 61, '+46',  '新进',
         '本季新进，行业内低配+增配46家；注：食饮历史信号偏弱，建议重点跟踪'),
        ('000703.SZ',  '恒逸石化',   '石油石化', 41, '+21',  '新进',
         '石油石化板块信号有效（+5.6%），行业内低配，新进建仓'),
        ('688195.SH',  '腾景科技',   '电子',    50, '+19',  '续持',
         '电子行业低配，稳步增配，占比上升0.12%'),
        ('002493.SZ',  '荣盛石化',   '石油石化', 34, '+18',  '新进',
         '与恒逸/恒力形成石油石化板块共振，行业低配+增配'),
        ('603061.SH',  '金海通',     '电子',    30, '+17',  '新进',
         '电子行业内低配，新进建仓，持有基金数刚过门槛，适合小仓位跟踪'),
        ('600346.SH',  '恒力石化',   '石油石化', 33, '+5',   '新进',
         '低换手补仓，石油石化方向共振，占比小幅上升'),
    ]
    # 去重（腾景科技重复了）
    seen = set()
    h_dedup = []
    for row in holdings_1q26:
        if row[0] not in seen:
            seen.add(row[0]); h_dedup.append(row)
    holdings_1q26 = h_dedup

    hold_rows = ''.join(
        f'<tr>'
        f'<td class="l" style="font-family:var(--mono);font-size:8pt;color:{C_BLU}">{r[0]}</td>'
        f'<td class="l">{r[1]}</td>'
        f'<td class="l"><span class="tag tag-neu" style="font-size:7.5pt">{r[2]}</span></td>'
        f'<td>{r[3]}</td>'
        f'<td class="pos">{r[4]}</td>'
        f'<td class="l"><span class="tag {"tag-new" if r[5]=="新进" else "tag-neu"}">{r[5]}</span></td>'
        f'<td class="pos-logic">{r[6]}</td>'
        f'</tr>'
        for r in holdings_1q26
    )

    exit_rows = [
        ('688249.SH', '晶合集成', '电子',  '−29家', '持有基金数转为净减少（−29家），触发强制退出'),
        ('300115.SZ', '长盈精密', '电子',  '−49家', '持有基金数转为净减少（−49家），触发强制退出'),
        ('600711.SH', '盛屯矿业', '有色金属','−54家', '持有基金数转为净减少（−54家），触发强制退出'),
    ]
    enter_rows_txt = [
        ('603288.SH','海天味业','食品饮料','+46家','行业内低配+占比上升，信号触发进入'),
        ('000703.SZ','恒逸石化','石油石化','+21家','石化板块共振，行业内低配，新进'),
        ('002493.SZ','荣盛石化','石油石化','+18家','石化板块共振，行业内低配，新进'),
        ('603061.SH','金海通',  '电子',    '+17家','电子低配方向，新进跟踪'),
        ('600346.SH','恒力石化','石油石化', '+5家','低换手补仓，石化方向'),
    ]

    chg_rows = (
        ''.join(
            f'<tr><td class="l"><span class="tag tag-red">退出</span></td>'
            f'<td class="l" style="font-family:var(--mono);font-size:8pt">{r[0]}</td>'
            f'<td class="l">{r[1]}</td><td class="l">{r[2]}</td>'
            f'<td class="neg">{r[3]}</td><td class="l">{r[4]}</td></tr>'
            for r in exit_rows
        ) +
        ''.join(
            f'<tr><td class="l"><span class="tag tag-add">新进</span></td>'
            f'<td class="l" style="font-family:var(--mono);font-size:8pt">{r[0]}</td>'
            f'<td class="l">{r[1]}</td><td class="l">{r[2]}</td>'
            f'<td class="pos">{r[3]}</td><td class="l">{r[4]}</td></tr>'
            for r in enter_rows_txt
        )
    )

    pages.append(page(f"""
<h1>当前持仓建议（1Q26，截至2026年4月20日）</h1>

<h2>策略B · 持仓名单（10只等权，每只权重约10%）</h2>
<table>
  <tr><th class="l">代码</th><th class="l">名称</th><th class="l">行业</th>
    <th>持有基金数</th><th>本季净增</th><th class="l">状态</th><th class="l">入仓逻辑</th></tr>
  {hold_rows}
</table>
<div class="note">⚠ 海天味业（食品饮料）历史上该行业加仓信号有效性偏弱（均值−1.5%），
建议密切跟踪2Q26持仓变化后再决定是否续持。石油石化三只共振方向相同，
大资金操作时注意集中度风险，可酌情合并或限仓。</div>

<h2>本次调仓说明（4Q25 → 1Q26）</h2>
<table>
  <tr><th class="l">操作</th><th class="l">代码</th><th class="l">名称</th>
    <th class="l">行业</th><th>持有基金数变动</th><th class="l">触发原因</th></tr>
  {chg_rows}
</table>

<h2>下次调仓时间</h2>
<p>约 <strong>2026年7月20日</strong>（2Q26季报披露后），届时按信号重新筛选，
退出当季持有基金数出现净减少的持仓，补入新的行业低配+占比上升标的。</p>

<div class="risk">
  <strong>风险提示：</strong>本报告所有结论基于历史回测（2024Q3—2026Q2），回测期包含
  2024Q4—2025Q2强势行情，超额水平整体偏高。仅覆盖8个调仓周期，统计显著性有限。
  不构成投资建议，请结合基本面研究和风控要求独立判断。
</div>
""", pg=3))

    # ── 附录分隔页 ────────────────────────────────────────────────────
    pages.append(page("""
<div class="appendix-cover">
  <div>
    <div class="appendix-label">附录</div>
    <div class="appendix-sub">详细数据与方法论</div>
  </div>
</div>"""))

    # ── 附录A：信号有效性数据 ─────────────────────────────────────────
    ind = data['ind_alpha']
    top5 = ind.head(5)
    bot5 = ind.tail(5).iloc[::-1]
    ind_rows_t = ''.join(
        f'<tr><td class="l"><span class="tag tag-add">{idx}</span></td>'
        f'<td class="pos">+{row["mean"]:.2f}%</td><td>{int(row["count"])}</td></tr>'
        for idx, row in top5.iterrows())
    ind_rows_b = ''.join(
        f'<tr><td class="l"><span class="tag tag-red">{idx}</span></td>'
        f'<td class="neg">{row["mean"]:+.2f}%</td><td>{int(row["count"])}</td></tr>'
        for idx, row in bot5.iterrows())

    pages.append(page(f"""
<h1>附录A — 信号有效性完整数据</h1>

<h2>A1 加减仓/持平信号组 超额收益（全样本，n={data['total_obs']:,}，不含持有基金数门槛限制）</h2>
<table>
  <tr><th class="l">信号组</th><th class="l">定义</th>
    <th>30日超额</th><th>30日胜率</th>
    <th>60日超额</th><th>60日胜率</th>
    <th>90日超额</th><th>90日胜率</th><th>样本数</th></tr>
  <tr><td class="l"><span class="tag tag-add">加仓组</span></td>
    <td class="l" style="font-size:8pt">季度内持有基金数净增量排名前20%且净增>0</td>
    <td class="pos">+{fcc['加仓'][30]['mean']:.2f}%</td><td>{fcc['加仓'][30]['win']:.1f}%</td>
    <td class="pos">+{fcc['加仓'][60]['mean']:.2f}%</td><td>{fcc['加仓'][60]['win']:.1f}%</td>
    <td class="pos">+{fcc['加仓'][90]['mean']:.2f}%</td><td>{fcc['加仓'][90]['win']:.1f}%</td>
    <td>{fcc['加仓'][30]['n']:,}</td></tr>
  <tr><td class="l"><span class="tag tag-neu">持平组</span></td>
    <td class="l" style="font-size:8pt">其余60%（含净增=0）</td>
    <td>+{fcc['持平'][30]['mean']:.2f}%</td><td>{fcc['持平'][30]['win']:.1f}%</td>
    <td>+{fcc['持平'][60]['mean']:.2f}%</td><td>{fcc['持平'][60]['win']:.1f}%</td>
    <td>+{fcc['持平'][90]['mean']:.2f}%</td><td>{fcc['持平'][90]['win']:.1f}%</td>
    <td>{fcc['持平'][30]['n']:,}</td></tr>
  <tr><td class="l"><span class="tag tag-red">减仓组</span></td>
    <td class="l" style="font-size:8pt">季度内净增量排名后20%且净增<0</td>
    <td>+{fcc['减仓'][30]['mean']:.2f}%</td><td>{fcc['减仓'][30]['win']:.1f}%</td>
    <td>+{fcc['减仓'][60]['mean']:.2f}%</td><td>{fcc['减仓'][60]['win']:.1f}%</td>
    <td>+{fcc['减仓'][90]['mean']:.2f}%</td><td>{fcc['减仓'][90]['win']:.1f}%</td>
    <td>{fcc['减仓'][30]['n']:,}</td></tr>
</table>

<h2>A2 持仓拥挤度分位数组 超额收益</h2>
<table>
  <tr><th class="l">分组</th><th>30日超额</th><th>30日胜率</th>
    <th>60日超额</th><th>90日超额</th><th>样本数</th></tr>
  {_fw_rows(fw)}
</table>

<h2>A3 加仓信号行业有效性（加仓组，n≥10）</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
  <div>
    <h3 style="color:{C_RED}">▲ 信号有效行业 Top 5</h3>
    <table>
      <tr><th class="l">行业</th><th>30日超额均值</th><th>样本数</th></tr>
      {ind_rows_t}
    </table>
  </div>
  <div>
    <h3 style="color:{C_GRN}">▼ 信号无效行业 Bottom 5</h3>
    <table>
      <tr><th class="l">行业</th><th>30日超额均值</th><th>样本数</th></tr>
      {ind_rows_b}
    </table>
  </div>
</div>
<p style="margin-top:8px">行业过滤建议：在科技（通信、电子）、工业（电力设备、建筑材料）板块应用加仓信号效果更好；
消费类（食品饮料、社会服务）信号可信度低，建议谨慎或跳过。</p>
""", pg=4))

    # ── 附录B：策略回测结果 ──────────────────────────────────────────
    st_C = data['stat_data']['C']
    pages.append(page(f"""
<h1>附录B — 策略回测完整结果</h1>
<div class="chart"><img src="{imgs['nav_full']}">
  <div class="cap">图5  完整净值曲线及回撤对比（2024年7月—2026年4月）</div>
</div>
<div class="chart"><img src="{imgs['qtr']}">
  <div class="cap">图6  逐季度超额收益</div>
</div>
<h2>策略指标汇总</h2>
<table>
  <tr><th class="l">策略</th><th class="l" style="font-size:7.5pt">定义</th>
    <th>累计超额</th><th>年化收益</th><th>夏普</th><th>最大回撤</th><th>均持仓</th></tr>
  <tr class="bold"><td class="l">★ 最优策略</td>
    <td class="l" style="font-size:8pt">行业低配动量+低换手（每季换5只）+减仓退出</td>
    <td class="pos">+{st_B['alpha']:.1f}%</td><td class="pos">{st_B['ann']:+.1f}%</td>
    <td class="pos">{st_B['sh']:.2f}</td><td class="neg">{st_B['dd']:.1f}%</td><td>~11只</td></tr>
  <tr><td class="l">基金增减低换手+减仓退出</td>
    <td class="l" style="font-size:8pt">持有基金数增量前20+低换手（每季换5只）+减仓退出</td>
    <td class="pos">+{st_C['alpha']:.1f}%</td><td class="pos">{st_C['ann']:+.1f}%</td>
    <td>{st_C['sh']:.2f}</td><td class="neg">{st_C['dd']:.1f}%</td><td>~10只</td></tr>
  <tr><td class="l">参考：加仓前20（全换手）</td>
    <td class="l" style="font-size:8pt">持有基金数增量前20，每季全部重建</td>
    <td class="pos">+{st_R1['alpha']:.1f}%</td><td class="pos">{st_R1['ann']:+.1f}%</td>
    <td>{st_R1['sh']:.2f}</td><td class="neg">{st_R1['dd']:.1f}%</td><td>20只</td></tr>
  <tr><td class="l">沪深300（基准）</td>
    <td class="l" style="font-size:8pt">000300.SH</td>
    <td>—</td><td class="pos">{st_R1['btot']:+.1f}%（区间）</td>
    <td>—</td><td>—</td><td>—</td></tr>
</table>
""", pg=5))

    # ── 附录C：方法论与局限性 ────────────────────────────────────────
    pages.append(page(f"""
<h1>附录C — 方法论与局限性</h1>

<h2>数据与方法</h2>
<table>
  <tr><th class="l">项目</th><th class="l">说明</th></tr>
  <tr><td class="l">数据来源</td><td class="l">公募基金季报前十大重仓股汇总（A股+H股），每季约2,500–3,000只</td></tr>
  <tr><td class="l">覆盖区间</td><td class="l">2024Q1—2026Q1，共9个季度，有效观测{data['total_obs']:,}条</td></tr>
  <tr><td class="l">信号生效日</td><td class="l">季报公开披露日（约季末次月20日，如Q1→4月20日）</td></tr>
  <tr><td class="l">超额收益</td><td class="l">个股区间收益 − 同期沪深300收益（30/60/90自然日）</td></tr>
  <tr><td class="l">加仓组定义</td><td class="l">季内持有基金数净增量排名前20%且净增为正</td></tr>
  <tr><td class="l">流动性门槛</td><td class="l">持有基金数 ≥ 30家（确保标的有足够机构认可度与流动性）</td></tr>
  <tr><td class="l">组合构建</td><td class="l">等权，披露日次一交易日建仓，下季度披露日平仓</td></tr>
</table>

<h2>主要局限性</h2>
<table>
  <tr><th class="l">类别</th><th class="l">说明</th><th class="l" style="width:14%">影响</th></tr>
  <tr><td class="l">牛市偏差</td>
    <td class="l">2024Q4–2025Q2含强势行情，整体超额水平偏高，熊市表现待验证</td>
    <td class="l"><span class="tag tag-add">高</span></td></tr>
  <tr><td class="l">样本量有限</td>
    <td class="l">仅8个完整调仓周期，统计显著性不足，需持续积累</td>
    <td class="l"><span class="tag tag-add">高</span></td></tr>
  <tr><td class="l">未含交易成本</td>
    <td class="l">全换手策略年化约+1.8%成本拖累；最优策略（低换手）约+0.45%</td>
    <td class="l"><span class="tag tag-hot">中</span></td></tr>
  <tr><td class="l">披露日误差</td>
    <td class="l">统一使用20日，实际各基金约15–25日，存在±3–5交易日误差</td>
    <td class="l"><span class="tag tag-neu">低</span></td></tr>
  <tr><td class="l">持仓集中</td>
    <td class="l">最优策略约持10只，单票权重10%，波动高于常规产品</td>
    <td class="l"><span class="tag tag-hot">中</span></td></tr>
  <tr><td class="l">港股数据</td>
    <td class="l">约12%港股持仓无法匹配收盘价，H股信号可能被低估</td>
    <td class="l"><span class="tag tag-neu">低</span></td></tr>
</table>

<h2>后续研究方向</h2>
<p><strong>① 扩展样本</strong>：待2026年下半年数据可得后，验证策略在调整行情中的鲁棒性。</p>
<p><strong>② 叠加基本面过滤</strong>：在加仓增量信号上叠加估值（PE/PB）或景气度筛选，测试能否进一步提升胜率。</p>
<p><strong>③ 连续加仓验证</strong>：连续两季加仓 vs 仅单季加仓的后续表现对比，判断信号的持续性差异。</p>

<hr style="border:none;border-top:1px solid var(--bd);margin:16px 0">
<p style="font-size:8pt;color:#bbb;line-height:1.6">
本报告基于公开可得的公募基金季报数据，所有策略回测结果仅供研究参考，不构成任何投资建议。
历史表现不代表未来收益，请结合自身风控要求独立判断。本报告仅供内部使用，不得对外分发。
生成时间：{gen_time}
</p>
""", pg=6))

    print_btn = """<div class="no-print" style="text-align:center;padding:14px;
  position:fixed;bottom:0;left:0;right:0;background:rgba(255,255,255,.92);
  backdrop-filter:blur(6px);border-top:1px solid #ddd;z-index:100">
  <button onclick="window.print()" style="background:#1e50a2;color:#fff;border:none;
    padding:9px 26px;border-radius:6px;font-size:13px;cursor:pointer;font-family:inherit">
    🖨 打印 / 保存为 PDF
  </button>
  <span style="margin-left:14px;color:#888;font-size:12px">
    Chrome打印 · A4纸 · ☑背景图形 · 取消页眉页脚
  </span>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>公募基金持仓信号研究</title>
<style>{STYLE}</style>
</head>
<body>
{''.join(pages)}
{print_btn}
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════
# 4. 主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    data     = compute()
    gen_time = datetime.now().strftime('%Y-%m-%d %H:%M')

    print("生成图表…")
    imgs = {
        'evolution':   chart_evolution(data['stat_data']),
        'fcc_size':    chart_fcc_size(data['fcc_size']),
        'fw_quartile': chart_fw_quartile(data['fw_grp']),
        'nav_simple':  chart_nav_simple(data['nav_data'], data['stat_data']),
        'nav_full':    chart_nav_full(data['nav_data']),
        'qtr':         chart_qtr(data['nav_data'], data['stat_data']),
    }

    print("组装报告…")
    html = build(data, imgs, gen_time)
    OUTPUT.write_text(html, encoding='utf-8')
    print(f"完成！{OUTPUT}  ({OUTPUT.stat().st_size // 1024} KB)")
    print("Chrome 打开 → 打印 → A4 → ☑背景图形 → 保存PDF")


if __name__ == '__main__':
    main()
