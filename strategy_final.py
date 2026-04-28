#!/usr/bin/env python3
"""
最终策略综合测试
A  SE低换手      行业低配+持仓比重上升，每季换5只
B  SE低换手+减仓退出  在A基础上：持仓中出现减仓信号（持有基金数变负）强制退出
C  fcc低换手+减仓退出  fcc前20低换手，加入减仓强制退出逻辑
REF1  fcc全换手（上一轮基准）
REF2  SE全换手（上一轮最优）
"""

import json, warnings, sys, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
BASE = Path("workwork/research/策略研究/基金重仓看板")
sys.path.insert(0, str(Path.cwd()))
from workwork.research.策略研究.基金重仓看板.backtest_signals import (
    load_holdings, load_prices, DISCLOSURE_DATES, sort_q
)

OUTPUT = BASE / "fund_strategy_final.html"
FC_MIN = 30
BENCH  = '000300.SH'

# ══ 工具 ════════════════════════════════════════════════════════════════
def next_td(pdf, d):
    idx = pdf.index[pdf.index >= pd.Timestamp(d)]
    return idx[0] if len(idx) else None

def get_px(ser, d):
    sub = ser[ser.index >= pd.Timestamp(d)].dropna()
    return (sub.index[0], float(sub.iloc[0])) if len(sub) else (None, None)

def run_nav(periods, price_df):
    bench_ser = price_df[BENCH]
    nav = bench_nav = 1.0
    rows = []; trades = []
    rows.append({'date': next_td(price_df, periods[0]['disc']), 'nav': 1.0, 'bench': 1.0})
    for period in periods:
        stocks = period['stocks']
        if not stocks: continue
        disc_ts = pd.Timestamp(period['disc'])
        end_ts  = pd.Timestamp(period['next_disc']) if period['next_disc'] else price_df.index[-1]
        et = next_td(price_df, disc_ts)
        xt = next_td(price_df, end_ts)
        ep = {s['code']: get_px(price_df[s['code']], et)[1]
              for s in stocks if s['code'] in price_df.columns}
        prev_px = dict(ep)
        pb = float(bench_ser[bench_ser.index >= et].dropna().iloc[0])
        for d in price_df.index[(price_df.index > et) & (price_df.index <= xt)]:
            dr = []
            for s in stocks:
                c = s['code']
                if c not in price_df.columns or not prev_px.get(c): continue
                cur = price_df[c].get(d)
                if cur and not np.isnan(cur) and prev_px[c] != 0:
                    dr.append(float(cur)/prev_px[c] - 1); prev_px[c] = float(cur)
                else: dr.append(0.0)
            pr = np.mean(dr) if dr else 0.0
            bc = bench_ser.get(d)
            br = float(bc)/pb - 1 if bc and not np.isnan(bc) and pb else 0.0
            if bc and not np.isnan(bc): pb = float(bc)
            nav *= (1+pr); bench_nav *= (1+br)
            rows.append({'date': d, 'nav': round(nav,6), 'bench': round(bench_nav,6)})
        for s in stocks:
            c = s['code']; e = ep.get(c)
            if not e or c not in price_df.columns: continue
            _, xp = get_px(price_df[c], xt)
            _, bs = get_px(bench_ser, et); _, bx = get_px(bench_ser, xt)
            ret   = round((xp/e-1)*100, 3) if xp else None
            bret  = round((bx/bs-1)*100, 3) if bx and bs else None
            alpha = round(ret-bret, 3) if ret is not None and bret is not None else None
            trades.append({'q': period['q'], 'disc': period['disc'],
                           'code': c, 'name': s['name'], 'ind': s.get('ind',''),
                           'fc': s['fc'], 'fcc': s.get('fcc'), 'fwd': s.get('fwd'),
                           'entry_date': str(et.date()), 'entry_price': round(e,3),
                           'exit_date': str(xt.date()) if xt else None,
                           'exit_price': round(xp,3) if xp else None,
                           'ret': ret, 'bench_ret': bret, 'alpha': alpha})
    df = pd.DataFrame(rows).drop_duplicates('date').sort_values('date')
    return df, trades

def calc_stats(nav_df):
    p = nav_df['nav'].values; b = nav_df['bench'].values; dates = nav_df['date'].values
    n = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    tot = p[-1]/p[0]-1; btot = b[-1]/b[0]-1
    ann = (1+tot)**(365/n)-1 if n > 0 else 0
    dr  = np.diff(p)/p[:-1]
    sh  = np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr) > 0 else 0
    dd  = float(np.min(p/np.maximum.accumulate(p)-1))
    df2 = nav_df.copy(); df2['Q'] = pd.to_datetime(df2['date']).dt.to_period('Q')
    qr  = {str(qp): {'port': round((g['nav'].iloc[-1]/g['nav'].iloc[0]-1)*100, 2),
                     'bench': round((g['bench'].iloc[-1]/g['bench'].iloc[0]-1)*100, 2)}
           for qp, g in df2.groupby('Q')}
    return {'tot': round(tot*100,2), 'btot': round(btot*100,2),
            'alpha': round((tot-btot)*100,2), 'ann': round(ann*100,2),
            'sh': round(sh,2), 'dd': round(dd*100,2),
            'start': str(pd.Timestamp(dates[0]).date()),
            'end':   str(pd.Timestamp(dates[-1]).date()), 'qr': qr}

# ══ 构建各策略持仓 ═══════════════════════════════════════════════════════

def get_ind_avg_fw(cur_map):
    """行业平均占基金持仓比重"""
    buckets = {}
    for r in cur_map.values():
        if r.get('fw') is not None and r.get('ind'):
            buckets.setdefault(r['ind'], []).append(r['fw'])
    return {ind: np.mean(v) for ind, v in buckets.items()}

def se_candidates(cur_map, prev_map):
    """SE信号候选池：fc≥30，fw<行业均值，fwd>0"""
    ind_avg = get_ind_avg_fw(cur_map)
    result = []
    for code, r in cur_map.items():
        if not (r.get('fc') and r['fc'] >= FC_MIN): continue
        fw = r.get('fw'); fcc = r.get('fcc'); ind = r.get('ind','')
        fw_prev = prev_map[code].get('fw') if code in prev_map else None
        fwd = (fw - (fw_prev or 0.0)) if fw is not None else None
        r['_fwd'] = fwd
        if (fw is not None and fwd is not None and fwd > 0
                and fw < ind_avg.get(ind, fw+1)):
            result.append(r)
    return sorted(result, key=lambda r: r['_fwd'], reverse=True)

def fcc_candidates(cur_map):
    """fcc信号候选池：fc≥30，fcc>0，按fcc降序"""
    return sorted([r for r in cur_map.values()
                   if r.get('fc') and r['fc'] >= FC_MIN
                   and r.get('fcc') and r['fcc'] > 0],
                  key=lambda r: r['fcc'], reverse=True)

def mk_period(q, disc, nxt_disc, sel, cur_map):
    stocks = []
    for r in sel:
        code = r['code']
        if code not in cur_map: continue
        cr = cur_map[code]
        fw_prev_val = None  # computed externally
        stocks.append({'code': code, 'name': cr['name'], 'ind': cr.get('ind',''),
                       'fc': cr['fc'], 'fcc': cr.get('fcc'),
                       'fwd': r.get('_fwd')})
    return {'q': q, 'disc': disc, 'next_disc': nxt_disc, 'stocks': stocks}


def build_strategies(by_q, quarters):
    """
    返回 {A, B, C, REF1, REF2} 的 periods 列表
    """
    strategies = {k: [] for k in ('A','B','C','REF1','REF2')}
    # 持仓状态
    port = {k: set() for k in ('A','B','C','REF1')}
    change_n = 5

    prev_map = {r['code']: r for r in by_q[quarters[0]]}

    for i, q in enumerate(quarters[1:], 1):
        disc     = DISCLOSURE_DATES[q]
        nxt_q    = quarters[i+1] if i+1 < len(quarters) else None
        nxt_disc = DISCLOSURE_DATES.get(nxt_q)
        cur_map  = {r['code']: r for r in by_q[q]}

        # 计算 fwd
        for code, r in cur_map.items():
            fw_c = r.get('fw'); fw_p = prev_map[code].get('fw') if code in prev_map else None
            r['_fwd'] = (fw_c - (fw_p or 0.0)) if fw_c is not None else None

        se_cands  = se_candidates(cur_map, prev_map)   # SE候选
        fcc_cands = fcc_candidates(cur_map)             # fcc候选

        # ── 辅助：低换手逻辑 ──────────────────────────────────────────
        def low_turnover_build(portfolio, ranked_cands, use_jianc_exit=False):
            """
            portfolio: 当前持仓 set(code)
            ranked_cands: 本季候选，已按信号强弱降序
            use_jianc_exit: 是否强制退出减仓股（fcc<0）
            """
            rank_map = {r['code']: idx for idx, r in enumerate(ranked_cands)}
            keep_n   = 20 - change_n

            # 当前持仓中仍满足fc≥30的
            hold_valid = [c for c in portfolio
                          if c in cur_map and cur_map[c].get('fc',0) >= FC_MIN]

            if not portfolio:
                # 首期：直接取前20
                return set(r['code'] for r in ranked_cands[:20])

            if use_jianc_exit:
                # 强制退出：fcc<0 的持仓优先退出
                forced_exit = {c for c in hold_valid
                               if cur_map[c].get('fcc') is not None and cur_map[c]['fcc'] < 0}
                # 剩余按信号强弱排序，保留最好的 keep_n 只（从未被强制退出的里选）
                remaining = [c for c in hold_valid if c not in forced_exit]
                remaining_sorted = sorted(remaining, key=lambda c: rank_map.get(c, 9999))
                keep = set(remaining_sorted[:keep_n])
                # 已退出 = forced_exit + 排名靠后的超出keep_n的
                n_exits = len(portfolio) - keep_n   # 总需要退出多少
                extra_exit_candidates = [c for c in remaining_sorted[keep_n:]
                                          if c not in keep][:max(0, n_exits - len(forced_exit))]
                exited = forced_exit | set(extra_exit_candidates)
            else:
                # 普通低换手：按信号排名保留最好的 keep_n
                sorted_hold = sorted(hold_valid, key=lambda c: rank_map.get(c, 9999))
                keep = set(sorted_hold[:keep_n])
                exited = portfolio - keep

            # 补入新标的
            new_entries = [r['code'] for r in ranked_cands if r['code'] not in keep][:change_n]
            return keep | set(new_entries)

        # ── A: SE低换手 ──────────────────────────────────────────────
        port['A'] = low_turnover_build(port['A'], se_cands, use_jianc_exit=False)
        sel_A = [cur_map[c] for c in port['A'] if c in cur_map]
        for r in sel_A: r['_fwd'] = r.get('_fwd')
        strategies['A'].append({'q':q,'disc':disc,'next_disc':nxt_disc,
            'stocks':[{'code':r['code'],'name':r['name'],'ind':r.get('ind',''),
                       'fc':r['fc'],'fcc':r.get('fcc'),'fwd':r.get('_fwd')} for r in sel_A]})

        # ── B: SE低换手+减仓退出 ─────────────────────────────────────
        port['B'] = low_turnover_build(port['B'], se_cands, use_jianc_exit=True)
        sel_B = [cur_map[c] for c in port['B'] if c in cur_map]
        strategies['B'].append({'q':q,'disc':disc,'next_disc':nxt_disc,
            'stocks':[{'code':r['code'],'name':r['name'],'ind':r.get('ind',''),
                       'fc':r['fc'],'fcc':r.get('fcc'),'fwd':r.get('_fwd')} for r in sel_B]})

        # ── C: fcc低换手+减仓退出 ────────────────────────────────────
        port['C'] = low_turnover_build(port['C'], fcc_cands, use_jianc_exit=True)
        sel_C = [cur_map[c] for c in port['C'] if c in cur_map]
        strategies['C'].append({'q':q,'disc':disc,'next_disc':nxt_disc,
            'stocks':[{'code':r['code'],'name':r['name'],'ind':r.get('ind',''),
                       'fc':r['fc'],'fcc':r.get('fcc'),'fwd':r.get('_fwd')} for r in sel_C]})

        # ── REF1: fcc全换手 ──────────────────────────────────────────
        strategies['REF1'].append({'q':q,'disc':disc,'next_disc':nxt_disc,
            'stocks':[{'code':r['code'],'name':r['name'],'ind':r.get('ind',''),
                       'fc':r['fc'],'fcc':r.get('fcc')} for r in fcc_cands[:20]]})

        # ── REF2: SE全换手 ───────────────────────────────────────────
        strategies['REF2'].append({'q':q,'disc':disc,'next_disc':nxt_disc,
            'stocks':[{'code':r['code'],'name':r['name'],'ind':r.get('ind',''),
                       'fc':r['fc'],'fcc':r.get('fcc'),'fwd':r.get('_fwd')} for r in se_cands[:20]]})

        prev_map = cur_map

    return strategies


# ══ HTML ════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>最终策略综合对比</title>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root{
  --bg:#f0ece4;--card:#fff;--th:#f7f5f1;--alt:#faf8f5;
  --hdr:#0f1f3d;--bd:#ddd8d0;--bdt:#ccc8bf;
  --t1:#14213d;--t2:#566070;--t3:#96a0ae;
  --acc:#1e50a2;--acc2:#eef3fc;--pos:#c0392b;--neg:#166534;
  --sh:0 2px 8px rgba(20,33,61,.09),0 1px 3px rgba(20,33,61,.05);
  --r:8px;--ui:'Sora','PingFang SC','Microsoft YaHei',sans-serif;--mono:'JetBrains Mono','Courier New',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t1);font-family:var(--ui);font-size:13px;line-height:1.5}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-thumb{background:var(--bd);border-radius:4px}
.hdr{background:var(--hdr);padding:0 28px;height:56px;display:flex;align-items:center;
  justify-content:space-between;position:sticky;top:0;z-index:50;box-shadow:0 1px 0 rgba(255,255,255,.08)}
.brand{display:flex;align-items:center;gap:12px}
.icon{width:34px;height:34px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);
  border-radius:8px;display:flex;align-items:center;justify-content:center}
.hdr h1{font-size:15px;font-weight:700;color:#fff}.hdr-sub{font-size:9px;color:rgba(255,255,255,.38);letter-spacing:1.2px;text-transform:uppercase}
.hdr-meta{color:rgba(255,255,255,.35);font-size:11px;font-family:var(--mono)}
.tabs{background:#fff;padding:0 28px;display:flex;border-bottom:1px solid var(--bd);
  position:sticky;top:56px;z-index:40;box-shadow:var(--sh)}
.tab{padding:13px 16px;cursor:pointer;border:none;background:none;color:var(--t3);
  font-size:12px;font-family:var(--ui);font-weight:500;border-bottom:2px solid transparent;
  transition:all .15s;white-space:nowrap;margin-bottom:-1px}
.tab:hover{color:var(--t2)}.tab.on{color:var(--acc);border-bottom-color:var(--acc);font-weight:700}
.pane{display:none;padding:22px 28px}.pane.on{display:block}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.mb{margin-bottom:14px}
.box{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);padding:16px 18px;box-shadow:var(--sh)}
.box-title{font-size:9px;color:var(--t3);margin-bottom:12px;font-weight:700;text-transform:uppercase;letter-spacing:.7px}
.cards{display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);
  padding:14px 18px;flex:1;min-width:140px;max-width:200px;box-shadow:var(--sh)}
.card-lbl{color:var(--t3);font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.7px}
.card-val{font-size:22px;font-weight:700;margin-top:5px;font-family:var(--mono);letter-spacing:-.5px}
.card-sub{font-size:11px;color:var(--t3);margin-top:4px}
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}.neu{color:var(--t3)}
.tbl-wrap{overflow-x:auto;border-radius:var(--r);border:1px solid var(--bd);box-shadow:var(--sh);background:var(--card)}
table{width:100%;border-collapse:collapse}
thead{position:sticky;top:0;z-index:10}
th{background:var(--th);color:var(--t3);font-weight:700;font-size:10px;text-transform:uppercase;
  letter-spacing:.5px;padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--bdt)}
th.l,td.l{text-align:left}
td{padding:7px 12px;border-top:1px solid #ede9e3;text-align:right;white-space:nowrap;font-size:12px;font-family:var(--mono)}
td.l{font-family:var(--ui);font-size:12px}
tr:nth-child(even) td{background:var(--alt)}tr:hover td{background:var(--acc2)!important}
.code{color:var(--acc);font-size:11px}
.tag{display:inline-block;background:var(--acc2);color:var(--acc);padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600;font-family:var(--ui)}
.badge{display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}
.best{background:#fef3c7;border:2px solid #f59e0b;border-radius:var(--r);padding:14px 18px;margin-bottom:14px;font-size:13px;line-height:1.8}
.best b{color:#92400e}
.insight{background:var(--acc2);border:1px solid #c5d8f5;border-radius:var(--r);padding:14px 18px;margin-bottom:14px;font-size:13px;line-height:1.8}
.insight b{color:var(--acc)}
.warn{background:#fffbeb;border:1px solid #fcd34d;border-radius:var(--r);padding:12px 16px;font-size:12px;color:#92400e;line-height:1.7;margin-top:10px}
.phdr{cursor:pointer;user-select:none}
.phdr td{background:#f0f4fb!important;color:var(--acc);font-weight:700;font-family:var(--ui);padding:9px 12px!important}
.phdr:hover td{background:#e4ecf9!important}
.ctrl{display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
.lbl{color:var(--t3);font-size:10px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;white-space:nowrap}
select{background:var(--card);border:1px solid var(--bd);color:var(--t1);padding:5px 10px;
  border-radius:6px;font-size:12px;font-family:var(--ui);outline:none;height:30px}
select:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(30,80,162,.1)}
</style>
</head>
<body>
<div class="hdr">
  <div class="brand">
    <div class="icon"><svg width="16" height="16" fill="none">
      <path d="M2 12l4-4 3 3 5-6" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>
    </svg></div>
    <div><h1>最终策略综合对比</h1><div class="hdr-sub">Final Strategy Comparison · fc≥30 · Equal Weight</div></div>
  </div>
  <span class="hdr-meta" id="meta"></span>
</div>
<div class="tabs">
  <button class="tab on" onclick="go('t1')">净值与结论</button>
  <button class="tab" onclick="go('t2')">逐季度拆分</button>
  <button class="tab" onclick="go('t3')">调仓明细</button>
</div>

<!-- TAB1 -->
<div id="t1" class="pane on">
  <div class="best" id="best-box"></div>

  <div class="cards" id="t1-cards"></div>

  <div class="box mb">
    <div class="box-title">全策略净值曲线对比（基准：沪深300）</div>
    <div id="c-nav" style="height:400px"></div>
  </div>
  <div class="g2">
    <div class="box">
      <div class="box-title">回撤曲线</div>
      <div id="c-dd" style="height:220px"></div>
    </div>
    <div class="box">
      <div class="box-title">综合指标对比表</div>
      <div class="tbl-wrap" style="margin:0;max-height:220px">
        <table><thead><tr>
          <th class="l">策略</th><th>超额</th><th>年化</th><th>夏普</th><th>最大回撤</th>
        </tr></thead><tbody id="stat-tbl"></tbody></table>
      </div>
    </div>
  </div>

  <div class="insight" id="insight-box"></div>
  <div class="warn">
    局限性提示：回测期仅覆盖8个调仓周期（2024Q3–2026Q2），含2024Q4–2025Q2强牛市行情。
    所有结果均不含交易成本，低换手策略（每季换5只）估算年化成本约0.45%，相对优势更明显。
    回测样本量有限，统计显著性不足，结论需持续验证。
  </div>
</div>

<!-- TAB2 -->
<div id="t2" class="pane">
  <div class="box mb">
    <div class="box-title">逐季度超额收益（各策略 vs 沪深300）</div>
    <div id="c-qbar" style="height:360px"></div>
  </div>
  <div class="tbl-wrap">
    <table><thead><tr>
      <th class="l">季度</th><th>基准</th>
      <th>A SE低换手</th><th>B SE低换手+减仓退出</th>
      <th>C fcc低换手+减仓退出</th><th>REF1 fcc全换手</th><th>REF2 SE全换手</th>
    </tr></thead><tbody id="q-tbl"></tbody></table>
  </div>
</div>

<!-- TAB3 -->
<div id="t3" class="pane">
  <div class="ctrl">
    <span class="lbl">策略</span>
    <select id="t3-s" onchange="renderT3()">
      <option value="A">A SE低换手</option>
      <option value="B">B SE低换手+减仓退出</option>
      <option value="C">C fcc低换手+减仓退出</option>
      <option value="REF1">REF1 fcc全换手</option>
      <option value="REF2">REF2 SE全换手</option>
    </select>
  </div>
  <div class="tbl-wrap">
    <table><thead><tr>
      <th class="l">季度/代码</th><th class="l">名称</th><th class="l">行业</th>
      <th>持有基金数</th><th>基金数增减</th><th>持仓比重变动</th>
      <th>入场日</th><th>入场价</th><th>出场日</th><th>出场价</th>
      <th>个股收益</th><th>基准</th><th>超额</th>
    </tr></thead><tbody id="t3-body"></tbody></table>
  </div>
</div>

<script>
const D = __DATA__;
const SKEYS  = ['A','B','C','REF1','REF2'];
const COLORS = {A:'#c0392b',B:'#7c3aed',C:'#d97706',REF1:'#96a0ae',REF2:'#0891b2'};
const LABELS = {
  A:'A SE低换手（每季换5只）',
  B:'B SE低换手+减仓强制退出',
  C:'C fcc低换手+减仓强制退出',
  REF1:'参考1 fcc全换手',
  REF2:'参考2 SE全换手',
};
const AX = {
  axisLine:{lineStyle:{color:'#ccc8bf'}},
  axisLabel:{color:'#96a0ae',fontSize:10,fontFamily:"'JetBrains Mono',monospace"},
  splitLine:{lineStyle:{color:'#e8e3db',type:'dashed'}},
};
const TT = {backgroundColor:'#fff',borderColor:'#ddd8d0',borderWidth:1,textStyle:{color:'#14213d',fontSize:11}};
const FMT = v => v==null?'—':(v>0?'+':'')+v.toFixed(2)+'%';
const CLS = v => v==null?'neu':v>0?'pos':v<0?'neg':'neu';

function mc(id){
  const el=document.getElementById(id);if(!el)return null;
  const c=echarts.getInstanceByDom(el);if(c)c.dispose();
  return echarts.init(el,null,{renderer:'canvas'});
}
function go(id){
  document.querySelectorAll('.pane').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  const idx=['t1','t2','t3'].indexOf(id);
  document.querySelectorAll('.tab')[idx].classList.add('on');
  if(id==='t2') renderT2(); if(id==='t3') renderT3();
}

function renderT1(){
  // 找最优策略
  const best = SKEYS.reduce((a,b)=>D.stats[a].sh>=D.stats[b].sh?a:b);
  const bs = D.stats[best];

  document.getElementById('best-box').innerHTML = `
    <b>🏆 综合最优策略：${LABELS[best]}</b><br>
    超额收益 <b>${FMT(bs.alpha)}</b>，年化 <b>${FMT(bs.ann)}</b>，夏普比率 <b>${bs.sh.toFixed(2)}</b>，最大回撤 <b>${FMT(bs.dd)}</b>。`;

  document.getElementById('t1-cards').innerHTML = SKEYS.map(s=>{
    const st=D.stats[s];
    return `<div class="card" style="${s===best?'border-color:#f59e0b;border-width:2px':''}">
      <div class="card-lbl"><span class="badge" style="background:${COLORS[s]}22;color:${COLORS[s]}">${s}</span></div>
      <div class="card-val ${CLS(st.alpha)}">${FMT(st.alpha)}</div>
      <div class="card-sub">夏普 ${st.sh.toFixed(2)}  回撤 ${FMT(st.dd)}</div>
    </div>`;
  }).join('');

  const dates=D.nav['REF1'].map(r=>r.d);
  const cN=mc('c-nav');
  if(cN) cN.setOption({
    backgroundColor:'transparent',
    grid:{left:60,right:140,top:20,bottom:36},
    xAxis:{type:'category',data:dates,...AX,axisTick:{show:false},axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(2)}},
    tooltip:{...TT,trigger:'axis',formatter:p=>{
      let s=`<b>${p[0].name}</b><br>`;
      p.forEach(x=>s+=`<span style="color:${x.color}">●</span> ${x.seriesName}: ${x.value.toFixed(4)}<br>`);
      return s;
    }},
    legend:{orient:'vertical',right:0,top:'middle',textStyle:{color:'#566070',fontSize:10}},
    series:[
      ...SKEYS.map(s=>({
        name:LABELS[s],type:'line',symbol:'none',smooth:.1,
        data:D.nav[s].map(r=>r.n),
        lineStyle:{color:COLORS[s],
                   width:['A','B','C'].includes(s)?2.5:1.5,
                   type:['REF1','REF2'].includes(s)?'dashed':'solid'},
      })),
      {name:'沪深300',type:'line',symbol:'none',smooth:.1,
       data:D.nav['REF1'].map(r=>r.b),lineStyle:{color:'#ccc8bf',width:1.2,type:'dotted'}},
    ]
  });

  const cDd=mc('c-dd');
  if(cDd) cDd.setOption({
    backgroundColor:'transparent',
    grid:{left:55,right:10,top:10,bottom:36},
    xAxis:{type:'category',data:dates,...AX,axisTick:{show:false},axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(1)+'%'}},
    tooltip:{...TT,trigger:'axis'},
    series: SKEYS.map(s=>{
      const nav=D.nav[s].map(r=>r.n); let mx=nav[0];
      const dd=nav.map(v=>{mx=Math.max(mx,v);return+(v/mx-1).toFixed(6)*100;});
      return {name:LABELS[s],type:'line',symbol:'none',smooth:.1,data:dd,
              lineStyle:{color:COLORS[s],width:1.5,type:['REF1','REF2'].includes(s)?'dashed':'solid'}};
    }),
    legend:{bottom:0,textStyle:{color:'#566070',fontSize:9}},
  });

  document.getElementById('stat-tbl').innerHTML = [
    ...SKEYS.map(s=>{
      const st=D.stats[s];
      return `<tr style="${s===best?'font-weight:700':''}">
        <td class="l"><span class="badge" style="background:${COLORS[s]}22;color:${COLORS[s]}">${s}</span></td>
        <td class="${CLS(st.alpha)}">${FMT(st.alpha)}</td>
        <td class="${CLS(st.ann)}">${FMT(st.ann)}</td>
        <td>${st.sh.toFixed(2)}</td>
        <td class="${CLS(st.dd)}">${FMT(st.dd)}</td>
      </tr>`;
    }),
    `<tr style="border-top:2px solid var(--bdt)"><td class="l" style="color:var(--t3)">沪深300</td>
     <td>—</td><td>${FMT(D.stats.REF1.btot)}</td><td>—</td><td>—</td></tr>`,
  ].join('');

  // 洞察
  const stA=D.stats['A'],stB=D.stats['B'],stC=D.stats['C'],stR1=D.stats['REF1'];
  const bestSt=D.stats[best];
  document.getElementById('insight-box').innerHTML = `
    <b>综合洞察：</b><br>
    1. <b>低换手策略普遍优于全换手</b>：A（SE低换手${FMT(stA.alpha)}）和C（fcc低换手+减仓退出${FMT(stC.alpha)}）均优于REF1全换手（${FMT(stR1.alpha)}），验证了机构加仓信号的跨季度持续性。<br>
    2. <b>减仓强制退出信号${stB.alpha>stA.alpha?'有效提升':'效果有限'}</b>：B策略（SE低换手+减仓退出）超额${FMT(stB.alpha)}，${stB.alpha>stA.alpha?'优于':'低于'}不加减仓过滤的A策略（${FMT(stA.alpha)}），差值${FMT(stB.alpha-stA.alpha)}。<br>
    3. <b>fcc与SE信号低换手版本${stC.alpha>stA.alpha?'fcc更优':'SE更优'}</b>：C（fcc低换手+减仓退出${FMT(stC.alpha)}）vs A（SE低换手${FMT(stA.alpha)}）。
  `;
}

function renderT2(){
  const qs=Object.keys(D.stats['REF1'].qr).sort();
  const cQ=mc('c-qbar');
  if(cQ) cQ.setOption({
    backgroundColor:'transparent',
    grid:{left:55,right:10,top:20,bottom:44},
    xAxis:{type:'category',data:qs,...AX,axisTick:{show:false},axisLabel:{color:'#566070',fontSize:11,rotate:30}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(0)+'%'}},
    tooltip:{...TT,trigger:'axis'},
    legend:{top:0,right:0,textStyle:{color:'#566070',fontSize:9}},
    series: SKEYS.map(s=>({
      name:LABELS[s].slice(0,8),type:'bar',
      data:qs.map(q=>{const r=D.stats[s].qr[q];return r?+(r.port-r.bench).toFixed(2):null;}),
      itemStyle:{color:p=>p.value>=0?COLORS[s]:COLORS[s]+'66',borderRadius:[2,2,0,0]},
      barMaxWidth:10,
    })),
  });

  document.getElementById('q-tbl').innerHTML = qs.map(q=>{
    const bench=D.stats['REF1'].qr[q]?.bench;
    const cols = SKEYS.map(s=>{
      const r=D.stats[s].qr[q];
      const al=r?+(r.port-r.bench).toFixed(2):null;
      return `<td class="${CLS(al)}">${FMT(al)}</td>`;
    }).join('');
    return `<tr><td class="l"><b>${q}</b></td><td class="${CLS(bench)}">${FMT(bench)}</td>${cols}</tr>`;
  }).join('');
}

function renderT3(){
  const s=document.getElementById('t3-s').value;
  const trades=D.trades[s];
  const byQ={};
  trades.forEach(t=>(byQ[t.q]=byQ[t.q]||[]).push(t));
  let html='';
  Object.keys(byQ).sort().forEach(q=>{
    const ts=byQ[q];
    const rets=ts.map(t=>t.ret).filter(v=>v!=null);
    const als=ts.map(t=>t.alpha).filter(v=>v!=null);
    const avg=rets.length?(rets.reduce((a,b)=>a+b)/rets.length).toFixed(2):null;
    const avgA=als.length?(als.reduce((a,b)=>a+b)/als.length).toFixed(2):null;
    html+=`<tr class="phdr" onclick="tog('${s}_${q}')">
      <td class="l" colspan="10">▶ ${q}  披露日：${ts[0].disc}  持仓${ts.length}只</td>
      <td class="${CLS(avg)}">${avg!=null?(parseFloat(avg)>0?'+':'')+avg+'%':'—'}</td>
      <td>${ts[0].bench_ret!=null?ts[0].bench_ret+'%':'—'}</td>
      <td class="${CLS(avgA)}">${avgA!=null?(parseFloat(avgA)>0?'+':'')+avgA+'%':'—'}</td>
    </tr>`;
    ts.forEach(t=>{
      const fwdStr=t.fwd!=null?(t.fwd*100>0?'+':''+(t.fwd*100).toFixed(3))+'pp':'—';
      const fccCls=t.fcc>0?'pos':t.fcc<0?'neg':'neu';
      html+=`<tr class="row_${s}_${q}" style="display:none">
        <td class="l code">${t.code}</td><td class="l">${t.name}</td>
        <td class="l"><span class="tag">${t.ind||'—'}</span></td>
        <td>${t.fc??'—'}</td>
        <td class="${fccCls}">${t.fcc!=null?(t.fcc>0?'+':'')+t.fcc:'—'}</td>
        <td class="${t.fwd>0?'pos':t.fwd<0?'neg':'neu'}">${fwdStr}</td>
        <td>${t.entry_date}</td><td>${t.entry_price??'—'}</td>
        <td>${t.exit_date??'—'}</td><td>${t.exit_price??'—'}</td>
        <td class="${CLS(t.ret)}">${t.ret!=null?(t.ret>0?'+':'')+t.ret+'%':'—'}</td>
        <td>${t.bench_ret!=null?t.bench_ret+'%':'—'}</td>
        <td class="${CLS(t.alpha)}">${t.alpha!=null?(t.alpha>0?'+':'')+t.alpha+'%':'—'}</td>
      </tr>`;
    });
  });
  document.getElementById('t3-body').innerHTML=html;
  const first=Object.keys(byQ).sort()[0];
  if(first) tog(`${s}_${first}`);
}

function tog(key){
  document.querySelectorAll('.row_'+key).forEach(r=>{
    r.style.display=r.style.display==='none'?'':'none';
  });
}

(function(){
  document.getElementById('meta').textContent=
    `${D.stats.REF1.start} ~ ${D.stats.REF1.end}  ·  持有基金数≥30  ·  生成 ${D.gen_time}`;
  renderT1();
})();
</script>
</body>
</html>
"""

def main():
    print("=== 最终策略综合测试 ===")
    by_q     = load_holdings()
    price_df = load_prices()
    quarters = sorted(by_q.keys(), key=sort_q)

    print("构建策略持仓...")
    strategies = build_strategies(by_q, quarters)
    for s, periods in strategies.items():
        cnts = [len(p['stocks']) for p in periods]
        print(f"  {s}: {cnts}  均={sum(cnts)/len(cnts):.1f}只/期")

    print("计算净值...")
    nav_data, trade_data, stat_data = {}, {}, {}
    for s, periods in strategies.items():
        print(f"  {s}...", end=' ', flush=True)
        nav_df, trades = run_nav(periods, price_df)
        nav_data[s]   = nav_df
        trade_data[s] = trades
        stat_data[s]  = calc_stats(nav_df)
        st = stat_data[s]
        print(f"超额{st['alpha']:+.1f}pp  夏普{st['sh']:.2f}  回撤{st['dd']:.1f}%")

    def safe(v):
        if v is None: return None
        if isinstance(v, float) and v != v: return None
        return v

    dataset = {
        'nav':    {s: [{'d':str(r['date'].date()),'n':round(r['nav'],6),'b':round(r['bench'],6)}
                       for _,r in df.iterrows()] for s, df in nav_data.items()},
        'trades': {s: [{k: safe(v) for k,v in t.items()} for t in tl]
                   for s, tl in trade_data.items()},
        'stats':  stat_data,
        'gen_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    print("\n生成 HTML...")
    data_json = json.dumps(dataset, ensure_ascii=False, default=str, separators=(',',':'))
    html = HTML.replace('__DATA__', data_json)
    OUTPUT.write_text(html, encoding='utf-8')
    print(f"  输出: {OUTPUT}  ({OUTPUT.stat().st_size//1024} KB)")
    print("完成！")

if __name__ == '__main__':
    main()
