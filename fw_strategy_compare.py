#!/usr/bin/env python3
"""
fw（占基金持仓比重）衍生策略对比回测，fc≥30，等权，季度调仓
SA  fwd 环比增量前20
SB  低fw × fwd上升（低拥挤 + 仓位动量）
SC  fwd × fcc 双维度扩张（仓位占比 + 基金数双增）
SD  高fw + fwd下降（减仓预警：空头/规避端）
SE  行业相对低配 + fwd上升（行业内被低估）
REF S2参考线：fcc前20（上一轮最优策略）
"""

import json, warnings, sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = Path("workwork/research/策略研究/基金重仓看板")
sys.path.insert(0, str(Path.cwd()))
from workwork.research.策略研究.基金重仓看板.backtest_signals import (
    load_holdings, load_prices, DISCLOSURE_DATES, sort_q
)

OUTPUT = BASE / "fund_fw_compare.html"
FC_MIN = 30
TOP_N  = 20
BENCH  = '000300.SH'

# ═══════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════

def next_td(price_df, d):
    idx = price_df.index[price_df.index >= pd.Timestamp(d)]
    return idx[0] if len(idx) else None

def get_px(ser, d):
    sub = ser[ser.index >= pd.Timestamp(d)].dropna()
    return (sub.index[0], float(sub.iloc[0])) if len(sub) else (None, None)


# ═══════════════════════════════════════════════════════════════════════
# 构建每期持仓
# ═══════════════════════════════════════════════════════════════════════

def build_holdings(by_q, quarters):
    strats = {k: [] for k in ('SA','SB','SC','SD','SE','REF')}
    prev_map = {r['code']: r for r in by_q[quarters[0]]}

    for i, q in enumerate(quarters[1:], 1):
        disc     = DISCLOSURE_DATES.get(q)
        nxt_q    = quarters[i+1] if i+1 < len(quarters) else None
        nxt_disc = DISCLOSURE_DATES.get(nxt_q) if nxt_q else None
        cur_map  = {r['code']: r for r in by_q[q]}

        # ── 计算 fwd（fw 环比变动）──
        for code, r in cur_map.items():
            fw_cur  = r.get('fw')
            fw_prev = prev_map[code].get('fw') if code in prev_map else None
            if fw_cur is not None:
                r['_fwd'] = fw_cur - (fw_prev or 0.0)
            else:
                r['_fwd'] = None

        # ── 行业平均 fw（用于 SE）──
        ind_fw = {}
        for r in cur_map.values():
            if r.get('fw') is not None and r.get('ind'):
                ind_fw.setdefault(r['ind'], []).append(r['fw'])
        ind_avg = {ind: np.mean(vals) for ind, vals in ind_fw.items()}

        def ok(r):
            return r.get('fc') is not None and r['fc'] >= FC_MIN

        def fw_pctile(r, universe):
            """fw 在 universe 中的百分位（0~1）"""
            fw_vals = sorted([x['fw'] for x in universe if x.get('fw') is not None])
            if not fw_vals or r.get('fw') is None: return 0.5
            below = sum(1 for v in fw_vals if v < r['fw'])
            return below / len(fw_vals)

        valid  = [r for r in cur_map.values() if ok(r)]
        fw_pcts = {}
        fw_vals_sorted = sorted([r['fw'] for r in valid if r.get('fw') is not None])
        for r in valid:
            if r.get('fw') is not None and fw_vals_sorted:
                below = sum(1 for v in fw_vals_sorted if v < r['fw'])
                fw_pcts[r['code']] = below / len(fw_vals_sorted)

        # SA: fwd 前20（含新进入，fwd = fw_cur）
        sa = sorted([r for r in valid if r.get('_fwd') is not None],
                    key=lambda r: r['_fwd'], reverse=True)[:TOP_N]

        # SB: fw 下半区（pctile<=0.5）+ fwd>0，按 fwd 排序
        sb = sorted([r for r in valid
                     if r.get('_fwd') is not None and r['_fwd'] > 0
                     and fw_pcts.get(r['code'], 1) <= 0.5],
                    key=lambda r: r['_fwd'], reverse=True)[:TOP_N]

        # SC: fwd>0 AND fcc>0，按 fwd 排序
        sc = sorted([r for r in valid
                     if r.get('_fwd') is not None and r['_fwd'] > 0
                     and r.get('fcc') is not None and r['fcc'] > 0],
                    key=lambda r: r['_fwd'], reverse=True)[:TOP_N]

        # SD: fw 上四分位（pctile>0.75）+ fwd<0（规避端，做空逻辑）
        sd = sorted([r for r in valid
                     if r.get('_fwd') is not None and r['_fwd'] < 0
                     and fw_pcts.get(r['code'], 0) > 0.75],
                    key=lambda r: r['_fwd'])[:TOP_N]   # 最负的前20

        # SE: fw < 行业平均 + fwd>0，按 fwd 排序
        se = sorted([r for r in valid
                     if r.get('_fwd') is not None and r['_fwd'] > 0
                     and r.get('fw') is not None and r.get('ind')
                     and r['fw'] < ind_avg.get(r['ind'], r['fw'] + 1)],
                    key=lambda r: r['_fwd'], reverse=True)[:TOP_N]

        # REF: fcc 前20（上一轮最优 S2）
        ref = sorted([r for r in valid
                      if r.get('fcc') is not None and r['fcc'] > 0],
                     key=lambda r: r['fcc'], reverse=True)[:TOP_N]

        def to_list(sel):
            return [{'code': r['code'], 'name': r['name'],
                     'ind': r.get('ind', ''), 'fc': r['fc'],
                     'fw': r.get('fw'), 'fwd': round(r['_fwd'], 6) if r.get('_fwd') is not None else None,
                     'fcc': r.get('fcc'), 'pv': r.get('pv')}
                    for r in sel]

        meta = {'q': q, 'disc': disc, 'next_disc': nxt_disc}
        for key, sel in (('SA',sa),('SB',sb),('SC',sc),('SD',sd),('SE',se),('REF',ref)):
            strats[key].append({**meta, 'stocks': to_list(sel)})

        prev_map = cur_map

    return strats


# ═══════════════════════════════════════════════════════════════════════
# 日度净值
# ═══════════════════════════════════════════════════════════════════════

def run_nav(periods, price_df):
    bench_ser = price_df[BENCH]
    nav = 1.0; bench_nav = 1.0
    nav_rows = []; trades = []

    start = next_td(price_df, periods[0]['disc'])
    nav_rows.append({'date': start, 'nav': nav, 'bench': bench_nav})

    for period in periods:
        disc_ts = pd.Timestamp(period['disc'])
        end_ts  = pd.Timestamp(period['next_disc']) if period['next_disc'] else price_df.index[-1]
        stocks  = period['stocks']
        if not stocks: continue
        w = 1.0 / len(stocks)

        entry_ts = next_td(price_df, disc_ts)
        exit_ts  = next_td(price_df, end_ts)

        entry_px = {}
        for s in stocks:
            if s['code'] in price_df.columns:
                _, p = get_px(price_df[s['code']], entry_ts)
                entry_px[s['code']] = p

        prev_px    = dict(entry_px)
        prev_bench = float(bench_ser[bench_ser.index >= entry_ts].dropna().iloc[0])

        mask = (price_df.index > entry_ts) & (price_df.index <= exit_ts)
        for d in price_df.index[mask]:
            dr = []
            for s in stocks:
                c = s['code']
                if c not in price_df.columns or not prev_px.get(c): continue
                cur = price_df[c].get(d)
                if cur and not np.isnan(cur) and prev_px[c] != 0:
                    dr.append(float(cur)/prev_px[c] - 1)
                    prev_px[c] = float(cur)
                else:
                    dr.append(0.0)
            pr = np.mean(dr) if dr else 0.0
            bc = bench_ser.get(d)
            br = float(bc)/prev_bench - 1 if bc and not np.isnan(bc) and prev_bench else 0.0
            if bc and not np.isnan(bc): prev_bench = float(bc)
            nav       *= (1 + pr)
            bench_nav *= (1 + br)
            nav_rows.append({'date': d, 'nav': round(nav,6), 'bench': round(bench_nav,6)})

        for s in stocks:
            c  = s['code']
            ep = entry_px.get(c)
            if c not in price_df.columns or not ep: continue
            _, xp = get_px(price_df[c], exit_ts)
            _, bs = get_px(bench_ser, entry_ts)
            _, bx = get_px(bench_ser, exit_ts)
            ret   = round((xp/ep-1)*100, 3)           if xp else None
            bret  = round((bx/bs-1)*100, 3)           if bx and bs else None
            alpha = round(ret - bret, 3)               if ret is not None and bret is not None else None
            trades.append({'q': period['q'], 'disc': period['disc'],
                           'code': c, 'name': s['name'], 'ind': s['ind'],
                           'fc': s['fc'], 'fwd': s.get('fwd'), 'fcc': s.get('fcc'),
                           'entry_date': str(entry_ts.date()),
                           'entry_price': round(ep, 3),
                           'exit_date': str(exit_ts.date()) if exit_ts else None,
                           'exit_price': round(xp, 3) if xp else None,
                           'ret': ret, 'bench_ret': bret, 'alpha': alpha})

    df = pd.DataFrame(nav_rows).drop_duplicates('date').sort_values('date')
    return df, trades


# ═══════════════════════════════════════════════════════════════════════
# 统计
# ═══════════════════════════════════════════════════════════════════════

def calc_stats(nav_df):
    p = nav_df['nav'].values; b = nav_df['bench'].values
    dates = nav_df['date'].values
    n_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    tot = p[-1]/p[0] - 1; b_tot = b[-1]/b[0] - 1
    ann = (1+tot)**(365/n_days) - 1 if n_days > 0 else 0
    dr  = np.diff(p)/p[:-1]
    sh  = np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr) > 0 else 0
    dd  = float(np.min(p / np.maximum.accumulate(p) - 1))
    df2 = nav_df.copy()
    df2['Q'] = pd.to_datetime(df2['date']).dt.to_period('Q')
    qr = {}
    for qp, g in df2.groupby('Q'):
        qr[str(qp)] = {'port':  round((g['nav'].iloc[-1]/g['nav'].iloc[0]-1)*100, 2),
                       'bench': round((g['bench'].iloc[-1]/g['bench'].iloc[0]-1)*100, 2)}
    return {'tot': round(tot*100,2), 'b_tot': round(b_tot*100,2),
            'alpha': round((tot-b_tot)*100,2),
            'ann': round(ann*100,2), 'sh': round(sh,2),
            'dd': round(dd*100,2),
            'start': str(pd.Timestamp(dates[0]).date()),
            'end':   str(pd.Timestamp(dates[-1]).date()),
            'q_rets': qr}


# ═══════════════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fw策略对比回测</title>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root{
  --bg:#f0ece4;--card:#fff;--th:#f7f5f1;--alt:#faf8f5;
  --hdr:#0f1f3d;--bd:#ddd8d0;--bdt:#ccc8bf;
  --t1:#14213d;--t2:#566070;--t3:#96a0ae;
  --acc:#1e50a2;--acc2:#eef3fc;--pos:#c0392b;--neg:#166534;
  --sh:0 2px 8px rgba(20,33,61,.09),0 1px 3px rgba(20,33,61,.05);
  --r:8px;
  --ui:'Sora','PingFang SC','Microsoft YaHei',sans-serif;
  --mono:'JetBrains Mono','Courier New',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t1);font-family:var(--ui);font-size:13px;line-height:1.5}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-thumb{background:var(--bd);border-radius:4px}
.hdr{background:var(--hdr);padding:0 28px;height:56px;display:flex;align-items:center;
  justify-content:space-between;position:sticky;top:0;z-index:50;box-shadow:0 1px 0 rgba(255,255,255,.08)}
.brand{display:flex;align-items:center;gap:12px}
.icon{width:34px;height:34px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);
  border-radius:8px;display:flex;align-items:center;justify-content:center}
.hdr h1{font-size:15px;font-weight:700;color:#fff}
.hdr-sub{font-size:9px;color:rgba(255,255,255,.38);letter-spacing:1.2px;text-transform:uppercase}
.hdr-meta{color:rgba(255,255,255,.35);font-size:11px;font-family:var(--mono)}
.tabs{background:#fff;padding:0 28px;display:flex;border-bottom:1px solid var(--bd);
  position:sticky;top:56px;z-index:40;box-shadow:var(--sh)}
.tab{padding:13px 15px;cursor:pointer;border:none;background:none;color:var(--t3);
  font-size:12px;font-family:var(--ui);font-weight:500;border-bottom:2px solid transparent;
  transition:all .15s;white-space:nowrap;margin-bottom:-1px}
.tab:hover{color:var(--t2)}.tab.on{color:var(--acc);border-bottom-color:var(--acc);font-weight:700}
.pane{display:none;padding:22px 28px}.pane.on{display:block}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.mb{margin-bottom:14px}
.box{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);
  padding:16px 18px;box-shadow:var(--sh)}
.box-title{font-size:9px;color:var(--t3);margin-bottom:12px;font-weight:700;
  text-transform:uppercase;letter-spacing:.7px}
.tbl-wrap{overflow-x:auto;border-radius:var(--r);border:1px solid var(--bd);
  box-shadow:var(--sh);background:var(--card)}
table{width:100%;border-collapse:collapse}
thead{position:sticky;top:0;z-index:10}
th{background:var(--th);color:var(--t3);font-weight:700;font-size:10px;text-transform:uppercase;
  letter-spacing:.5px;padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--bdt)}
th.l,td.l{text-align:left}
td{padding:7px 12px;border-top:1px solid #ede9e3;text-align:right;white-space:nowrap;
  font-size:12px;font-family:var(--mono)}
td.l{font-family:var(--ui);font-size:12px}
tr:nth-child(even) td{background:var(--alt)}
tr:hover td{background:var(--acc2)!important}
.code{color:var(--acc);font-size:11px}
.tag{display:inline-block;background:var(--acc2);color:var(--acc);padding:1px 6px;
  border-radius:4px;font-size:10px;font-weight:600;font-family:var(--ui)}
.badge{display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;
  font-size:11px;font-weight:600}
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}.neu{color:var(--t3)}
.phdr{cursor:pointer;user-select:none}
.phdr td{background:#f0f4fb!important;color:var(--acc);font-weight:700;
  font-family:var(--ui);padding:9px 12px!important}
.phdr:hover td{background:#e4ecf9!important}
.ctrl{display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.lbl{color:var(--t3);font-size:10px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;white-space:nowrap}
select{background:var(--card);border:1px solid var(--bd);color:var(--t1);padding:5px 10px;
  border-radius:6px;font-size:12px;font-family:var(--ui);outline:none;height:30px}
select:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(30,80,162,.1)}
.hint{color:var(--t3);font-size:11px;margin-top:10px;line-height:1.7}
.warn{background:#fffbeb;border:1px solid #fcd34d;border-radius:6px;
  padding:10px 14px;font-size:12px;color:#92400e;margin-top:10px}
</style>
</head>
<body>
<div class="hdr">
  <div class="brand">
    <div class="icon"><svg width="16" height="16" fill="none">
      <path d="M8 2L4 6h3v8h2V6h3L8 2z" fill="white" opacity=".9"/>
      <path d="M3 12h10v2H3v-2z" fill="white" opacity=".5"/>
    </svg></div>
    <div><h1>基金持仓比重信号策略对比</h1>
    <div class="hdr-sub">Fund Weight Signal Comparison · fc≥30 · Equal Weight</div></div>
  </div>
  <span class="hdr-meta" id="meta"></span>
</div>

<div class="tabs">
  <button class="tab on" onclick="go('t1')">净值对比曲线</button>
  <button class="tab" onclick="go('t2')">综合指标汇总</button>
  <button class="tab" onclick="go('t3')">逐季度表现</button>
  <button class="tab" onclick="go('t4')">调仓明细</button>
</div>

<!-- TAB1 净值曲线 -->
<div id="t1" class="pane on">
  <div class="box mb">
    <div class="box-title">各策略净值曲线对比（基准：沪深300）</div>
    <div id="c-nav" style="height:420px"></div>
  </div>
  <div class="g2">
    <div class="box">
      <div class="box-title">回撤曲线（多头策略）</div>
      <div id="c-dd" style="height:220px"></div>
    </div>
    <div class="box">
      <div class="box-title">持仓股数（各策略每期实际持仓）</div>
      <div id="c-cnt" style="height:220px"></div>
    </div>
  </div>
  <p class="hint">
    「规避端」策略（高拥挤度+持仓比重下降）为空头逻辑：净值越低说明这批股票跑输市场越多，规避效果越好。其余均为多头组合。基准：沪深300（000300.SH）。
  </p>
</div>

<!-- TAB2 指标汇总 -->
<div id="t2" class="pane">
  <div class="tbl-wrap mb">
    <table>
      <thead><tr>
        <th class="l">策略</th><th class="l">逻辑</th>
        <th>累计收益</th><th>超额</th><th>年化</th>
        <th>夏普</th><th>最大回撤</th><th>每期平均持仓</th>
      </tr></thead>
      <tbody id="stat-body"></tbody>
    </table>
  </div>
  <div class="box mb">
    <div class="box-title">多头策略综合评分雷达图（规避端策略已排除）</div>
    <div id="c-radar" style="height:380px"></div>
  </div>
  <div class="warn" id="sd-note"></div>
</div>

<!-- TAB3 逐季度 -->
<div id="t3" class="pane">
  <div class="ctrl">
    <span class="lbl">策略</span>
    <select id="t3-s" onchange="renderT3()"></select>
  </div>
  <div class="box mb">
    <div class="box-title">逐季度收益 vs 沪深300</div>
    <div id="c-qbar" style="height:320px"></div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th class="l">季度</th><th>组合收益</th><th>基准</th><th>超额</th><th>持仓数</th>
      </tr></thead>
      <tbody id="t3-body"></tbody>
    </table>
  </div>
</div>

<!-- TAB4 调仓明细 -->
<div id="t4" class="pane">
  <div class="ctrl">
    <span class="lbl">策略</span>
    <select id="t4-s" onchange="renderT4()"></select>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th class="l">季度/代码</th><th class="l">名称</th><th class="l">行业</th>
        <th>fc</th><th>fwd(pp)</th><th>fcc</th>
        <th>入场日</th><th>入场价</th><th>出场日</th><th>出场价</th>
        <th>个股收益</th><th>基准</th><th>超额</th>
      </tr></thead>
      <tbody id="t4-body"></tbody>
    </table>
  </div>
</div>

<script>
const D = __DATA__;
const SKEYS  = ['SA','SB','SC','SD','SE','REF'];
const COLORS = {SA:'#c0392b',SB:'#1e50a2',SC:'#7c3aed',SD:'#96a0ae',SE:'#d97706',REF:'#0891b2'};
const LABELS = {
  SA:'持仓比重环比增量前20',
  SB:'低拥挤度+持仓比重上升',
  SC:'持仓比重与基金数双扩张',
  SD:'高拥挤度+持仓比重下降（规避）',
  SE:'行业相对低配+持仓比重上升',
  REF:'基金数增量前20（上轮最优参考）',
};
const DESCS  = {
  SA:'fc≥30，以持仓比重(fw)季度环比变动最大的前20只',
  SB:'fc≥30，fw处于全市场下半区（尚未拥挤）且fw本季上升，按环比变动排序前20',
  SC:'fc≥30，fw环比上升且持有基金数也增加，双维度同步扩张，按fw变动排序前20',
  SD:'fc≥30，fw处于上四分位（高度拥挤）且本季fw下降，规避/空头端',
  SE:'fc≥30，fw低于同行业平均水平且本季fw上升，捕捉行业内部资金轮动方向',
  REF:'fc≥30，持有基金数增量最大前20（上一轮对比测试中的最优策略，作为参考基线）',
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
  const idx=['t1','t2','t3','t4'].indexOf(id);
  document.querySelectorAll('.tab')[idx].classList.add('on');
  ({t2:renderT2,t3:renderT3,t4:renderT4}[id]||function(){})();
}

// ── TAB1 ──────────────────────────────────────────────────────────────
function renderT1(){
  const dates = D.nav['REF'].map(r=>r.d);
  const cNav = mc('c-nav');
  if(cNav){
    const series = SKEYS.map(s=>({
      name:LABELS[s], type:'line', symbol:'none', smooth:.1,
      data:D.nav[s].map(r=>r.n),
      lineStyle:{color:COLORS[s], width:s==='REF'?1.5:2.2,
                 type:s==='SD'?'dotted':s==='REF'?'dashed':'solid'},
    }));
    series.push({name:'沪深300',type:'line',symbol:'none',smooth:.1,
      data:D.nav['REF'].map(r=>r.b),
      lineStyle:{color:'#ccc8bf',width:1.2,type:'dashed'}});
    cNav.setOption({
      backgroundColor:'transparent',
      grid:{left:60,right:140,top:20,bottom:36},
      xAxis:{type:'category',data:dates,...AX,axisTick:{show:false},
             axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
      yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(2)}},
      tooltip:{...TT,trigger:'axis',formatter:p=>{
        let s=`<b>${p[0].name}</b><br>`;
        p.forEach(x=>s+=`<span style="color:${x.color}">●</span> ${x.seriesName}: ${x.value.toFixed(4)}<br>`);
        return s;
      }},
      legend:{orient:'vertical',right:0,top:'middle',
              textStyle:{color:'#566070',fontSize:10,fontFamily:"'Sora',sans-serif"}},
      series,
    });
  }

  // Drawdown（多头策略）
  const longKeys = ['SA','SB','SC','SE','REF'];
  const cDd = mc('c-dd');
  if(cDd) cDd.setOption({
    backgroundColor:'transparent',
    grid:{left:55,right:10,top:10,bottom:36},
    xAxis:{type:'category',data:dates,...AX,axisTick:{show:false},
           axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(1)+'%'}},
    tooltip:{...TT,trigger:'axis'},
    series: longKeys.map(s=>{
      const nav=D.nav[s].map(r=>r.n); let mx=nav[0];
      const dd=nav.map(v=>{mx=Math.max(mx,v);return+(v/mx-1).toFixed(6)*100;});
      return {name:LABELS[s],type:'line',symbol:'none',smooth:.1,
              data:dd,lineStyle:{color:COLORS[s],width:1.5}};
    }),
    legend:{bottom:0,textStyle:{color:'#566070',fontSize:10}},
  });

  // Holdings count
  const qs = Object.keys(D.cnt['SA']).sort();
  const cCnt = mc('c-cnt');
  if(cCnt) cCnt.setOption({
    backgroundColor:'transparent',
    grid:{left:36,right:10,top:20,bottom:40},
    xAxis:{type:'category',data:qs,...AX,axisTick:{show:false},
           axisLabel:{color:'#566070',fontSize:10,rotate:30}},
    yAxis:{type:'value',min:0,max:22,...AX,axisLine:{show:false}},
    tooltip:{...TT,trigger:'axis'},
    legend:{top:0,right:0,textStyle:{color:'#566070',fontSize:10}},
    series: SKEYS.map(s=>({
      name:LABELS[s],type:'bar',
      data:qs.map(q=>D.cnt[s][q]??0),
      itemStyle:{color:COLORS[s],borderRadius:[2,2,0,0]},barMaxWidth:8,
    })),
  });
}

// ── TAB2 ──────────────────────────────────────────────────────────────
function renderT2(){
  const avgCnt = s => {
    const vals = Object.values(D.cnt[s]);
    return vals.length ? (vals.reduce((a,b)=>a+b)/vals.length).toFixed(1) : '—';
  };
  document.getElementById('stat-body').innerHTML = [
    ...SKEYS.map(s=>{
      const st=D.stats[s];
      const isSd = s==='SD';
      return `<tr>
        <td class="l"><span class="badge" style="background:${COLORS[s]}22;color:${COLORS[s]}">${s}</span></td>
        <td class="l" style="font-size:11px;color:var(--t2);max-width:200px;white-space:normal">${DESCS[s]}</td>
        <td class="${isSd?'neu':CLS(st.tot)}">${FMT(st.tot)}</td>
        <td class="${isSd?'neu':CLS(st.alpha)}">${FMT(st.alpha)}</td>
        <td class="${isSd?'neu':CLS(st.ann)}">${FMT(st.ann)}</td>
        <td>${st.sh.toFixed(2)}</td>
        <td class="${CLS(st.dd)}">${FMT(st.dd)}</td>
        <td>${avgCnt(s)}</td>
      </tr>`;
    }),
    `<tr style="border-top:2px solid var(--bdt)">
      <td class="l" colspan="2" style="font-family:var(--ui);color:var(--t3);font-size:11px">沪深300（基准）</td>
      <td>${FMT(D.stats.REF.b_tot)}</td>
      <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
    </tr>`,
  ].join('');

  // Radar（多头策略）
  const longKeys = ['SA','SB','SC','SE','REF'];
  const radar_data = longKeys.map(s=>{
    const st=D.stats[s];
    return {name:LABELS[s],value:[
      Math.max(0,st.tot+10),
      Math.max(0,st.alpha+10),
      Math.max(0,st.ann+10),
      Math.max(0,(st.sh+0.5)*30),
      Math.max(0,40+st.dd),
    ]};
  });
  const cR = mc('c-radar');
  if(cR) cR.setOption({
    backgroundColor:'transparent',
    color: longKeys.map(s=>COLORS[s]),
    legend:{bottom:0,textStyle:{color:'#566070',fontSize:11}},
    radar:{
      indicator:[{name:'累计收益',max:120},{name:'超额收益',max:80},
                 {name:'年化收益',max:80},{name:'夏普比率',max:60},{name:'抗回撤',max:50}],
      shape:'polygon',radius:'58%',
      axisName:{color:'#566070',fontSize:11},
      splitLine:{lineStyle:{color:'#e8e3db'}},
      axisLine:{lineStyle:{color:'#ccc8bf'}},
      splitArea:{show:false},
    },
    series:[{type:'radar',data:radar_data.map((d,i)=>({
      name:d.name,value:d.value,symbol:'circle',symbolSize:5,
      lineStyle:{color:longKeys.map(s=>COLORS[s])[i],width:2},
      areaStyle:{color:longKeys.map(s=>COLORS[s])[i],opacity:.12},
    }))}],
  });

  // SD 规避端说明
  const sdSt = D.stats['SD'];
  const note = sdSt.tot < 0
    ? `SD 规避端净值累计 ${FMT(sdSt.tot)}，相对基准 ${FMT(sdSt.alpha)}。净值下跌说明这批"高fw+仓位下降"的股票确实跑输市场，规避信号有效。`
    : `SD 规避端净值累计 ${FMT(sdSt.tot)}，相对基准 ${FMT(sdSt.alpha)}。净值未明显下跌，规避信号在本期数据中效果有限。`;
  document.getElementById('sd-note').textContent = '⚠ SD规避端注释：' + note;
}

// ── TAB3 ──────────────────────────────────────────────────────────────
function renderT3(){
  const s  = document.getElementById('t3-s').value;
  const qr = D.stats[s].q_rets;
  const qs = Object.keys(qr).sort();
  const cQ = mc('c-qbar');
  if(cQ) cQ.setOption({
    backgroundColor:'transparent',
    grid:{left:55,right:10,top:20,bottom:44},
    xAxis:{type:'category',data:qs,...AX,axisTick:{show:false},
           axisLabel:{color:'#566070',fontSize:11,rotate:30}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(0)+'%'}},
    tooltip:{...TT,trigger:'axis'},
    legend:{top:0,right:0,textStyle:{color:'#566070',fontSize:11}},
    series:[
      {name:LABELS[s],type:'bar',data:qs.map(q=>qr[q].port),barGap:'5%',barMaxWidth:20,
       itemStyle:{borderRadius:[3,3,0,0],
                  color:p=>p.value>=0?COLORS[s]:COLORS[s]+'88'}},
      {name:'沪深300',type:'bar',data:qs.map(q=>qr[q].bench),barMaxWidth:20,
       itemStyle:{color:'#ccc8bf',borderRadius:[3,3,0,0]}},
    ],
  });
  const cnt = D.cnt[s];
  document.getElementById('t3-body').innerHTML = qs.map(q=>{
    const pr=qr[q].port, br=qr[q].bench, al=+(pr-br).toFixed(2);
    // 季度key对应cnt key
    const cntKey = Object.keys(cnt).find(k=>q.includes(k.replace(/Q(\d+) (\d+)/,'$2Q$1').slice(-4).replace(/Q/,'Q')))
                   || Object.keys(cnt)[Object.keys(D.stats[s].q_rets).sort().indexOf(q)];
    return `<tr>
      <td class="l"><b>${q}</b></td>
      <td class="${CLS(pr)}">${FMT(pr)}</td>
      <td class="${CLS(br)}">${FMT(br)}</td>
      <td class="${CLS(al)}">${FMT(al)}</td>
      <td>${Object.values(cnt)[Object.keys(D.stats[s].q_rets).sort().indexOf(q)]??'—'}</td>
    </tr>`;
  }).join('');
}

// ── TAB4 ──────────────────────────────────────────────────────────────
function renderT4(){
  const s = document.getElementById('t4-s').value;
  const trades = D.trades[s];
  const byQ = {};
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
      const fwdStr = t.fwd!=null?(t.fwd*100).toFixed(3)+'pp':'—';
      html+=`<tr class="row_${s}_${q}" style="display:none">
        <td class="l code">${t.code}</td>
        <td class="l">${t.name}</td>
        <td class="l"><span class="tag">${t.ind||'—'}</span></td>
        <td>${t.fc??'—'}</td>
        <td class="${t.fwd>0?'pos':t.fwd<0?'neg':'neu'}">${fwdStr}</td>
        <td class="${t.fcc>0?'pos':t.fcc<0?'neg':'neu'}">${t.fcc!=null?(t.fcc>0?'+':'')+t.fcc:'—'}</td>
        <td>${t.entry_date}</td><td>${t.entry_price??'—'}</td>
        <td>${t.exit_date??'—'}</td><td>${t.exit_price??'—'}</td>
        <td class="${CLS(t.ret)}">${t.ret!=null?(t.ret>0?'+':'')+t.ret+'%':'—'}</td>
        <td>${t.bench_ret!=null?t.bench_ret+'%':'—'}</td>
        <td class="${CLS(t.alpha)}">${t.alpha!=null?(t.alpha>0?'+':'')+t.alpha+'%':'—'}</td>
      </tr>`;
    });
  });
  document.getElementById('t4-body').innerHTML=html;
  const first=Object.keys(byQ).sort()[0];
  if(first) tog(`${s}_${first}`);
}

function tog(key){
  document.querySelectorAll('.row_'+key).forEach(r=>{
    r.style.display=r.style.display==='none'?'':'none';
  });
}

// ── Init ───────────────────────────────────────────────────────────────
(function(){
  document.getElementById('meta').textContent=
    `${D.stats.REF.start} ~ ${D.stats.REF.end}  ·  fc≥${D.fc_min}  ·  生成 ${D.gen_time}`;
  // 填充策略选择器
  ['t3-s','t4-s'].forEach(id=>{
    const sel=document.getElementById(id);
    SKEYS.forEach(s=>{
      const o=document.createElement('option');o.value=s;o.text=LABELS[s];sel.appendChild(o);
    });
  });
  renderT1();
})();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    from datetime import datetime
    print("=== fw衍生策略对比回测 ===")
    print("加载数据...")
    by_q     = load_holdings()
    price_df = load_prices()
    quarters = sorted(by_q.keys(), key=sort_q)

    print("构建各期持仓...")
    sp = build_holdings(by_q, quarters)
    for s, periods in sp.items():
        cnts = [len(p['stocks']) for p in periods]
        print(f"  {s}: {cnts}  均={sum(cnts)/len(cnts):.1f}只/期")

    print("计算净值曲线...")
    nav_data, trade_data, stat_data, cnt_data = {}, {}, {}, {}
    for s, periods in sp.items():
        print(f"  {s}...", end=' ', flush=True)
        nav_df, trades = run_nav(periods, price_df)
        nav_data[s]   = nav_df
        trade_data[s] = trades
        stat_data[s]  = calc_stats(nav_df)
        cnt_data[s]   = {p['q']: len(p['stocks']) for p in periods}
        st = stat_data[s]
        print(f"累计{st['tot']:+.1f}%  超额{st['alpha']:+.1f}%  夏普{st['sh']:.2f}  回撤{st['dd']:.1f}%")

    def safe(v):
        if v is None: return None
        if isinstance(v, float) and v != v: return None
        return v

    nav_json = {s: [{'d':str(r['date'].date()),'n':round(r['nav'],6),'b':round(r['bench'],6)}
                    for _,r in df.iterrows()]
                for s, df in nav_data.items()}
    trades_json = {s: [{k:safe(v) for k,v in t.items()} for t in tl]
                   for s, tl in trade_data.items()}

    dataset = {'nav': nav_json, 'trades': trades_json, 'stats': stat_data,
               'cnt': cnt_data, 'fc_min': FC_MIN,
               'gen_time': datetime.now().strftime('%Y-%m-%d %H:%M')}

    print("\n生成 HTML...")
    data_json = json.dumps(dataset, ensure_ascii=False, default=str, separators=(',',':'))
    html = HTML.replace('__DATA__', data_json)
    OUTPUT.write_text(html, encoding='utf-8')
    print(f"  输出: {OUTPUT}  ({OUTPUT.stat().st_size//1024} KB)")
    print("完成！")


if __name__ == '__main__':
    main()
