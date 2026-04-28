#!/usr/bin/env python3
"""
四策略对比回测（fc ≥ 30 硬门槛，等权，季度调仓）
S1 全持仓抱团前20   S2 加仓增量前20
S3 新进入fc≥30      S4 新晋重仓（跨越fc=30门槛）
输出：fund_strategy_compare.html
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

OUTPUT  = BASE / "fund_strategy_compare.html"
FC_MIN  = 30
TOP_N   = 20
BENCH   = '000300.SH'

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
    """返回 {strategy: [{q, disc, next_disc, stocks:[{code,name,...}]}]}"""
    strats = {k: [] for k in ('S1','S2','S3','S4')}
    prev_map = {r['code']: r for r in by_q[quarters[0]]}   # 以1Q24初始化

    for i, q in enumerate(quarters[1:], 1):
        disc     = DISCLOSURE_DATES.get(q)
        nxt_q    = quarters[i + 1] if i + 1 < len(quarters) else None
        nxt_disc = DISCLOSURE_DATES.get(nxt_q) if nxt_q else None
        cur_map  = {r['code']: r for r in by_q[q]}

        def ok(r):   # fc 硬门槛
            return r['fc'] is not None and r['fc'] >= FC_MIN

        # S1: 全持仓 fc≥30, 按 fc 前20
        s1 = sorted([r for r in cur_map.values() if ok(r)],
                    key=lambda r: r['fc'], reverse=True)[:TOP_N]

        # S2: fc≥30 且 fcc>0, 按 fcc 前20
        s2 = sorted([r for r in cur_map.values()
                     if ok(r) and r.get('fcc') and r['fcc'] > 0],
                    key=lambda r: r['fcc'], reverse=True)[:TOP_N]

        # S3: 新进入（上季度不在）且 fc≥30, 按 fc 前20
        s3 = sorted([r for c, r in cur_map.items()
                     if c not in prev_map and ok(r)],
                    key=lambda r: r['fc'], reverse=True)[:TOP_N]

        # S4: 新晋重仓 = fc≥30 且 (新进入 OR 上季度fc<FC_MIN)
        def is_new_jiin(code, r):
            if code not in prev_map:
                return True
            prev_fc = prev_map[code].get('fc')
            return prev_fc is None or prev_fc < FC_MIN
        s4 = sorted([r for c, r in cur_map.items()
                     if ok(r) and is_new_jiin(c, r)],
                    key=lambda r: r['fc'], reverse=True)[:TOP_N]

        meta = {'q': q, 'disc': disc, 'next_disc': nxt_disc}
        for key, sel in (('S1',s1),('S2',s2),('S3',s3),('S4',s4)):
            stocks = [{'code': r['code'], 'name': r['name'],
                       'ind':  r.get('ind',''), 'fc': r['fc'],
                       'fcc': r.get('fcc'), 'fw': r.get('fw'), 'pv': r.get('pv')}
                      for r in sel]
            strats[key].append({**meta, 'stocks': stocks})

        prev_map = cur_map

    return strats


# ═══════════════════════════════════════════════════════════════════════
# 日度净值
# ═══════════════════════════════════════════════════════════════════════

def run_nav(periods, price_df):
    """返回 (nav_df, trades_list)"""
    bench_ser = price_df[BENCH]
    nav, bench_nav = 1.0, 1.0
    nav_rows = []
    trades   = []

    start = next_td(price_df, periods[0]['disc'])
    nav_rows.append({'date': start, 'nav': nav, 'bench': bench_nav})

    for period in periods:
        disc_ts = pd.Timestamp(period['disc'])
        end_ts  = (pd.Timestamp(period['next_disc'])
                   if period['next_disc'] else price_df.index[-1])

        stocks = period['stocks']
        if not stocks:
            continue
        n = len(stocks)
        w = 1.0 / n

        entry_ts = next_td(price_df, disc_ts)
        exit_ts  = next_td(price_df, end_ts)

        # 入场价
        entry_px = {}
        for s in stocks:
            if s['code'] in price_df.columns:
                _, p = get_px(price_df[s['code']], entry_ts)
                entry_px[s['code']] = p

        prev_px    = dict(entry_px)
        prev_bench = float(bench_ser[bench_ser.index >= entry_ts].dropna().iloc[0])

        # 逐日
        mask = (price_df.index > entry_ts) & (price_df.index <= exit_ts)
        for d in price_df.index[mask]:
            day_rets = []
            for s in stocks:
                c = s['code']
                if c not in price_df.columns or prev_px.get(c) is None: continue
                cur = price_df[c].get(d)
                if cur and not np.isnan(cur) and prev_px[c] != 0:
                    day_rets.append(float(cur) / prev_px[c] - 1)
                    prev_px[c] = float(cur)
                else:
                    day_rets.append(0.0)

            pr  = np.mean(day_rets) if day_rets else 0.0
            bc  = bench_ser.get(d)
            br  = float(bc)/prev_bench - 1 if bc and not np.isnan(bc) and prev_bench else 0.0
            if bc and not np.isnan(bc): prev_bench = float(bc)

            nav       *= (1 + pr)
            bench_nav *= (1 + br)
            nav_rows.append({'date': d, 'nav': round(nav, 6),
                             'bench': round(bench_nav, 6)})

        # 交易记录
        for s in stocks:
            c   = s['code']
            ep  = entry_px.get(c)
            if c not in price_df.columns or ep is None: continue
            _, xp = get_px(price_df[c], exit_ts)
            _, bs = get_px(bench_ser, entry_ts)
            _, bx = get_px(bench_ser, exit_ts)
            ret    = round((xp/ep - 1)*100, 3)   if xp and ep else None
            b_ret  = round((bx/bs - 1)*100, 3)   if bx and bs else None
            alpha  = round(ret - b_ret, 3)         if ret is not None and b_ret is not None else None
            trades.append({
                'q': period['q'], 'disc': period['disc'],
                'code': c, 'name': s['name'], 'ind': s['ind'],
                'fc': s['fc'], 'fcc': s.get('fcc'),
                'entry_date': str(entry_ts.date()),
                'entry_price': round(ep, 3),
                'exit_date': str(exit_ts.date()) if exit_ts else None,
                'exit_price': round(xp, 3) if xp else None,
                'ret': ret, 'bench_ret': b_ret, 'alpha': alpha,
            })

    df = pd.DataFrame(nav_rows).drop_duplicates('date').sort_values('date')
    return df, trades


# ═══════════════════════════════════════════════════════════════════════
# 统计指标
# ═══════════════════════════════════════════════════════════════════════

def stats(nav_df):
    p = nav_df['nav'].values
    b = nav_df['bench'].values
    dates = nav_df['date'].values

    n_days   = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    tot      = p[-1]/p[0] - 1
    ann      = (1+tot)**(365/n_days) - 1 if n_days > 0 else 0
    dr       = np.diff(p)/p[:-1]
    sharpe   = np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr) > 0 else 0
    dd       = float(np.min(p / np.maximum.accumulate(p) - 1))
    b_tot    = b[-1]/b[0] - 1

    # 季度拆分
    df2 = nav_df.copy()
    df2['Q'] = pd.to_datetime(df2['date']).dt.to_period('Q')
    q_rets = {}
    for qp, g in df2.groupby('Q'):
        q_rets[str(qp)] = {
            'port':  round((g['nav'].iloc[-1]/g['nav'].iloc[0]-1)*100, 2),
            'bench': round((g['bench'].iloc[-1]/g['bench'].iloc[0]-1)*100, 2),
        }
    return {
        'tot': round(tot*100, 2), 'b_tot': round(b_tot*100, 2),
        'alpha': round((tot - b_tot)*100, 2),
        'ann': round(ann*100, 2), 'sharpe': round(sharpe, 2),
        'max_dd': round(dd*100, 2),
        'start': str(pd.Timestamp(dates[0]).date()),
        'end':   str(pd.Timestamp(dates[-1]).date()),
        'q_rets': q_rets,
    }


def holdings_summary(strat_periods):
    """每期实际持仓数"""
    return {p['q']: len(p['stocks']) for p in strat_periods}


# ═══════════════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>四策略对比回测 · fc≥30</title>
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
.tab{padding:14px 16px;cursor:pointer;border:none;background:none;color:var(--t3);
  font-size:12px;font-family:var(--ui);font-weight:500;border-bottom:2px solid transparent;
  transition:all .15s;white-space:nowrap;margin-bottom:-1px}
.tab:hover{color:var(--t2)}.tab.on{color:var(--acc);border-bottom-color:var(--acc);font-weight:700}
.pane{display:none;padding:22px 28px;min-height:calc(100vh - 109px)}.pane.on{display:block}
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
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}.neu{color:var(--t3)}
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;
  font-size:11px;font-weight:600;font-family:var(--ui)}
.phdr{cursor:pointer;user-select:none}
.phdr td{background:#f0f4fb!important;color:var(--acc);font-weight:700;font-family:var(--ui);padding:9px 12px!important}
.phdr:hover td{background:#e4ecf9!important}
.ctrl{display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.lbl{color:var(--t3);font-size:10px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;white-space:nowrap}
select{background:var(--card);border:1px solid var(--bd);color:var(--t1);padding:5px 10px;
  border-radius:6px;font-size:12px;font-family:var(--ui);outline:none;height:30px}
select:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(30,80,162,.1)}
.hint{color:var(--t3);font-size:11px;margin-top:10px;line-height:1.7}
</style>
</head>
<body>
<div class="hdr">
  <div class="brand">
    <div class="icon"><svg width="16" height="16" fill="none">
      <path d="M1 15h3V8H1v7zm4-4h3v11H5V11zm4-3h3v14H9V8zm4-4h2v18h-2V4z" fill="white" opacity=".85"/>
    </svg></div>
    <div><h1>四策略对比回测 · fc ≥ 30</h1>
    <div class="hdr-sub">Strategy Comparison · Equal Weight · vs CSI300</div></div>
  </div>
  <span class="hdr-meta" id="meta"></span>
</div>

<div class="tabs">
  <button class="tab on" onclick="go('t1')">净值对比</button>
  <button class="tab" onclick="go('t2')">指标汇总</button>
  <button class="tab" onclick="go('t3')">逐季度</button>
  <button class="tab" onclick="go('t4')">调仓明细</button>
</div>

<!-- TAB1 净值曲线 -->
<div id="t1" class="pane on">
  <div class="box mb">
    <div class="box-title">四策略净值曲线对比（基准：沪深300）</div>
    <div id="c-nav" style="height:420px"></div>
  </div>
  <div class="g2">
    <div class="box">
      <div class="box-title">回撤曲线</div>
      <div id="c-dd" style="height:220px"></div>
    </div>
    <div class="box">
      <div class="box-title">每期实际持仓股数（各策略）</div>
      <div id="c-cnt" style="height:220px"></div>
    </div>
  </div>
</div>

<!-- TAB2 指标汇总 -->
<div id="t2" class="pane">
  <div class="tbl-wrap mb">
    <table>
      <thead><tr>
        <th class="l">策略</th><th class="l">定义</th>
        <th>累计收益</th><th>超额</th><th>年化</th>
        <th>夏普</th><th>最大回撤</th>
      </tr></thead>
      <tbody id="stat-body"></tbody>
    </table>
  </div>
  <div class="box mb">
    <div class="box-title">综合评分雷达图</div>
    <div id="c-radar" style="height:360px"></div>
  </div>
</div>

<!-- TAB3 逐季度 -->
<div id="t3" class="pane">
  <div class="ctrl">
    <span class="lbl">策略</span>
    <select id="t3-strat" onchange="renderT3()">
      <option value="S1">S1 抱团前20</option>
      <option value="S2">S2 增量前20</option>
      <option value="S3">S3 新进入</option>
      <option value="S4">S4 新晋重仓</option>
    </select>
  </div>
  <div class="box mb">
    <div class="box-title">逐季度收益 vs 沪深300</div>
    <div id="c-qbar" style="height:320px"></div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th class="l">季度</th><th>组合收益</th><th>基准收益</th><th>超额</th><th>持仓数</th>
      </tr></thead>
      <tbody id="t3-body"></tbody>
    </table>
  </div>
</div>

<!-- TAB4 调仓明细 -->
<div id="t4" class="pane">
  <div class="ctrl">
    <span class="lbl">策略</span>
    <select id="t4-strat" onchange="renderT4()">
      <option value="S1">S1 抱团前20</option>
      <option value="S2">S2 增量前20</option>
      <option value="S3">S3 新进入</option>
      <option value="S4">S4 新晋重仓</option>
    </select>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th class="l">季度/代码</th><th class="l">名称</th><th class="l">行业</th>
        <th>fc</th><th>fcc</th><th>入场日</th><th>入场价</th>
        <th>出场日</th><th>出场价</th><th>个股收益</th><th>基准</th><th>超额</th>
      </tr></thead>
      <tbody id="t4-body"></tbody>
    </table>
  </div>
</div>

<script>
const D = __DATA__;
const COLORS = {S1:'#c0392b', S2:'#1e50a2', S3:'#d97706', S4:'#7c3aed', bench:'#96a0ae'};
const LABELS = {S1:'S1 抱团前20', S2:'S2 增量前20', S3:'S3 新进入', S4:'S4 新晋重仓'};
const DESCS  = {
  S1:'全持仓 fc≥30，按fc降序取前20',
  S2:'fc≥30且fcc>0，按fcc降序取前20',
  S3:'新进入持仓且fc≥30，按fc降序取前20',
  S4:'新晋达到fc≥30门槛（新进入或上季fc<30），按fc降序取前20',
};
const AX = {
  axisLine:{lineStyle:{color:'#ccc8bf'}},
  axisLabel:{color:'#96a0ae',fontSize:10,fontFamily:"'JetBrains Mono',monospace"},
  splitLine:{lineStyle:{color:'#e8e3db',type:'dashed'}},
};
const TT = {backgroundColor:'#fff',borderColor:'#ddd8d0',borderWidth:1,
            textStyle:{color:'#14213d',fontSize:11}};
const FMT  = v => v==null?'—':(v>0?'+':'')+v.toFixed(2)+'%';
const FMTN = v => v==null?'—':v.toFixed(2);
const CLS  = v => v==null?'neu':v>0?'pos':v<0?'neg':'neu';

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

// ── TAB 1 ──────────────────────────────────────────────────────────────
function renderT1(){
  const strats = ['S1','S2','S3','S4'];

  // NAV chart
  const allDates = D.nav.S1.map(r=>r.d);
  const cNav = mc('c-nav');
  if(cNav){
    const series = strats.map(s=>({
      name:LABELS[s], type:'line', symbol:'none', smooth:.1,
      data:D.nav[s].map(r=>r.n),
      lineStyle:{color:COLORS[s],width:s==='S1'?2.5:2},
      emphasis:{lineStyle:{width:3}},
    }));
    series.push({
      name:'沪深300', type:'line', symbol:'none', smooth:.1,
      data:D.nav.S1.map(r=>r.b),
      lineStyle:{color:COLORS.bench,width:1.5,type:'dashed'},
    });
    cNav.setOption({
      backgroundColor:'transparent',
      grid:{left:60,right:130,top:20,bottom:36},
      xAxis:{type:'category',data:allDates,...AX,axisTick:{show:false},
             axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
      yAxis:{type:'value',...AX,axisLine:{show:false},
             axisLabel:{formatter:v=>v.toFixed(2)}},
      tooltip:{...TT,trigger:'axis',formatter:p=>{
        let s=`<b>${p[0].name}</b><br>`;
        p.forEach(x=>s+=`<span style="color:${x.color}">●</span> ${x.seriesName}: ${x.value.toFixed(4)}<br>`);
        return s;
      }},
      legend:{orient:'vertical',right:0,top:'middle',
              textStyle:{color:'#566070',fontSize:11,fontFamily:"'Sora',sans-serif"}},
      series,
    });
  }

  // Drawdown
  const cDd = mc('c-dd');
  if(cDd){
    const ddSeries = strats.map(s=>{
      const nav = D.nav[s].map(r=>r.n);
      let mx=nav[0];
      const dd = nav.map(v=>{mx=Math.max(mx,v);return+(v/mx-1).toFixed(6)*100;});
      return {name:LABELS[s],type:'line',symbol:'none',smooth:.1,
              data:dd,lineStyle:{color:COLORS[s],width:1.5},
              areaStyle:{color:COLORS[s]+'18'}};
    });
    cDd.setOption({
      backgroundColor:'transparent',
      grid:{left:55,right:10,top:10,bottom:36},
      xAxis:{type:'category',data:allDates,...AX,axisTick:{show:false},
             axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
      yAxis:{type:'value',...AX,axisLine:{show:false},
             axisLabel:{formatter:v=>v.toFixed(1)+'%'}},
      tooltip:{...TT,trigger:'axis'},
      series:ddSeries,
    });
  }

  // Holdings count bar
  const qs = Object.keys(D.cnt.S1).sort();
  const cCnt = mc('c-cnt');
  if(cCnt) cCnt.setOption({
    backgroundColor:'transparent',
    grid:{left:36,right:10,top:20,bottom:36},
    xAxis:{type:'category',data:qs,...AX,axisTick:{show:false},
           axisLabel:{color:'#566070',fontSize:10,rotate:30}},
    yAxis:{type:'value',min:0,max:22,...AX,axisLine:{show:false}},
    tooltip:{...TT,trigger:'axis'},
    legend:{top:0,right:0,textStyle:{color:'#566070',fontSize:10}},
    series: strats.map(s=>({
      name:LABELS[s], type:'bar',
      data:qs.map(q=>D.cnt[s][q]??0),
      itemStyle:{color:COLORS[s],borderRadius:[2,2,0,0]},
      barMaxWidth:10,
    })),
  });
}

// ── TAB 2 ──────────────────────────────────────────────────────────────
function renderT2(){
  const strats = ['S1','S2','S3','S4'];
  // Table
  document.getElementById('stat-body').innerHTML = [
    ...strats.map(s=>{
      const st = D.stats[s];
      return `<tr>
        <td class="l"><span class="badge" style="background:${COLORS[s]}22;color:${COLORS[s]}">${s}</span></td>
        <td class="l" style="font-size:11px;color:var(--t2);max-width:240px;white-space:normal">${DESCS[s]}</td>
        <td class="${CLS(st.tot)}">${FMT(st.tot)}</td>
        <td class="${CLS(st.alpha)}">${FMT(st.alpha)}</td>
        <td class="${CLS(st.ann)}">${FMT(st.ann)}</td>
        <td>${FMTN(st.sharpe)}</td>
        <td class="${CLS(st.max_dd)}">${FMT(st.max_dd)}</td>
      </tr>`;
    }),
    `<tr style="border-top:2px solid var(--bdt)">
      <td class="l" colspan="2" style="font-family:var(--ui);color:var(--t3);font-size:11px">沪深300（基准）</td>
      <td>${FMT(D.stats.S1.b_tot)}</td>
      <td>—</td><td>—</td><td>—</td><td>—</td>
    </tr>`
  ].join('');

  // Radar
  const radar_data = strats.map(s=>{
    const st = D.stats[s];
    return {name:LABELS[s],value:[
      Math.max(0, st.tot + 20),
      Math.max(0, st.alpha + 20),
      Math.max(0, st.ann + 20),
      Math.max(0, (st.sharpe + 1) * 20),
      Math.max(0, 40 + st.max_dd),
    ]};
  });
  const cR = mc('c-radar');
  if(cR) cR.setOption({
    backgroundColor:'transparent',
    color: strats.map(s=>COLORS[s]),
    legend:{bottom:0,textStyle:{color:'#566070',fontSize:11}},
    radar:{
      indicator:[
        {name:'累计收益',max:150},{name:'超额收益',max:100},
        {name:'年化收益',max:100},{name:'夏普比率',max:60},
        {name:'抗回撤',max:50},
      ],
      shape:'polygon',radius:'62%',
      axisName:{color:'#566070',fontSize:11},
      splitLine:{lineStyle:{color:'#e8e3db'}},
      axisLine:{lineStyle:{color:'#ccc8bf'}},
      splitArea:{show:false},
    },
    series:[{type:'radar',data:radar_data.map((d,i)=>({
      name:d.name,value:d.value,
      symbol:'circle',symbolSize:5,
      lineStyle:{color:strats.map(s=>COLORS[s])[i],width:2},
      areaStyle:{color:strats.map(s=>COLORS[s])[i],opacity:.12},
    }))}],
  });
}

// ── TAB 3 ──────────────────────────────────────────────────────────────
function renderT3(){
  const s  = document.getElementById('t3-strat').value;
  const qr = D.stats[s].q_rets;
  const qs = Object.keys(qr).sort();
  const pv = qs.map(q=>qr[q].port);
  const bv = qs.map(q=>qr[q].bench);

  const cQ = mc('c-qbar');
  if(cQ) cQ.setOption({
    backgroundColor:'transparent',
    grid:{left:55,right:10,top:20,bottom:40},
    xAxis:{type:'category',data:qs,...AX,axisTick:{show:false},
           axisLabel:{color:'#566070',fontSize:11,rotate:30}},
    yAxis:{type:'value',...AX,axisLine:{show:false},
           axisLabel:{formatter:v=>v.toFixed(0)+'%'}},
    tooltip:{...TT,trigger:'axis'},
    legend:{top:0,right:0,textStyle:{color:'#566070',fontSize:11}},
    series:[
      {name:LABELS[s],type:'bar',data:pv,barGap:'5%',barMaxWidth:20,
       itemStyle:{color:p=>p.value>=0?COLORS[s]:COLORS[s]+'88',borderRadius:[3,3,0,0]}},
      {name:'沪深300',type:'bar',data:bv,barMaxWidth:20,
       itemStyle:{color:'#ccc8bf',borderRadius:[3,3,0,0]}},
    ],
  });

  const cnt = D.cnt[s];
  document.getElementById('t3-body').innerHTML = qs.map(q=>{
    const pr = qr[q].port, br = qr[q].bench, al = +(pr-br).toFixed(2);
    return `<tr>
      <td class="l"><b>${q}</b></td>
      <td class="${CLS(pr)}">${FMT(pr)}</td>
      <td class="${CLS(br)}">${FMT(br)}</td>
      <td class="${CLS(al)}">${FMT(al)}</td>
      <td>${cnt[q.replace('Q','Q')]??'—'}</td>
    </tr>`;
  }).join('');
}

// ── TAB 4 ──────────────────────────────────────────────────────────────
function renderT4(){
  const s = document.getElementById('t4-strat').value;
  const trades = D.trades[s];
  // Group by period
  const byQ = {};
  trades.forEach(t=>(byQ[t.q]=byQ[t.q]||[]).push(t));
  let html = '';
  Object.keys(byQ).sort().forEach(q=>{
    const ts = byQ[q];
    const rets = ts.map(t=>t.ret).filter(v=>v!=null);
    const als  = ts.map(t=>t.alpha).filter(v=>v!=null);
    const avg  = rets.length ? (rets.reduce((a,b)=>a+b)/rets.length).toFixed(2) : null;
    const avgA = als.length  ? (als.reduce((a,b)=>a+b)/als.length).toFixed(2)  : null;
    html += `<tr class="phdr" onclick="tog('${s}_${q}')">
      <td class="l" colspan="9">▶ ${q}  披露日：${ts[0].disc}  持仓${ts.length}只</td>
      <td class="${CLS(avg)}">${avg!=null?(parseFloat(avg)>0?'+':'')+avg+'%':'—'}</td>
      <td>${ts[0].bench_ret!=null?ts[0].bench_ret+'%':'—'}</td>
      <td class="${CLS(avgA)}">${avgA!=null?(parseFloat(avgA)>0?'+':'')+avgA+'%':'—'}</td>
    </tr>`;
    ts.forEach(t=>{
      html += `<tr class="row_${s}_${q}" style="display:none">
        <td class="l code">${t.code}</td>
        <td class="l">${t.name}</td>
        <td class="l"><span class="tag">${t.ind||'—'}</span></td>
        <td>${t.fc??'—'}</td>
        <td class="${t.fcc>0?'pos':t.fcc<0?'neg':'neu'}">${t.fcc!=null?(t.fcc>0?'+':'')+t.fcc:'—'}</td>
        <td>${t.entry_date}</td>
        <td>${t.entry_price??'—'}</td>
        <td>${t.exit_date??'—'}</td>
        <td>${t.exit_price??'—'}</td>
        <td class="${CLS(t.ret)}">${t.ret!=null?(t.ret>0?'+':'')+t.ret+'%':'—'}</td>
        <td>${t.bench_ret!=null?t.bench_ret+'%':'—'}</td>
        <td class="${CLS(t.alpha)}">${t.alpha!=null?(t.alpha>0?'+':'')+t.alpha+'%':'—'}</td>
      </tr>`;
    });
  });
  document.getElementById('t4-body').innerHTML = html;
  // Open first
  const first = Object.keys(byQ).sort()[0];
  if(first) tog(`${s}_${first}`);
}

function tog(key){
  document.querySelectorAll('.row_'+key).forEach(r=>{
    r.style.display = r.style.display==='none'?'':'none';
  });
}

// ── Init ───────────────────────────────────────────────────────────────
(function(){
  document.getElementById('meta').textContent =
    `${D.stats.S1.start} ~ ${D.stats.S1.end}  ·  fc≥${D.fc_min}  ·  生成 ${D.gen_time}`;
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

    print("=== 四策略对比回测 ===")
    print("加载数据...")
    by_q     = load_holdings()
    price_df = load_prices()
    quarters = sorted(by_q.keys(), key=sort_q)

    print("构建各期持仓...")
    strat_periods = build_holdings(by_q, quarters)

    # 持仓数汇总
    cnt = {s: holdings_summary(strat_periods[s]) for s in strat_periods}
    for s, periods in strat_periods.items():
        print(f"  {s}: ", end='')
        for p in periods:
            print(f"{p['q']}({len(p['stocks'])})", end=' ')
        print()

    print("计算净值曲线...")
    nav_data, trade_data, stat_data = {}, {}, {}
    for s, periods in strat_periods.items():
        print(f"  {s}...", end=' ', flush=True)
        nav_df, trades = run_nav(periods, price_df)
        nav_data[s]   = nav_df
        trade_data[s] = trades
        stat_data[s]  = stats(nav_df)
        st = stat_data[s]
        print(f"累计{st['tot']:+.1f}%  超额{st['alpha']:+.1f}%  夏普{st['sharpe']:.2f}  回撤{st['max_dd']:.1f}%")

    print("\n构建数据集...")
    def safe(v):
        if v is None: return None
        if isinstance(v, float) and v != v: return None
        return v

    nav_json = {}
    for s, df in nav_data.items():
        nav_json[s] = [{'d': str(r['date'].date()),
                        'n': round(r['nav'], 6),
                        'b': round(r['bench'], 6)}
                       for _, r in df.iterrows()]

    trades_json = {}
    for s, tlist in trade_data.items():
        trades_json[s] = [{k: safe(v) for k, v in t.items()} for t in tlist]

    dataset = {
        'nav':      nav_json,
        'trades':   trades_json,
        'stats':    stat_data,
        'cnt':      cnt,
        'fc_min':   FC_MIN,
        'gen_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    print("生成 HTML...")
    data_json = json.dumps(dataset, ensure_ascii=False, default=str, separators=(',', ':'))
    html = HTML.replace('__DATA__', data_json)
    OUTPUT.write_text(html, encoding='utf-8')
    print(f"  输出: {OUTPUT}  ({OUTPUT.stat().st_size//1024} KB)")
    print("完成！")


if __name__ == '__main__':
    main()
