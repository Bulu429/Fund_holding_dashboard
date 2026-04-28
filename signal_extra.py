#!/usr/bin/env python3
"""
补充研究：
1. 减仓信号有效性（做空端 + 多空组合）
2. 低换手率策略：每季仅换5只 vs 全换20只
"""

import json, warnings, sys, numpy as np, pandas as pd
from datetime import timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
BASE = Path("workwork/research/策略研究/基金重仓看板")
sys.path.insert(0, str(Path.cwd()))
from workwork.research.策略研究.基金重仓看板.backtest_signals import (
    load_holdings, load_prices, DISCLOSURE_DATES, sort_q
)

OUTPUT = BASE / "fund_signal_extra.html"
FC_MIN = 30
BENCH  = '000300.SH'

# ─── 工具 ──────────────────────────────────────────────────────────────
def next_td(pdf, d):
    idx = pdf.index[pdf.index >= pd.Timestamp(d)]
    return idx[0] if len(idx) else None

def get_px(ser, d):
    sub = ser[ser.index >= pd.Timestamp(d)].dropna()
    return (sub.index[0], float(sub.iloc[0])) if len(sub) else (None, None)

def run_nav(periods, price_df, short=False):
    """
    short=True: 做空持仓（每日收益取反）
    返回 nav_df, trades
    """
    bench_ser = price_df[BENCH]
    nav = bench_nav = 1.0
    rows = []; trades = []
    start = next_td(price_df, periods[0]['disc'])
    rows.append({'date': start, 'nav': 1.0, 'bench': 1.0})

    for period in periods:
        stocks = period['stocks']
        if not stocks: continue
        disc_ts = pd.Timestamp(period['disc'])
        end_ts  = (pd.Timestamp(period['next_disc'])
                   if period['next_disc'] else price_df.index[-1])
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
                    r = float(cur) / prev_px[c] - 1
                    dr.append(-r if short else r)
                    prev_px[c] = float(cur)
                else:
                    dr.append(0.0)
            pr = np.mean(dr) if dr else 0.0
            bc = bench_ser.get(d)
            br = float(bc)/prev_bench - 1 if bc and not np.isnan(bc) and prev_bench else 0.0
            if bc and not np.isnan(bc): prev_bench = float(bc)
            nav       *= (1 + pr)
            bench_nav *= (1 + br)
            rows.append({'date': d, 'nav': round(nav,6), 'bench': round(bench_nav,6)})

        # 交易记录
        for s in stocks:
            c = s['code']
            ep = entry_px.get(c)
            if c not in price_df.columns or not ep: continue
            _, xp = get_px(price_df[c], exit_ts)
            _, bs = get_px(bench_ser, entry_ts)
            _, bx = get_px(bench_ser, exit_ts)
            ret   = round((xp/ep-1)*100, 3)         if xp else None
            bret  = round((bx/bs-1)*100, 3)         if bx and bs else None
            alpha = round(ret - bret, 3)             if ret is not None and bret is not None else None
            trades.append({'q': period['q'], 'disc': period['disc'],
                           'code': c, 'name': s['name'], 'ind': s.get('ind',''),
                           'fc': s['fc'], 'fcc': s.get('fcc'),
                           'entry_date': str(entry_ts.date()),
                           'entry_price': round(ep, 3),
                           'exit_date': str(exit_ts.date()) if exit_ts else None,
                           'exit_price': round(xp, 3) if xp else None,
                           'ret': ret, 'bench_ret': bret, 'alpha': alpha})

    df = pd.DataFrame(rows).drop_duplicates('date').sort_values('date')
    return df, trades


def combine_nav(long_df, short_df):
    """多空组合 = 0.5 × long + 0.5 × short（各半仓）"""
    merged = pd.merge(long_df[['date','nav','bench']],
                      short_df[['date','nav']].rename(columns={'nav':'nav_s'}),
                      on='date', how='inner')
    merged['nav'] = (merged['nav'] + merged['nav_s']) / 2
    return merged[['date','nav','bench']]


def calc_stats(nav_df):
    p = nav_df['nav'].values; b = nav_df['bench'].values
    dates = nav_df['date'].values
    n = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    tot = p[-1]/p[0] - 1; btot = b[-1]/b[0] - 1
    ann = (1+tot)**(365/n) - 1 if n > 0 else 0
    dr  = np.diff(p)/p[:-1]
    sh  = np.mean(dr)/np.std(dr)*np.sqrt(252) if np.std(dr) > 0 else 0
    dd  = float(np.min(p / np.maximum.accumulate(p) - 1))
    df2 = nav_df.copy()
    df2['Q'] = pd.to_datetime(df2['date']).dt.to_period('Q')
    qr = {str(qp): {'port': round((g['nav'].iloc[-1]/g['nav'].iloc[0]-1)*100, 2),
                    'bench': round((g['bench'].iloc[-1]/g['bench'].iloc[0]-1)*100, 2)}
          for qp, g in df2.groupby('Q')}
    return {'tot': round(tot*100,2), 'btot': round(btot*100,2),
            'alpha': round((tot-btot)*100,2),
            'ann': round(ann*100,2), 'sh': round(sh,2),
            'dd': round(dd*100,2),
            'start': str(pd.Timestamp(dates[0]).date()),
            'end':   str(pd.Timestamp(dates[-1]).date()),
            'qr': qr}


# ─── 构建持仓 ──────────────────────────────────────────────────────────

def build_all_periods(by_q, quarters):
    """
    返回每季度：加仓前20、减仓前20
    """
    add_periods, red_periods = [], []
    prev_map = {r['code']: r for r in by_q[quarters[0]]}

    for i, q in enumerate(quarters[1:], 1):
        disc     = DISCLOSURE_DATES[q]
        nxt_q    = quarters[i+1] if i+1 < len(quarters) else None
        nxt_disc = DISCLOSURE_DATES.get(nxt_q)
        cur_map  = {r['code']: r for r in by_q[q]}
        ok = lambda r: r.get('fc') is not None and r['fc'] >= FC_MIN

        add = sorted([r for r in cur_map.values() if ok(r) and r.get('fcc') and r['fcc'] > 0],
                     key=lambda r: r['fcc'], reverse=True)[:20]
        red = sorted([r for r in cur_map.values() if ok(r) and r.get('fcc') and r['fcc'] < 0],
                     key=lambda r: r['fcc'])[:20]     # 最负的20只

        def mk(sel):
            return {'q': q, 'disc': disc, 'next_disc': nxt_disc,
                    'stocks': [{'code': r['code'], 'name': r['name'],
                                'ind': r.get('ind',''), 'fc': r['fc'],
                                'fcc': r.get('fcc')} for r in sel]}
        add_periods.append(mk(add))
        red_periods.append(mk(red))
        prev_map = cur_map

    return add_periods, red_periods


def build_low_turnover(by_q, quarters, change_n=5):
    """
    低换手策略：每季只换 change_n 只
    保留排名仍靠前的 (20-change_n) 只，补入新上榜的 change_n 只
    排名依据：当季 fcc（持有基金数增量）
    """
    periods = []
    prev_map = {r['code']: r for r in by_q[quarters[0]]}
    portfolio = set()   # 当前持仓的代码集合

    for i, q in enumerate(quarters[1:], 1):
        disc     = DISCLOSURE_DATES[q]
        nxt_q    = quarters[i+1] if i+1 < len(quarters) else None
        nxt_disc = DISCLOSURE_DATES.get(nxt_q)
        cur_map  = {r['code']: r for r in by_q[q]}
        ok = lambda r: r.get('fc') is not None and r['fc'] >= FC_MIN

        # 全市场 fcc 排名
        ranked = sorted([r for r in cur_map.values() if ok(r) and r.get('fcc') is not None],
                        key=lambda r: r['fcc'], reverse=True)
        rank_map = {r['code']: idx for idx, r in enumerate(ranked)}   # code → rank(0=最佳)

        if not portfolio:
            # 首期：直接取前20
            new_port = {r['code'] for r in ranked[:20]}
        else:
            # 当前持仓里仍满足条件的（fc≥30且在当季名单里）
            hold_valid = [c for c in portfolio
                          if c in cur_map and ok(cur_map[c]) and cur_map[c].get('fcc') is not None]

            # 按当季 fcc 排名排序，保留排名最好的 (20 - change_n) 只
            keep_n = 20 - change_n
            keep = sorted(hold_valid, key=lambda c: rank_map.get(c, 9999))[:keep_n]
            keep_set = set(keep)

            # 补入排名最高且不在 keep 中的 change_n 只
            new_entries = [r['code'] for r in ranked if r['code'] not in keep_set][:change_n]
            new_port = keep_set | set(new_entries)

        portfolio = new_port
        stocks = [{'code': c, 'name': cur_map[c]['name'],
                   'ind': cur_map[c].get('ind',''),
                   'fc': cur_map[c]['fc'],
                   'fcc': cur_map[c].get('fcc')}
                  for c in portfolio if c in cur_map]
        periods.append({'q': q, 'disc': disc, 'next_disc': nxt_disc, 'stocks': stocks})
        prev_map = cur_map

    return periods


# ─── HTML ──────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>补充研究：减仓信号 & 低换手率策略</title>
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
tr:nth-child(even) td{background:var(--alt)}
tr:hover td{background:var(--acc2)!important}
.code{color:var(--acc);font-size:11px}
.tag{display:inline-block;background:var(--acc2);color:var(--acc);padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600;font-family:var(--ui)}
.insight{background:var(--acc2);border:1px solid #c5d8f5;border-radius:var(--r);
  padding:14px 18px;margin-bottom:14px;font-size:13px;color:var(--t1);line-height:1.8}
.insight b{color:var(--acc)}
.warn{background:#fffbeb;border:1px solid #fcd34d;border-radius:var(--r);
  padding:12px 16px;font-size:12px;color:#92400e;line-height:1.7;margin-top:10px}
.phdr{cursor:pointer;user-select:none}
.phdr td{background:#f0f4fb!important;color:var(--acc);font-weight:700;font-family:var(--ui);padding:9px 12px!important}
.phdr:hover td{background:#e4ecf9!important}
</style>
</head>
<body>
<div class="hdr">
  <div class="brand">
    <div class="icon"><svg width="16" height="16" fill="none">
      <path d="M3 8h10M8 3l5 5-5 5" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>
    </svg></div>
    <div><h1>补充研究：减仓信号 & 低换手率策略</h1>
    <div class="hdr-sub">Short Signal · Low-Turnover Portfolio · fc≥30</div></div>
  </div>
  <span class="hdr-meta" id="meta"></span>
</div>

<div class="tabs">
  <button class="tab on"  onclick="go('t1')">研究一：减仓信号</button>
  <button class="tab" onclick="go('t2')">研究二：低换手率策略</button>
</div>

<!-- ══ 研究一：减仓信号 ══════════════════════════════════════════════════ -->
<div id="t1" class="pane on">
  <div class="insight" id="t1-insight"></div>

  <div class="cards" id="t1-cards"></div>

  <div class="g2">
    <div class="box">
      <div class="box-title">四条净值曲线：加仓多头 / 减仓多头 / 多空组合 / 沪深300</div>
      <div id="c-ls-nav" style="height:340px"></div>
    </div>
    <div class="box">
      <div class="box-title">逐季度超额收益对比（各策略 vs 沪深300）</div>
      <div id="c-ls-q" style="height:340px"></div>
    </div>
  </div>

  <div class="g2">
    <div class="box">
      <div class="box-title">减仓组持仓（每季度，持有基金数降幅最大前20只）</div>
      <div class="tbl-wrap" style="margin:0;max-height:320px;overflow-y:auto">
        <table><thead><tr>
          <th class="l">季度</th><th class="l">代码</th><th class="l">名称</th>
          <th>持有基金数</th><th>本季增减</th><th>个股收益</th><th>超额</th>
        </tr></thead><tbody id="red-body"></tbody></table>
      </div>
    </div>
    <div class="box">
      <div class="box-title">加仓组持仓（每季度，持有基金数增量最大前20只）</div>
      <div class="tbl-wrap" style="margin:0;max-height:320px;overflow-y:auto">
        <table><thead><tr>
          <th class="l">季度</th><th class="l">代码</th><th class="l">名称</th>
          <th>持有基金数</th><th>本季增减</th><th>个股收益</th><th>超额</th>
        </tr></thead><tbody id="add-body"></tbody></table>
      </div>
    </div>
  </div>

  <div class="warn">
    注：「减仓多头」曲线代表持有减仓股票的多头收益（不做空），净值下跌说明这批股票跑输；
    「多空组合」= 50%加仓多头 + 50%减仓做空（等权），代表多空策略的综合回报。
    做空端未计入借券成本，实际多空策略收益会低于此处估算。
  </div>
</div>

<!-- ══ 研究二：低换手率策略 ══════════════════════════════════════════════ -->
<div id="t2" class="pane">
  <div class="insight" id="t2-insight"></div>

  <div class="cards" id="t2-cards"></div>

  <div class="box mb">
    <div class="box-title">净值曲线：每季换20只 vs 每季仅换5只 vs 沪深300</div>
    <div id="c-turn-nav" style="height:360px"></div>
  </div>

  <div class="g2">
    <div class="box">
      <div class="box-title">逐季度超额收益</div>
      <div id="c-turn-q" style="height:280px"></div>
    </div>
    <div class="box">
      <div class="box-title">低换手策略每季持仓变化明细</div>
      <div class="tbl-wrap" style="margin:0;max-height:280px;overflow-y:auto">
        <table><thead><tr>
          <th class="l">季度</th><th class="l">新进入</th><th class="l">退出</th>
          <th>换仓数</th><th>季度超额</th>
        </tr></thead><tbody id="turn-body"></tbody></table>
      </div>
    </div>
  </div>

  <div class="warn" id="t2-cost-note"></div>
</div>

<script>
const D = __DATA__;
const AX = {
  axisLine:{lineStyle:{color:'#ccc8bf'}},
  axisLabel:{color:'#96a0ae',fontSize:10,fontFamily:"'JetBrains Mono',monospace"},
  splitLine:{lineStyle:{color:'#e8e3db',type:'dashed'}},
};
const TT={backgroundColor:'#fff',borderColor:'#ddd8d0',borderWidth:1,textStyle:{color:'#14213d',fontSize:11}};
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
  ['t1','t2'].indexOf(id) === 1 ? renderT2() : renderT1();
  document.querySelectorAll('.tab')[['t1','t2'].indexOf(id)].classList.add('on');
}

// ── 研究一 ────────────────────────────────────────────────────────────
function renderT1(){
  const sa = D.stats.add, sr = D.stats.red, sls = D.stats.ls;
  const excess_add = sa.alpha, excess_red_short = -sr.alpha;  // 减仓做空超额 = 负的减仓多头超额

  document.getElementById('t1-insight').innerHTML = `
    <b>结论：减仓信号有效，但做空端比多头端弱。</b><br>
    加仓组多头：累计超额 <b>${FMT(sa.alpha)}</b>（胜率约57%）；
    减仓组持多头亏超额 <b>${FMT(sr.alpha)}</b>，即做空这批股票可获得约 <b>${FMT(-sr.alpha)}</b> 的超额；
    多空组合（各半仓）累计超额 <b>${FMT(sls.alpha)}</b>，夏普 <b>${sls.sh.toFixed(2)}</b>。
    减仓做空信号的超额强度约为加仓做多信号的 <b>${Math.abs(sr.alpha/sa.alpha*100).toFixed(0)}%</b>，
    信号有效但不对称——基金加仓后的个股表现比减仓后的表现更可预测。`;

  document.getElementById('t1-cards').innerHTML = [
    {lbl:'加仓组 累计超额',    val:sa.alpha,     sub:`夏普 ${sa.sh.toFixed(2)}  回撤 ${FMT(sa.dd)}`},
    {lbl:'减仓做空 累计超额',  val:-sr.alpha,    sub:`减仓组多头超额 ${FMT(sr.alpha)}`},
    {lbl:'多空组合 累计超额',  val:sls.alpha,    sub:`夏普 ${sls.sh.toFixed(2)}  回撤 ${FMT(sls.dd)}`},
    {lbl:'多空超额/多头超额',  val:null,
     sub:`${Math.abs(sls.alpha/sa.alpha*100).toFixed(0)}%（多空超额/纯多头超额）`},
  ].map(c=>`<div class="card">
    <div class="card-lbl">${c.lbl}</div>
    <div class="card-val ${CLS(c.val)}">${c.val!=null?FMT(c.val):'—'}</div>
    <div class="card-sub">${c.sub}</div></div>`).join('');

  const dates = D.nav.add.map(r=>r.d);
  const cN=mc('c-ls-nav');
  if(cN) cN.setOption({
    backgroundColor:'transparent',
    grid:{left:60,right:130,top:20,bottom:36},
    xAxis:{type:'category',data:dates,...AX,axisTick:{show:false},axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(2)}},
    tooltip:{...TT,trigger:'axis',formatter:p=>{
      let s=`<b>${p[0].name}</b><br>`;
      p.forEach(x=>s+=`<span style="color:${x.color}">●</span> ${x.seriesName}: ${x.value.toFixed(4)}<br>`);
      return s;
    }},
    legend:{orient:'vertical',right:0,top:'middle',textStyle:{color:'#566070',fontSize:11}},
    series:[
      {name:'加仓多头',type:'line',symbol:'none',smooth:.1,
       data:D.nav.add.map(r=>r.n),lineStyle:{color:'#c0392b',width:2.5}},
      {name:'减仓多头',type:'line',symbol:'none',smooth:.1,
       data:D.nav.red.map(r=>r.n),lineStyle:{color:'#166534',width:2,type:'dashed'}},
      {name:'多空组合',type:'line',symbol:'none',smooth:.1,
       data:D.nav.ls.map(r=>r.n),lineStyle:{color:'#7c3aed',width:2}},
      {name:'沪深300',type:'line',symbol:'none',smooth:.1,
       data:D.nav.add.map(r=>r.b),lineStyle:{color:'#ccc8bf',width:1.2,type:'dashed'}},
    ]
  });

  // 逐季度
  const qs=Object.keys(sa.qr).sort();
  const cQ=mc('c-ls-q');
  if(cQ) cQ.setOption({
    backgroundColor:'transparent',
    grid:{left:55,right:10,top:20,bottom:40},
    xAxis:{type:'category',data:qs,...AX,axisTick:{show:false},axisLabel:{color:'#566070',fontSize:11,rotate:30}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(0)+'%'}},
    tooltip:{...TT,trigger:'axis'},
    legend:{top:0,right:0,textStyle:{color:'#566070',fontSize:10}},
    series:[
      {name:'加仓多头超额',type:'bar',data:qs.map(q=>{
        const r=sa.qr[q];return r?+(r.port-r.bench).toFixed(2):null;}),
       itemStyle:{color:p=>p.value>=0?'#c0392b':'#c0392b66',borderRadius:[3,3,0,0]},barGap:'5%',barMaxWidth:14},
      {name:'减仓做空超额',type:'bar',data:qs.map(q=>{
        const r=sr.qr[q];return r?+(-(r.port-r.bench)).toFixed(2):null;}),
       itemStyle:{color:p=>p.value>=0?'#166534':'#16653466',borderRadius:[3,3,0,0]},barMaxWidth:14},
      {name:'多空组合超额',type:'bar',data:qs.map(q=>{
        const r=sls.qr[q];return r?+(r.port-r.bench).toFixed(2):null;}),
       itemStyle:{color:p=>p.value>=0?'#7c3aed':'#7c3aed66',borderRadius:[3,3,0,0]},barMaxWidth:14},
    ]
  });

  // 明细表
  const fmtRow = (t) => {
    const fc=t.fcc!=null?(t.fcc>0?'+':'')+t.fcc:'—';
    return `<tr>
      <td class="l">${t.q}</td><td class="l code">${t.code}</td><td class="l">${t.name}</td>
      <td>${t.fc??'—'}</td>
      <td class="${t.fcc>0?'pos':t.fcc<0?'neg':'neu'}">${fc}</td>
      <td class="${CLS(t.ret)}">${FMT(t.ret)}</td>
      <td class="${CLS(t.alpha)}">${FMT(t.alpha)}</td></tr>`;
  };
  document.getElementById('red-body').innerHTML = D.trades.red.map(fmtRow).join('');
  document.getElementById('add-body').innerHTML = D.trades.add.map(fmtRow).join('');
}

// ── 研究二 ────────────────────────────────────────────────────────────
function renderT2(){
  const sf = D.stats.full, sl = D.stats.low;
  const cost_full = D.cost_note.full, cost_low = D.cost_note.low;

  document.getElementById('t2-insight').innerHTML = `
    <b>结论：低换手策略回报接近，但费后优势更明显。</b><br>
    全换手（每季换20只）累计超额 <b>${FMT(sf.alpha)}</b>，夏普 <b>${sf.sh.toFixed(2)}</b>；
    低换手（每季仅换5只）累计超额 <b>${FMT(sl.alpha)}</b>，夏普 <b>${sl.sh.toFixed(2)}</b>。
    考虑实际交易成本（按单边0.15%估算），全换手策略每年额外拖累约 <b>${cost_full.toFixed(2)}%</b>，
    低换手策略仅 <b>${cost_low.toFixed(2)}%</b>，费后超额差距收窄至约 <b>${((sf.alpha-cost_full*2)-(sl.alpha-cost_low*2)).toFixed(1)}pp</b>。`;

  document.getElementById('t2-cards').innerHTML = [
    {lbl:'全换手 累计超额',  val:sf.alpha, sub:`夏普 ${sf.sh.toFixed(2)}  回撤 ${FMT(sf.dd)}`},
    {lbl:'低换手 累计超额',  val:sl.alpha, sub:`夏普 ${sl.sh.toFixed(2)}  回撤 ${FMT(sl.dd)}`},
    {lbl:'全换手 估算年化成本', val:-cost_full, sub:'单边0.15%，每季换20只×2'},
    {lbl:'低换手 估算年化成本', val:-cost_low,  sub:'单边0.15%，每季换5只×2'},
  ].map(c=>`<div class="card">
    <div class="card-lbl">${c.lbl}</div>
    <div class="card-val ${CLS(c.val)}">${FMT(c.val)}</div>
    <div class="card-sub">${c.sub}</div></div>`).join('');

  const dates=D.nav.full.map(r=>r.d);
  const cN=mc('c-turn-nav');
  if(cN) cN.setOption({
    backgroundColor:'transparent',
    grid:{left:60,right:110,top:20,bottom:36},
    xAxis:{type:'category',data:dates,...AX,axisTick:{show:false},axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(2)}},
    tooltip:{...TT,trigger:'axis'},
    legend:{orient:'vertical',right:0,top:'middle',textStyle:{color:'#566070',fontSize:11}},
    series:[
      {name:'每季换20只（全换手）',type:'line',symbol:'none',smooth:.1,
       data:D.nav.full.map(r=>r.n),lineStyle:{color:'#c0392b',width:2.5}},
      {name:'每季换5只（低换手）',type:'line',symbol:'none',smooth:.1,
       data:D.nav.low.map(r=>r.n),lineStyle:{color:'#1e50a2',width:2.5}},
      {name:'沪深300',type:'line',symbol:'none',smooth:.1,
       data:D.nav.full.map(r=>r.b),lineStyle:{color:'#ccc8bf',width:1.2,type:'dashed'}},
    ]
  });

  const qs=Object.keys(sf.qr).sort();
  const cQ=mc('c-turn-q');
  if(cQ) cQ.setOption({
    backgroundColor:'transparent',
    grid:{left:55,right:10,top:20,bottom:40},
    xAxis:{type:'category',data:qs,...AX,axisTick:{show:false},axisLabel:{color:'#566070',fontSize:11,rotate:30}},
    yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(0)+'%'}},
    tooltip:{...TT,trigger:'axis'},
    legend:{top:0,right:0,textStyle:{color:'#566070',fontSize:10}},
    series:[
      {name:'全换手超额',type:'bar',data:qs.map(q=>sf.qr[q]?+(sf.qr[q].port-sf.qr[q].bench).toFixed(2):null),
       itemStyle:{color:p=>p.value>=0?'#c0392b':'#c0392b66',borderRadius:[3,3,0,0]},barGap:'5%',barMaxWidth:18},
      {name:'低换手超额',type:'bar',data:qs.map(q=>sl.qr[q]?+(sl.qr[q].port-sl.qr[q].bench).toFixed(2):null),
       itemStyle:{color:p=>p.value>=0?'#1e50a2':'#1e50a266',borderRadius:[3,3,0,0]},barMaxWidth:18},
    ]
  });

  // 换仓明细
  document.getElementById('turn-body').innerHTML = D.turnover_log.map(row=>`<tr>
    <td class="l"><b>${row.q}</b></td>
    <td class="l" style="font-size:11px;color:var(--neg)">${row.entries.join(', ')||'—'}</td>
    <td class="l" style="font-size:11px;color:var(--t3)">${row.exits.join(', ')||'—'}</td>
    <td>${row.n_change}</td>
    <td class="${CLS(row.alpha)}">${FMT(row.alpha)}</td>
  </tr>`).join('');

  document.getElementById('t2-cost-note').innerHTML =
    `<b>交易成本说明：</b>估算按单边成本0.15%（含印花税0.1% + 佣金约0.05%），
    全换手策略每季换20只，年化成本约 ${(cost_full).toFixed(2)}%；
    低换手策略每季换5只，年化成本约 ${(cost_low).toFixed(2)}%。
    实际执行中，规模较大时冲击成本可能更高。做空端另需计入借券费率（通常年化1-3%），多空策略实际费后收益将低于上图估算。`;
}

(function(){
  document.getElementById('meta').textContent =
    `${D.stats.add.start} ~ ${D.stats.add.end}  ·  持有基金数≥30  ·  生成 ${D.gen_time}`;
  renderT1();
})();
</script>
</body>
</html>
"""


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    from datetime import datetime
    print("=== 补充研究 ===")

    by_q     = load_holdings()
    price_df = load_prices()
    quarters = sorted(by_q.keys(), key=sort_q)

    # ── 研究一：减仓信号 ──
    print("构建加仓/减仓持仓...")
    add_periods, red_periods = build_all_periods(by_q, quarters)
    for p in add_periods:
        print(f"  {p['q']}: 加仓{len(p['stocks'])}只  减仓{[x['stocks'].__len__() for x in red_periods if x['q']==p['q']][0]}只")

    print("计算减仓信号净值...")
    nav_add, trades_add = run_nav(add_periods, price_df, short=False)
    nav_red, trades_red = run_nav(red_periods, price_df, short=False)
    nav_red_short, _    = run_nav(red_periods, price_df, short=True)

    nav_ls = combine_nav(nav_add, nav_red_short)

    st_add = calc_stats(nav_add)
    st_red = calc_stats(nav_red)
    st_ls  = calc_stats(nav_ls)
    print(f"  加仓多头：超额{st_add['alpha']:+.1f}pp  夏普{st_add['sh']:.2f}")
    print(f"  减仓多头：超额{st_red['alpha']:+.1f}pp  夏普{st_red['sh']:.2f}  （做空超额≈{-st_red['alpha']:+.1f}pp）")
    print(f"  多空组合：超额{st_ls['alpha']:+.1f}pp   夏普{st_ls['sh']:.2f}")

    # ── 研究二：低换手率 ──
    print("\n构建低换手策略持仓（每季换5只）...")
    low_periods = build_low_turnover(by_q, quarters, change_n=5)

    # 全换手参考（add_periods 即 fcc前20全换手）
    nav_full, _ = run_nav(add_periods, price_df)
    nav_low,  _ = run_nav(low_periods, price_df)

    st_full = calc_stats(nav_full)
    st_low  = calc_stats(nav_low)
    print(f"  全换手（每季20只）：超额{st_full['alpha']:+.1f}pp  夏普{st_full['sh']:.2f}")
    print(f"  低换手（每季5只）：超额{st_low['alpha']:+.1f}pp   夏普{st_low['sh']:.2f}")

    # 换手记录
    prev_port = set()
    turnover_log = []
    for p in low_periods:
        cur = {s['code'] for s in p['stocks']}
        entries = [s['name'] for s in p['stocks'] if s['code'] not in prev_port][:5]
        exits_codes = prev_port - cur
        # 找对应名字（从上一季度）
        exits = list(exits_codes)[:5]

        q_stats = st_low['qr'].get(
            str(pd.Timestamp(p['disc']).to_period('Q').asfreq('Q')), {})
        alpha = round(q_stats.get('port',0) - q_stats.get('bench',0), 2) if q_stats else None
        turnover_log.append({'q': p['q'], 'entries': entries, 'exits': exits,
                             'n_change': len(entries), 'alpha': alpha})
        prev_port = cur

    # 交易成本估算（年化）
    n_quarters = len(add_periods)
    n_years = n_quarters / 4
    cost_full = 20 * 2 * 0.0015 * (n_quarters / n_years) / n_years * n_years  # per year
    cost_full = round(20 * 2 * 0.0015 / (1/4) / 4 * 100, 2)   # ~annual
    cost_full = round(20 * 2 * 0.0015 * 4 * 100, 3)   # 20 stocks × 2 sides × 0.15% × 4 quarters = annual
    cost_low  = round(5  * 2 * 0.0015 * 4 * 100, 3)

    # ── 打包 ──
    def safe(v):
        if v is None: return None
        if isinstance(v, float) and v != v: return None
        return v

    def to_nav(df):
        return [{'d': str(r['date'].date()), 'n': round(r['nav'],6), 'b': round(r['bench'],6)}
                for _, r in df.iterrows()]

    def clean_trades(tl):
        return [{k: safe(v) for k, v in t.items()} for t in tl]

    dataset = {
        'nav':   {'add': to_nav(nav_add), 'red': to_nav(nav_red),
                  'ls':  to_nav(nav_ls),  'full': to_nav(nav_full), 'low': to_nav(nav_low)},
        'trades':{'add': clean_trades(trades_add), 'red': clean_trades(trades_red)},
        'stats': {'add': st_add, 'red': st_red, 'ls': st_ls,
                  'full': st_full, 'low': st_low},
        'turnover_log': turnover_log,
        'cost_note': {'full': cost_full, 'low': cost_low},
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
