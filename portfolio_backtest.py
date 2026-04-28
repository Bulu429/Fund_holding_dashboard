#!/usr/bin/env python3
"""
模拟组合回测：基金首次建仓信号（新进入 fc 前20）
每次披露日调仓，等权持有，输出净值曲线 + 交易记录 HTML
"""

import json, warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from workwork.research.策略研究.基金重仓看板.backtest_signals import (
    load_holdings, load_prices, DISCLOSURE_DATES, sort_q
)

BASE   = Path("workwork/research/策略研究/基金重仓看板")
OUTPUT = BASE / "fund_portfolio.html"

TOP_N     = 20
BENCHMARK = '000300.SH'

# ── 工具函数 ────────────────────────────────────────────────────────────────

def next_trade_date(price_df, target):
    """从 target 日起第一个有收盘价的交易日"""
    idx = price_df.index[price_df.index >= pd.Timestamp(target)]
    return idx[0] if len(idx) else None


def get_price(series, date):
    sub = series[series.index >= pd.Timestamp(date)].dropna()
    return (sub.index[0], float(sub.iloc[0])) if len(sub) else (None, None)


# ── 主回测逻辑 ──────────────────────────────────────────────────────────────

def run_portfolio(by_q, price_df):
    quarters   = sorted(by_q.keys(), key=sort_q)
    bench_ser  = price_df[BENCHMARK]

    # ── 逐期构建持仓 ──
    periods    = []          # 每期元信息
    all_trades = []          # 交易明细

    prev_codes = {r['code'] for r in by_q[quarters[0]]}   # 以 1Q24 初始化

    for i, q in enumerate(quarters[1:], 1):               # 2Q24 起
        disc_date = DISCLOSURE_DATES.get(q)
        next_q    = quarters[i + 1] if i + 1 < len(quarters) else None
        next_disc = DISCLOSURE_DATES.get(next_q) if next_q else None

        cur_map = {r['code']: r for r in by_q[q]}
        new_map = {c: r for c, r in cur_map.items()
                   if c not in prev_codes and r['fc'] is not None}
        top20   = sorted(new_map.items(),
                         key=lambda x: x[1]['fc'], reverse=True)[:TOP_N]

        # 过滤有价格的票
        holdings = []
        for code, rec in top20:
            if code not in price_df.columns:
                continue
            et, ep = get_price(price_df[code], disc_date)
            if et is None or ep is None or ep == 0:
                continue
            holdings.append({'code': code, 'name': rec['name'],
                              'ind': rec['ind'], 'fc': rec['fc'],
                              'fw': rec['fw'], 'pv': rec['pv'],
                              'entry_date': et, 'entry_price': ep})

        periods.append({
            'q': q, 'disc': disc_date,
            'next_disc': next_disc,
            'holdings': holdings,
        })
        prev_codes = set(cur_map.keys())

    # ── 逐日计算净值 ──
    nav_rows   = []    # [{date, port_nav, bench_nav}]
    port_nav   = 1.0
    bench_nav  = 1.0

    # 组合起始日 = 第一期披露日的下一个交易日
    start_date = next_trade_date(price_df, periods[0]['disc'])

    # 先把基准起始价格记录
    _, bench_start = get_price(bench_ser, start_date)

    nav_rows.append({'date': start_date, 'port': port_nav, 'bench': bench_nav})

    for pi, period in enumerate(periods):
        disc_ts  = pd.Timestamp(period['disc'])
        end_ts   = (pd.Timestamp(period['next_disc'])
                    if period['next_disc'] else price_df.index[-1])

        holdings = period['holdings']
        n        = len(holdings)
        if n == 0:
            continue

        w = 1.0 / n   # 等权

        # 取出本期所有持仓的价格序列
        # 日期范围：entry_date（可能各不同，但统一用 disc_date）到 end_ts
        entry_date_ts = next_trade_date(price_df, disc_ts)
        exit_date_ts  = next_trade_date(price_df, end_ts)

        # 日期范围内所有交易日
        dates_mask = (price_df.index > entry_date_ts) & (price_df.index <= exit_date_ts)
        period_dates = price_df.index[dates_mask]

        # 上一日收盘（即 entry_date 当日收盘）
        prev_prices = {}
        for h in holdings:
            p = price_df[h['code']]
            _, ep = get_price(p, entry_date_ts)
            prev_prices[h['code']] = ep

        prev_bench = float(bench_ser[bench_ser.index >= entry_date_ts].dropna().iloc[0])

        for d in period_dates:
            day_rets = []
            for h in holdings:
                p = price_df[h['code']]
                cur = p.get(d)
                prev = prev_prices[h['code']]
                if cur is not None and not np.isnan(cur) and prev and prev != 0:
                    day_rets.append(float(cur) / prev - 1.0)
                    prev_prices[h['code']] = float(cur)
                else:
                    day_rets.append(0.0)

            port_ret  = np.mean(day_rets) if day_rets else 0.0
            bench_cur = bench_ser.get(d)
            bench_ret = (float(bench_cur) / prev_bench - 1.0
                         if bench_cur and not np.isnan(bench_cur) and prev_bench != 0
                         else 0.0)
            if bench_cur and not np.isnan(bench_cur):
                prev_bench = float(bench_cur)

            port_nav  *= (1 + port_ret)
            bench_nav *= (1 + bench_ret)
            nav_rows.append({'date': d, 'port': round(port_nav, 6),
                             'bench': round(bench_nav, 6)})

        # 计算本期退出价格 & 交易记录
        for h in holdings:
            p = price_df[h['code']]
            xt, xp = get_price(p, exit_date_ts)
            if xp and h['entry_price'] and h['entry_price'] != 0:
                ret = round(xp / h['entry_price'] - 1.0, 6)
            else:
                ret = None
            # 同期基准收益
            be, bs_ = get_price(bench_ser, entry_date_ts)
            bx, bx_ = get_price(bench_ser, exit_date_ts)
            bench_ret_period = (round(bx_ / bs_ - 1.0, 6)
                                if bs_ and bx_ and bs_ != 0 else None)
            all_trades.append({
                'period': q,
                'disc_date': str(period['disc']),
                'code': h['code'], 'name': h['name'],
                'ind': h['ind'], 'fc': h['fc'],
                'fw': h['fw'], 'pv': h['pv'],
                'entry_date': str(h['entry_date'].date()),
                'entry_price': round(h['entry_price'], 3),
                'exit_date': str(xt.date()) if xt else None,
                'exit_price': round(xp, 3) if xp else None,
                'ret': round(ret * 100, 3) if ret is not None else None,
                'bench_ret': round(bench_ret_period * 100, 3) if bench_ret_period else None,
                'alpha': round((ret - bench_ret_period) * 100, 3)
                         if ret is not None and bench_ret_period is not None else None,
            })

    nav_df = pd.DataFrame(nav_rows).drop_duplicates('date').sort_values('date')
    return nav_df, all_trades, periods


# ── 统计指标 ────────────────────────────────────────────────────────────────

def calc_stats(nav_df):
    port  = nav_df['port'].values
    bench = nav_df['bench'].values
    dates = nav_df['date'].values

    total_ret   = port[-1] / port[0] - 1
    bench_total = bench[-1] / bench[0] - 1

    n_days  = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    ann_ret = (1 + total_ret) ** (365 / n_days) - 1 if n_days > 0 else 0

    daily_rets  = np.diff(port) / port[:-1]
    sharpe      = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
                   if np.std(daily_rets) > 0 else 0)

    running_max = np.maximum.accumulate(port)
    drawdowns   = port / running_max - 1
    max_dd      = float(np.min(drawdowns))

    # 季度收益
    q_rets = {}
    nav_df2 = nav_df.copy()
    nav_df2['ym'] = pd.to_datetime(nav_df2['date']).dt.to_period('Q')
    for qp, grp in nav_df2.groupby('ym'):
        q_rets[str(qp)] = {
            'port':  round(grp['port'].iloc[-1] / grp['port'].iloc[0] - 1, 4) * 100,
            'bench': round(grp['bench'].iloc[-1] / grp['bench'].iloc[0] - 1, 4) * 100,
        }

    return {
        'total_ret':   round(total_ret * 100, 2),
        'bench_total': round(bench_total * 100, 2),
        'ann_ret':     round(ann_ret * 100, 2),
        'sharpe':      round(sharpe, 2),
        'max_dd':      round(max_dd * 100, 2),
        'start_date':  str(pd.Timestamp(dates[0]).date()),
        'end_date':    str(pd.Timestamp(dates[-1]).date()),
        'q_rets':      q_rets,
    }


# ── HTML ────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>模拟组合净值 · 基金首次建仓fc前20</title>
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
  justify-content:space-between;position:sticky;top:0;z-index:50}
.brand{display:flex;align-items:center;gap:12px}
.icon{width:34px;height:34px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);
  border-radius:8px;display:flex;align-items:center;justify-content:center}
.hdr h1{font-size:15px;font-weight:700;color:#fff}
.hdr-sub{font-size:9px;color:rgba(255,255,255,.38);letter-spacing:1.2px;text-transform:uppercase}
.hdr-meta{color:rgba(255,255,255,.35);font-size:11px;font-family:var(--mono)}
.wrap{padding:22px 28px}
.cards{display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);
  padding:14px 18px;flex:1;min-width:130px;max-width:180px;box-shadow:var(--sh)}
.card-lbl{color:var(--t3);font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.7px}
.card-val{font-size:22px;font-weight:700;margin-top:5px;font-family:var(--mono);letter-spacing:-.5px}
.card-sub{font-size:11px;color:var(--t3);margin-top:4px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.neu{color:var(--t3)}
.box{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);padding:16px 18px;box-shadow:var(--sh);margin-bottom:14px}
.box-title{font-size:9px;color:var(--t3);margin-bottom:12px;font-weight:700;text-transform:uppercase;letter-spacing:.7px}
.g2{display:grid;grid-template-columns:3fr 1fr;gap:14px;margin-bottom:14px}
.tbl-wrap{overflow-x:auto;border-radius:var(--r);border:1px solid var(--bd);box-shadow:var(--sh);background:var(--card);margin-bottom:14px}
table{width:100%;border-collapse:collapse}
thead{position:sticky;top:0;z-index:10}
th{background:var(--th);color:var(--t3);font-weight:700;font-size:10px;text-transform:uppercase;
  letter-spacing:.5px;padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--bdt)}
th.l,td.l{text-align:left}
td{padding:7px 12px;border-top:1px solid #ede9e3;text-align:right;white-space:nowrap;
  font-size:12px;font-family:var(--mono);color:var(--t1)}
td.l{font-family:var(--ui);font-size:12px}
tr:nth-child(even) td{background:var(--alt)}
tr:hover td{background:var(--acc2)!important}
.code{color:var(--acc);font-size:11px}
.tag{display:inline-block;background:var(--acc2);color:var(--acc);padding:1px 6px;
  border-radius:4px;font-size:10px;font-weight:600;font-family:var(--ui)}
.period-hdr{background:#f0f4fb;cursor:pointer;user-select:none;font-weight:600;font-size:12px;font-family:var(--ui)}
.period-hdr td{padding:9px 12px!important;color:var(--acc)}
.period-hdr:hover td{background:#e4ecf9!important}
.hint{color:var(--t3);font-size:11px;margin-top:8px;line-height:1.6}
</style>
</head>
<body>
<div class="hdr">
  <div class="brand">
    <div class="icon"><svg width="16" height="16" fill="none"><path d="M1 15h4V9H1v6zm5-5h4v10H6V10zm5-4h4v14h-4V6z" fill="white" opacity=".9"/></svg></div>
    <div><h1>模拟组合净值回测</h1><div class="hdr-sub">首次建仓 fc前20 · 等权 · vs CSI300</div></div>
  </div>
  <span class="hdr-meta" id="meta"></span>
</div>

<div class="wrap">
  <div class="cards" id="cards"></div>

  <div class="g2">
    <div class="box">
      <div class="box-title">组合净值曲线</div>
      <div id="nav-chart" style="height:360px"></div>
    </div>
    <div class="box">
      <div class="box-title">季度收益对比</div>
      <div id="qret-chart" style="height:360px"></div>
    </div>
  </div>

  <div class="box">
    <div class="box-title">回撤曲线</div>
    <div id="dd-chart" style="height:180px"></div>
  </div>

  <div class="box" style="margin-bottom:14px">
    <div class="box-title">调仓记录（按季度展开）</div>
    <div class="tbl-wrap" style="margin:0">
      <table>
        <thead><tr>
          <th class="l">季度/代码</th><th class="l">名称</th><th class="l">行业</th>
          <th>持有基金数</th><th>入场日</th><th>入场价</th>
          <th>出场日</th><th>出场价</th><th>个股收益</th><th>基准收益</th><th>超额收益</th>
        </tr></thead>
        <tbody id="trade-body"></tbody>
      </table>
    </div>
  </div>

  <p class="hint">
    策略：每季度季报披露日（约次月20日）调仓，从当季「首次进入基金重仓名单」的个股中，
    取持有基金数（fc）排名前20只，等权持有至下季度披露日。基准：沪深300（000300.SH）。
    起始净值1.0，不计交易成本和印花税。
  </p>
</div>

<script>
const D = __DATA__;
const AX = {
  axisLine:{lineStyle:{color:'#ccc8bf'}},
  axisLabel:{color:'#96a0ae',fontSize:10,fontFamily:"'JetBrains Mono',monospace"},
  splitLine:{lineStyle:{color:'#e8e3db',type:'dashed'}},
};
const TT = {backgroundColor:'#fff',borderColor:'#ddd8d0',borderWidth:1,textStyle:{color:'#14213d',fontSize:11}};
const FMT = v => v==null?'—':(v>0?'+':'')+v.toFixed(2)+'%';
const CLS = v => v==null?'neu':v>0?'pos':v<0?'neg':'neu';

// ── Summary cards ────────────────────────────────────────────────────────
const s = D.stats;
const excess = s.total_ret - s.bench_total;
document.getElementById('meta').textContent = `${s.start_date} ~ ${s.end_date}  ·  生成 ${D.gen_time}`;
document.getElementById('cards').innerHTML = [
  {lbl:'组合累计收益', val:s.total_ret, sub:`基准 ${FMT(s.bench_total)}`},
  {lbl:'超额收益',    val:excess,       sub:'相对沪深300'},
  {lbl:'年化收益',    val:s.ann_ret,    sub:''},
  {lbl:'夏普比率',    val:s.sharpe,     sub:'日度，年化'},
  {lbl:'最大回撤',    val:s.max_dd,     sub:'组合持仓期'},
].map(c=>`<div class="card">
  <div class="card-lbl">${c.lbl}</div>
  <div class="card-val ${CLS(c.val)}">${c.lbl==='夏普比率'?c.val.toFixed(2):FMT(c.val)}</div>
  <div class="card-sub">${c.sub}</div></div>`).join('');

// ── NAV chart ────────────────────────────────────────────────────────────
const dates = D.nav.map(r=>r.d);
const ports  = D.nav.map(r=>r.p);
const benchs = D.nav.map(r=>r.b);

const cNav = echarts.init(document.getElementById('nav-chart'),null,{renderer:'canvas'});
cNav.setOption({
  backgroundColor:'transparent',
  grid:{left:55,right:20,top:20,bottom:36},
  xAxis:{type:'category',data:dates,...AX,axisTick:{show:false},
         axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
  yAxis:{type:'value',...AX,axisLine:{show:false},
         axisLabel:{formatter:v=>v.toFixed(2)}},
  tooltip:{...TT,trigger:'axis',formatter:p=>`${p[0].name}<br>
    <span style="color:#c0392b">●</span> 组合: ${p[0].value.toFixed(4)}<br>
    <span style="color:#96a0ae">●</span> 沪深300: ${p[1].value.toFixed(4)}`},
  series:[
    {name:'组合',type:'line',data:ports,smooth:.1,symbol:'none',
     lineStyle:{color:'#c0392b',width:2.5},
     areaStyle:{color:{type:'linear',x:0,y:0,x2:0,y2:1,
       colorStops:[{offset:0,color:'#c0392b22'},{offset:1,color:'#c0392b04'}]}}},
    {name:'沪深300',type:'line',data:benchs,smooth:.1,symbol:'none',
     lineStyle:{color:'#96a0ae',width:1.5,type:'dashed'}},
  ]
});

// ── Drawdown chart ────────────────────────────────────────────────────────
let maxP=ports[0], dds=ports.map(p=>{maxP=Math.max(maxP,p);return+(p/maxP-1).toFixed(6)*100;});
const cDd = echarts.init(document.getElementById('dd-chart'),null,{renderer:'canvas'});
cDd.setOption({
  backgroundColor:'transparent',
  grid:{left:55,right:20,top:10,bottom:36},
  xAxis:{type:'category',data:dates,...AX,axisTick:{show:false},axisLabel:{color:'#566070',fontSize:10,interval:'auto'}},
  yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(1)+'%'}},
  tooltip:{...TT,trigger:'axis',formatter:p=>`${p[0].name}<br>回撤: ${p[0].value.toFixed(2)}%`},
  series:[{type:'line',data:dds,smooth:.1,symbol:'none',
    lineStyle:{color:'#c0392b',width:1.5},
    areaStyle:{color:'#c0392b18'}}]
});

// ── Quarterly bar chart ────────────────────────────────────────────────────
const qKeys = Object.keys(D.stats.q_rets).sort();
const qPort  = qKeys.map(k=>D.stats.q_rets[k].port);
const qBench = qKeys.map(k=>D.stats.q_rets[k].bench);
const cQ = echarts.init(document.getElementById('qret-chart'),null,{renderer:'canvas'});
cQ.setOption({
  backgroundColor:'transparent',
  grid:{left:55,right:10,top:20,bottom:50},
  xAxis:{type:'category',data:qKeys,...AX,axisTick:{show:false},
         axisLabel:{color:'#566070',fontSize:10,rotate:30}},
  yAxis:{type:'value',...AX,axisLine:{show:false},axisLabel:{formatter:v=>v.toFixed(0)+'%'}},
  tooltip:{...TT,trigger:'axis'},
  legend:{top:0,right:0,textStyle:{color:'#566070',fontSize:10}},
  series:[
    {name:'组合',type:'bar',data:qPort,barGap:'5%',barMaxWidth:18,
     itemStyle:{color:p=>p.value>=0?'#c0392b':'#166534',borderRadius:[3,3,0,0]}},
    {name:'基准',type:'bar',data:qBench,barMaxWidth:18,
     itemStyle:{color:'#ccc8bf',borderRadius:[3,3,0,0]}},
  ]
});

// ── Trade log ────────────────────────────────────────────────────────────
const tbody = document.getElementById('trade-body');
let html = '';
// Group by period
const byPeriod = {};
D.trades.forEach(t=>{
  (byPeriod[t.period] = byPeriod[t.period]||[]).push(t);
});

// Period-level stats
Object.entries(byPeriod).sort((a,b)=>a[0].localeCompare(b[0])).forEach(([period, trades])=>{
  const rets  = trades.map(t=>t.ret).filter(v=>v!=null);
  const alphas= trades.map(t=>t.alpha).filter(v=>v!=null);
  const portR = rets.length ? (rets.reduce((s,r)=>s+r,0)/rets.length).toFixed(2) : '—';
  const portA = alphas.length ? (alphas.reduce((s,r)=>s+r,0)/alphas.length).toFixed(2) : '—';
  const bench = trades[0]?.bench_ret!=null ? trades[0].bench_ret.toFixed(2) : '—';
  const clsR  = portR==='—'?'neu':parseFloat(portR)>0?'pos':'neg';
  const clsA  = portA==='—'?'neu':parseFloat(portA)>0?'pos':'neg';

  html += `<tr class="period-hdr" onclick="togglePeriod('${period}')">
    <td class="l" colspan="8">▶ ${period}  披露日：${trades[0].disc_date}  持仓${trades.length}只</td>
    <td class="${clsR}">${portR==='—'?'—':'+'+portR+'%' }</td>
    <td>${bench==='—'?'—':bench+'%'}</td>
    <td class="${clsA}">${portA==='—'?'—':parseFloat(portA)>0?'+'+portA+'%':portA+'%'}</td>
  </tr>`;

  const rows = trades.map(t=>{
    const rc = t.ret!=null?CLS(t.ret):'neu';
    const ac = t.alpha!=null?CLS(t.alpha):'neu';
    return `<tr class="detail-${period}" style="display:none">
      <td class="l code">${t.code}</td>
      <td class="l">${t.name}</td>
      <td class="l"><span class="tag">${t.ind||'—'}</span></td>
      <td>${t.fc??'—'}</td>
      <td>${t.entry_date}</td>
      <td>${t.entry_price??'—'}</td>
      <td>${t.exit_date??'—'}</td>
      <td>${t.exit_price??'—'}</td>
      <td class="${rc}">${t.ret!=null?(t.ret>0?'+':'')+t.ret+'%':'—'}</td>
      <td>${t.bench_ret!=null?t.bench_ret+'%':'—'}</td>
      <td class="${ac}">${t.alpha!=null?(t.alpha>0?'+':'')+t.alpha+'%':'—'}</td>
    </tr>`;
  }).join('');
  html += rows;
});
tbody.innerHTML = html;

function togglePeriod(p){
  document.querySelectorAll('.detail-'+p).forEach(r=>{
    r.style.display = r.style.display==='none'?'':'none';
  });
}
// Open first period by default
const firstPeriod = Object.keys(byPeriod).sort()[0];
if(firstPeriod) togglePeriod(firstPeriod);
</script>
</body>
</html>
"""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    from datetime import datetime

    print("=== 模拟组合回测 ===")
    print("加载数据...")
    by_q      = load_holdings()
    price_df  = load_prices()

    print("运行组合回测...")
    nav_df, trades, periods = run_portfolio(by_q, price_df)
    print(f"  净值序列: {len(nav_df)} 天  交易记录: {len(trades)} 条")

    print("计算统计指标...")
    stats = calc_stats(nav_df)
    print(f"  累计收益: {stats['total_ret']:+.2f}%  基准: {stats['bench_total']:+.2f}%")
    print(f"  年化: {stats['ann_ret']:+.2f}%  夏普: {stats['sharpe']:.2f}  最大回撤: {stats['max_dd']:.2f}%")

    # 打包数据
    nav_data = [{'d': str(r['date'].date()), 'p': round(r['port'], 6), 'b': round(r['bench'], 6)}
                for _, r in nav_df.iterrows()]

    def safe(v):
        if v is None: return None
        if isinstance(v, float) and v != v: return None
        if isinstance(v, (pd.Timestamp,)): return str(v.date())
        return v

    trades_clean = [{k: safe(v) for k, v in t.items()} for t in trades]

    dataset = {
        'nav':      nav_data,
        'trades':   trades_clean,
        'stats':    stats,
        'gen_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    print("生成 HTML...")
    data_json = json.dumps(dataset, ensure_ascii=False, default=str, separators=(',', ':'))
    html = HTML.replace('__DATA__', data_json)
    OUTPUT.write_text(html, encoding='utf-8')
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"  输出: {OUTPUT}  ({size_kb:.0f} KB)")
    print("完成！")


if __name__ == '__main__':
    main()
