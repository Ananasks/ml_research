import {lazy,Suspense} from 'react'
import {isDateTime} from '../lib/dateTime'
import type {Data,Layout,ScatterData} from 'plotly.js'
import {Card} from './ui'
import {RadioPlotFormula} from './RadioMath'
import type {IQ} from './radioTypes'
const Plot=lazy(()=>import('./Plot'))
export const colors={measured:'#168bd2',clarke:'#a759d3',predicted:'#13a974',fit:'#ed9735'}
export const fmt=(v:number|null|undefined,digits=3)=>v==null||!Number.isFinite(v)?'—':v.toLocaleString('ru-RU',{maximumFractionDigits:digits})
export function line(x:(number|string)[],y:(number|null)[],name:string,color:string,dash=false):Partial<ScatterData>{
 const xx:(number|string)[]=[],yy:(number|null)[]=[]
 const date=typeof x[0]==='string',stamps=date?x.map(t=>Date.parse(String(t))):[]
 const steps=stamps.slice(1).map((t,i)=>t-stamps[i]).filter(v=>v>0).sort((a,b)=>a-b),median=steps[Math.floor(steps.length/2)]
 x.forEach((t,i)=>{if(date&&i&&stamps[i]-stamps[i-1]>3*median){xx.push(t);yy.push(null)}xx.push(t);yy.push(y[i]??null)})
 return {x:xx,y:yy,name,type:'scatter',mode:'lines',connectgaps:false,line:{color,width:1.7,dash:dash?'dash':'solid'}}
}
export function dots(iq:IQ,name:string,color:string):Partial<ScatterData>{return {x:iq.i,y:iq.q,name,type:'scattergl',mode:'markers',marker:{color,size:3,opacity:.5}}}
export default function RadioPlot({title,data,x,y,formula,description,source='tec',height=210,log=false,equal=false,onTime}:{title:string;data:Data[];x:string;y:string;formula:string;description?:string;source?:'tec'|'clarke'|'statistics'|'signal'|'screen'|'physical';height?:number;log?:boolean;equal?:boolean;onTime?:(time:string)=>void}){
 const dateValues=data.flatMap(row=>('x' in row?Array.from((row.x??[]) as ArrayLike<unknown>):[]) as (number|string)[]).filter(v=>typeof v==='string') as string[]
 const oneDay=dateValues.length>0&&new Set(dateValues.map(t=>t.slice(0,10))).size===1
 const layout:Partial<Layout>={height,paper_bgcolor:'transparent',plot_bgcolor:'#fff',font:{size:10,color:'#56677c'},margin:{l:53,r:12,t:9,b:60},legend:{orientation:'h',y:-.35,font:{size:9}},xaxis:{title:{text:x,font:{size:10}},gridcolor:'#edf1f6',...(dateValues.some(isDateTime)?{type:'date',tickformat:'%H:%M:%S %Y-%m-%d',hoverformat:'%H:%M:%S %Y-%m-%d',tickangle:oneDay?0:-20}:{})},yaxis:{title:{text:y,font:{size:10}},gridcolor:'#edf1f6',type:log?'log':'linear',...(equal?{scaleanchor:'x',scaleratio:1}:{})}}
 const url=source==='screen'?'https://www.ion.org/publications/abstract.cfm?articleID=15286':source==='physical'?'https://gssc.esa.int/navipedia/index.php/Ionospheric_Delay':source==='clarke'?'https://doi.org/10.1002/j.1538-7305.1968.tb00069.x':source==='tec'?'https://www.itu.int/rec/R-REC-P.531/en':source==='statistics'?'https://docs.scipy.org/doc/scipy/reference/signal.html':'https://www.mathworks.com/help/comm/ref/comm.evm-system-object.html'
 return <Card className="rr-plot"><h3>{title}</h3>{data.length?<Suspense fallback={<div className="rr-chart-empty">Загрузка графика…</div>}><Plot data={data} layout={layout} config={{displaylogo:false,displayModeBar:'hover',toImageButtonOptions:{format:'svg',filename:title}}} onClick={event=>{const x=event.points[0]?.x;if(typeof x==='string')onTime?.(x)}}/></Suspense>:<div className="rr-chart-empty">Нет доступных данных для этого графика</div>}<details className="rr-formula"><summary>Формула и источник</summary><p>{description??title}. X: {x}; Y: {y}.</p><RadioPlotFormula title={title} formula={formula}/><p>{source==='screen'?'Один тонкий гауссов фазовый экран и распространение Френеля. Конечная периодическая сетка; параметры задаются сценарием.':source==='physical'?'Первый порядок холодной плазмы и отдельно выбранное рассеяние; локальное продолжение STEC на время опыта.':source==='tec'?'Первый порядок холодной плазмы. STEC в электронах/м²; частота в Гц; 1 TECU = 10¹⁶ электронов/м². Конвенция exp(+j2πft).':source==='clarke'?'Статистический эталон изотропного рассеяния. fD задаётся независимо от ПЭС; конечная реализация отличается от ансамбля.':source==='statistics'?'Описательный анализ доступных наблюдений. Пропуски не заполняются, дни не склеиваются. Аппроксимация не является прогнозом.':'Метрики информационных символов выбранной модуляции. Шум задаётся через Es/N₀ на передатчике; пилоты исключены.'}</p><a href={url} target="_blank" rel="noreferrer">{source==='screen'?'Фазовые экраны · Rino и соавторы':source==='physical'?'ESA · Ionospheric Delay':source==='tec'?'ITU-R P.531-16':source==='clarke'?'Clarke, 1968':'Определения и методы'}</a></details></Card>
}
