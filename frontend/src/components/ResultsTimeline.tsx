import { lazy, Suspense } from 'react'
import type { RunResult, ResultSeries } from '../types'
const Plot=lazy(()=>import('./Plot'))
type Row=Record<string,unknown>
export default function ResultsTimeline({result,series,split,title,xLabel,yLabel,factColor,forecastColor,width}:{result:RunResult;series:ResultSeries|null;split:string;title:string;xLabel:string;yLabel:string;factColor:string;forecastColor:string;width:number}){
  const score=result.task==='Регрессия'?'R²':'F1 macro'
  const best=result.leaderboard.filter(row=>row['Статус']==='Готово'&&typeof row[score]==='number').sort((a,b)=>Number(b[score])-Number(a[score]))[0]
  const validation=(result.validationPredictions??[]).filter(row=>!row['Модель']||row['Модель']===`${best?.['Вариант']} · ${best?.['Модель']}`)
  const fallback={train:result.training,validation,test:result.predictions}
  const groups=series?.splits??fallback
  const isTime=series?.splitMode!=='random'&&String((result.config?.params as Row|undefined)?.split??'time')!=='random'
  const regression=result.task==='Регрессия'
  const labels:Record<string,string>={train:'Train · обучающая выборка',validation:'Validation · выбор лучшей модели',test:'Test · независимая оценка'}
  const descriptions:Record<string,string>={train:'Показывает подгонку к данным, на которых модель обучалась. Высокое качество train само по себе не означает хорошего прогноза.',validation:'Показан прогноз только победившей конфигурации. Эта часть данных использовалась для выбора параметров.',test:'Отложенные наблюдения, которые не использовались для обучения и выбора победителя.'}
  return <div className="results-timeline-panels">{Object.entries(groups).filter(([key])=>split==='all'||key===split).map(([key,values])=>{
    const ordered=[...values].sort((a,b)=>{if(a['Время']&&b['Время'])return String(a['Время']).localeCompare(String(b['Время']));return Number(a['Индекс']??0)-Number(b['Индекс']??0)})
    const date=ordered.some(row=>typeof row['Время']==='string')
    const hasRestored=ordered.some(row=>row['Цель после восстановления']!=null)
    const traces=['Факт','Прогноз',...(hasRestored?['Восстановленные значения']:[]),...(ordered.some(row=>row['Базовый прогноз']!=null)?['Базовый прогноз']:[])].map((name,index)=>{
      const x:unknown[]=[];const y:unknown[]=[]
      ordered.forEach((row,i)=>{if(row._gapBefore){x.push(null);y.push(null)}x.push(row['Время']??row['Индекс']??i+1);
        const restored=name==='Восстановленные значения'
        const adjacent=[ordered[i-1],row,ordered[i+1]].some(point=>point&&point['Факт']==null)
        y.push(restored?(adjacent?row['Цель после восстановления']??null:null):row[name]??null)
      })
      return {x,y,type:'scatter',mode:regression&&isTime?'lines':'markers',name,connectgaps:false,line:{color:name==='Восстановленные значения'?'#d99128':index===0?factColor:index===1?forecastColor:'#16a34a',width:name==='Восстановленные значения'?Math.max(width,1.5):width,dash:name==='Восстановленные значения'?'dash':name==='Базовый прогноз'?'dot':'solid'},marker:{color:index===0?factColor:index===1?forecastColor:'#16a34a',size:4},opacity:name==='Базовый прогноз'?.65:1}
    })
    const gaps=ordered.filter(row=>row._gapBefore).length
    const restoredCount=ordered.filter(row=>row['Факт']==null&&row['Цель после восстановления']!=null).length
    const total=series?.totals[key as keyof typeof series.totals]??result.sizes[key]
    return <section className="results-timeline-panel" key={`${result.id}-${key}`}><h4>{split==='all'?labels[key]:title}</h4><p className="chart-explanation">{descriptions[key]} Показано {values.length.toLocaleString('ru-RU')} из {total?.toLocaleString('ru-RU')??'—'} строк.{values.length<Number(total)?' Ряд прорежен по всему периоду с сохранением экстремумов.':''}</p>{gaps>0&&<p className="timeline-gap-note">В этой выборке есть разрывы временного индекса: {gaps}. Для непрерывного ряда нужен новый расчёт с восстановлением сетки времени.</p>}{restoredCount>0&&<p className="field-help">Штриховая линия показывает восстановленные значения. Метрики рассчитаны только по исходным измерениям.</p>}{values.length?<Suspense fallback={<div className="plot-loading">Загрузка графика…</div>}><Plot key={`${result.id}-${key}-${date}`} data={traces as never} layout={{height:360,margin:{l:70,r:25,t:20,b:100},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:document.documentElement.dataset.theme==='dark'?'#a4b1c5':'#64748b'},xaxis:{type:date?'date':'linear',title:{text:date?`${xLabel} · UTC`:xLabel},automargin:true,nticks:6,tickformat:date?'%d.%m<br>%Y':undefined},yaxis:{title:{text:yLabel||result.target},type:regression?'linear':'category',automargin:true},legend:{orientation:'h',x:0,y:-.25,yanchor:'top'},hovermode:'x unified'}} config={{displayModeBar:true,toImageButtonOptions:{format:'png',filename:`prediction-${key}`}}}/></Suspense>:<p className="chart-explanation">Для этой выборки прогнозы не сохранены.</p>}</section>
  })}</div>
}
