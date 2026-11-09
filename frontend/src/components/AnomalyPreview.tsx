import {lazy,Suspense,useEffect,useState} from 'react'
import {Button} from './ui'
const Plot=lazy(()=>import('./Plot'))
type Options=Record<string,unknown>
type Point={x:string|number;before:number|null;after:number|null;anomaly:boolean;score?:number|null}
type Preview={repairSeries:Record<string,Point[]>;repairRecords:Record<string,unknown>[];sampled:boolean}
export default function AnomalyPreview({file,target,task,preparation}:{file:string;target:string;task:string;preparation:Options}){
 const timeOptions=Object.fromEntries(['time_columns','dayfirst','strict_dates','drop_invalid_time','frequency','duplicate_policy','generate_index','start_datetime','index_frequency'].filter(key=>preparation[key]!==undefined).map(key=>[key,preparation[key]]))
 const key=JSON.stringify({file,target,task,preparation:{...timeOptions,calendar:false,repair:preparation.repair}})
 const [revision,setRevision]=useState(0)
 const [result,setResult]=useState<Preview|null>(null)
 const [busy,setBusy]=useState(false)
 const [error,setError]=useState('')
 const [column,setColumn]=useState('')
 useEffect(()=>{
  setResult(null);setError('');setBusy(false)
  if(!((preparation.repair as {columns?:string[]})?.columns?.length))return
  const controller=new AbortController();setBusy(true)
  const request=JSON.parse(key) as {preparation:Options}
  const repair=request.preparation.repair as Options
  // Detection uses original observations. This isolated request never commits
  // prepared values or changes the shared experiment configuration.
  const previewPreparation={...request.preparation,horizon:0,lags:[],rolling_windows:[],ewm_spans:[],difference_lags:[],repair:null,anomaly_stage:repair,restoration:null}
  let timeout:ReturnType<typeof setTimeout>|undefined
  const timer=setTimeout(()=>{
   timeout=setTimeout(()=>{controller.abort();setBusy(false);setError('Расчёт занял больше минуты. Повторите запрос или выберите ручную настройку.')},60000)
   fetch(`/api/datasets/${encodeURIComponent(file)}/prepare`,{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,body:JSON.stringify({target,task,preparation:previewPreparation,features:[],params:{}})})
   .then(async response=>{const value=await response.json();if(!response.ok)throw new Error(typeof value.detail==='string'?value.detail:'Не удалось выполнить поиск.');return value as Preview})
   .then(value=>{if(!controller.signal.aborted){setResult(value);setBusy(false)}})
   .catch(err=>{if(!controller.signal.aborted){setError(err.message);setBusy(false)}}).finally(()=>clearTimeout(timeout))
  },450)
  return()=>{clearTimeout(timer);clearTimeout(timeout);controller.abort()}
 },[key,revision])
 const names=Object.keys(result?.repairSeries??{})
 const selected=names.includes(column)?column:names[0]??''
 const points=result?.repairSeries[selected]??[]
 const plotType=points.length>2000?'scattergl':'scatter'
 const anomalies=points.filter(point=>point.anomaly)
 const record=result?.repairRecords.find(row=>row['Колонка']===selected)
 return <section className="anomaly-preview"><div className="anomaly-preview-toolbar"><h4>Результат обработки</h4><Button variant="secondary" disabled={busy||!((preparation.repair as {columns?:string[]})?.columns?.length)} onClick={()=>setRevision(value=>value+1)}>{busy?'Расчёт…':'Пересчитать'}</Button></div>{error&&<p role="alert" className="inline-error">{error}</p>}{names.length>1&&<label className="field-label">Колонка<select className="field" value={selected} onChange={event=>setColumn(event.target.value)}>{names.map(name=><option key={name}>{name}</option>)}</select></label>}<div aria-busy={busy}>{points.some(point=>point.before!=null)?<Suspense fallback={<p>Загрузка графика…</p>}><Plot data={[{x:points.map(point=>point.x),y:points.map(point=>point.before),type:plotType,mode:'lines',name:'Исходный ряд',line:{color:'#2563eb',width:1.2},connectgaps:false},{x:points.map(point=>point.x),y:points.map(point=>point.after),type:plotType,mode:'lines',name:'После обработки',line:{color:'#f59e0b',width:1.5},connectgaps:false},{x:anomalies.map(point=>point.x),y:anomalies.map(point=>point.before),customdata:anomalies.map(point=>point.score??null),type:plotType,mode:'markers',name:'Выбранные точки',marker:{color:'#dc2626',size:8,symbol:'circle-open',line:{width:2}},hovertemplate:'%{x}<br>Значение: %{y}<br>Оценка: %{customdata}<extra>Выбранная точка</extra>'}]} layout={{autosize:true,height:460,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:document.documentElement.dataset.theme==='dark'?'#a4b1c5':'#64748b'},margin:{l:65,r:25,t:25,b:100},xaxis:{title:{text:'Дата / наблюдение'}},yaxis:{title:{text:selected}},legend:{orientation:'h',y:-.25}}} config={{responsive:true,displayModeBar:true}} style={{width:'100%'}}/></Suspense>:<div className="anomaly-chart-placeholder" role="status">{busy?<><span className="spinner"/>Поиск аномалий и подготовка графика…</>:error?'Не удалось загрузить график. Повторите расчёт.':result?'В выбранном столбце нет числовых измерений для графика.':'Выберите столбец для отображения.'}</div>}</div><p className="settings-note">{busy?'Обновляем график…':result?`Выбрано точек: ${Number(record?.['Аномалий']??anomalies.length)}. Синий — исходный ряд, оранжевый — результат, красный — выбранные точки.`:'Выберите колонки для расчёта.'}</p>{Boolean(record?.['Автоподбор детектора'])&&<details className="anomaly-help"><summary>Результаты автоподбора</summary><p className="settings-note">Параметры: {String(record?.['Параметры детектора']??'Сохранены ручные настройки')} · проверено: {Number(record?.['Проверено конфигураций']??0)}.{record?.['F1 синтетической проверки']!=null&&<> F1 на искусственных выбросах: {String(record['F1 синтетической проверки'])}.</>} {String(record?.['Причина автоподбора']??'')}</p></details>}{Boolean(record?.['Диагностика обработки'])&&<p className="settings-note">{String(record?.['Диагностика обработки'])}</p>}</section>
}
