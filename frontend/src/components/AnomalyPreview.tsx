import {lazy,Suspense,useEffect,useState} from 'react'
import {Button} from './ui'
const Plot=lazy(()=>import('./Plot'))
type Options=Record<string,unknown>
type Point={x:string|number;before:number|null;anomaly:boolean;score?:number|null}
type Preview={repairSeries:Record<string,Point[]>;repairRecords:Record<string,unknown>[];sampled:boolean}
export default function AnomalyPreview({file,target,task,preparation}:{file:string;target:string;task:string;preparation:Options}){
 const key=JSON.stringify({file,target,task,preparation})
 const [revision,setRevision]=useState(0)
 const [requested,setRequested]=useState<string|null>(null)
 const [result,setResult]=useState<Preview|null>(null)
 const [busy,setBusy]=useState(false)
 const [error,setError]=useState('')
 const [column,setColumn]=useState('')
 useEffect(()=>{
  setResult(null);setError('');setBusy(false)
  if(requested!==key)return
  const controller=new AbortController();setBusy(true)
  const repair=preparation.repair as Options
  // Detection uses original observations. This isolated request never commits
  // prepared values or changes the shared experiment configuration.
  const previewPreparation={...preparation,horizon:0,lags:[],rolling_windows:[],ewm_spans:[],difference_lags:[],repair:{...repair,fill_method:'none',anomaly_action:'clip'}}
  fetch(`/api/datasets/${encodeURIComponent(file)}/prepare`,{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,body:JSON.stringify({target,task,preparation:previewPreparation,features:[],params:{}})})
   .then(async response=>{const value=await response.json();if(!response.ok)throw new Error(typeof value.detail==='string'?value.detail:'Не удалось выполнить поиск.');return value as Preview})
   .then(value=>{if(!controller.signal.aborted){setResult(value);setBusy(false)}})
   .catch(err=>{if(!controller.signal.aborted){setError(err.message);setBusy(false)}})
  return()=>controller.abort()
 },[key,requested,revision])
 const names=Object.keys(result?.repairSeries??{})
 const selected=names.includes(column)?column:names[0]??''
 const points=result?.repairSeries[selected]??[]
 const anomalies=points.filter(point=>point.anomaly)
 const record=result?.repairRecords.find(row=>row['Колонка']===selected)
 return <section className="repair-model-settings"><h4>Предпросмотр аномалий · до фильтрации</h4><p className="settings-note">Проверьте, какие исходные измерения отмечает выбранный алгоритм. Предпросмотр не применяет восстановление и не меняет данные эксперимента. После изменения настроек постройте его заново.</p><Button variant="secondary" disabled={busy||!(preparation.repair as Options)?.columns||(preparation.repair as {columns:string[]}).columns.length===0} onClick={()=>{setResult(null);setError('');setRequested(key);setRevision(value=>value+1)}}>{busy?'Поиск аномалий…':'Показать аномалии на графике'}</Button>{error&&<p role="alert" className="inline-error">{error}</p>}{result&&<><label className="field-label">Колонка для предпросмотра<select className="field" value={selected} onChange={event=>setColumn(event.target.value)}>{names.map(name=><option key={name}>{name}</option>)}</select></label><p>Найдено аномалий: <strong>{Number(record?.['Аномалий']??anomalies.length)}</strong>. Красные точки — кандидаты на коррекцию; пропуски не считаются аномальными измерениями.</p><Suspense fallback={<p>Загрузка графика…</p>}><Plot data={[{x:points.map(point=>point.x),y:points.map(point=>point.before),type:'scatter',mode:'lines',name:'Исходный ряд',line:{color:'#f59e0b',width:1.2},connectgaps:false},{x:anomalies.map(point=>point.x),y:anomalies.map(point=>point.before),customdata:anomalies.map(point=>point.score??null),type:'scatter',mode:'markers',name:'Выбранные аномалии',marker:{color:'#dc2626',size:8,symbol:'circle-open',line:{width:2}},hovertemplate:'%{x}<br>Значение: %{y}<br>Оценка: %{customdata}<extra>Аномалия</extra>'}]} layout={{autosize:true,height:460,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:document.documentElement.dataset.theme==='dark'?'#a4b1c5':'#64748b'},margin:{l:65,r:25,t:25,b:100},xaxis:{title:{text:'Дата / наблюдение'}},yaxis:{title:{text:selected}},legend:{orientation:'h',y:-.25}}} config={{responsive:true,displayModeBar:true}} style={{width:'100%'}}/></Suspense><p className="settings-note">Увеличивайте участок выделением мышью, двойной щелчок сбрасывает масштаб. {result.sampled?'Линия прорежена по всему периоду; все найденные аномалии сохранены.':''} Поиск учитывает прошлую историю и физические границы. До накопления минимальной истории модельные выбросы не отмечаются. Интерполяция найденных аномалий выполняется отдельно от восстановления исходных пропусков.</p></>}</section>
}
