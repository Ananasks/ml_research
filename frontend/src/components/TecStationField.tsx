import {useEffect,useState,lazy,Suspense} from 'react'
import {Card} from './ui'
const Plot=lazy(()=>import('./Plot'))
type Field={stations:string[];time:string[];values:(number|null)[][];note:string}
export default function TecStationField({dataset,times,onData}:{dataset:string;times:string[];onData:()=>void}){
 const [field,setField]=useState<Field|null>(null),[error,setError]=useState('')
 const key=JSON.stringify({dataset,times})
 useEffect(()=>{const abort=new AbortController();setField(null);setError('');if(!dataset)return;fetch('/api/radio/tec/stations',{method:'POST',headers:{'Content-Type':'application/json'},body:key,signal:abort.signal}).then(async r=>{if(!r.ok)throw Error('Не удалось прочитать наблюдения станций');return r.json()}).then(setField).catch(e=>{if(!abort.signal.aborted)setError(e.message)});return()=>abort.abort()},[key])
 return <Card className="rr-summary-card"><h3>Пространственно-временные вариации ПЭС</h3>{field?.stations.length&&field.time.length?<><Suspense fallback={<p>Загрузка поля станций…</p>}><Plot data={[{type:'heatmap',x:field.time,y:field.stations,z:field.values,colorscale:'YlGnBu',connectgaps:false,hoverongaps:false,colorbar:{title:{text:'TECU'},thickness:10},hovertemplate:'Станция %{y}<br>%{x}<br>ПЭС %{z:.3f} TECU<extra></extra>'}]} layout={{height:240,margin:{l:48,r:65,t:10,b:48},paper_bgcolor:'white',plot_bgcolor:'white',xaxis:{title:{text:'Время, UTC'},type:'date',tickformat:'%H:%M'},yaxis:{title:{text:'Станция'}}}}/></Suspense><p>{field.note}</p><p className="formula-caption"><strong>Формула поля наблюдений:</strong> T(sᵢ,tⱼ) = TEC станции i в момент tⱼ. Пропуски не заполняются.</p></>:<><p>{error||field?.note||'Чтение пространственных наблюдений…'}</p><p>Географическая карта TEC(φ, λ, t) требует координат станций. Временной ряд одной станции не определяет пространственное поле.</p><button className="button" onClick={onData}>Посмотреть колонки данных</button></>}</Card>
}
