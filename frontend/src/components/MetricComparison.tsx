import {lazy,Suspense} from 'react'
const Plot=lazy(()=>import('./Plot'))
const colors=['#3978d5','#8b78cb','#31a58e']
const format=(value:number)=>new Intl.NumberFormat('ru-RU',{maximumFractionDigits:3}).format(value)
export default function MetricComparison({metric,values}:{metric:string;values:{name:string;value:number|null|undefined}[]}){
 const points=values.filter((item):item is {name:string;value:number}=>typeof item.value==='number'&&Number.isFinite(item.value))
 const lower=/MAE|RMSE|MAPE|MSE|loss/i.test(metric)
 const dark=document.documentElement.dataset.theme==='dark'
 return <div className="metric-comparison"><div className="metric-scorecards">{points.map((item,index)=><div key={item.name}><span><i style={{background:colors[index]}}/>{item.name}</span><strong>{format(item.value)}<small>{metric}</small></strong></div>)}</div><p className="field-help">{lower?'Меньше — лучше.':'Больше — лучше.'} Сравнивайте одну метрику между выборками.</p><div className="metric-comparison-chart"><Suspense fallback={<p>Загрузка графика…</p>}><Plot data={[{x:points.map(item=>item.name),y:points.map(item=>item.value),type:'bar',width:.36,text:points.map(item=>format(item.value)),textposition:'outside',cliponaxis:false,marker:{color:colors,line:{width:0}},hovertemplate:`%{x}<br>${metric}: %{y:.4g}<extra></extra>`}]} layout={{height:300,autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:dark?'#a4b1c5':'#64748b',size:14},margin:{l:65,r:30,t:45,b:45},bargap:.65,xaxis:{type:'category',tickfont:{size:14},showgrid:false,zeroline:false,fixedrange:true},yaxis:{title:{text:metric,font:{size:14}},rangemode:'tozero',gridcolor:dark?'#2c3a50':'#edf1f6',zerolinecolor:dark?'#35445c':'#cbd5e1',automargin:true,nticks:5},showlegend:false}} config={{displayModeBar:false}}/></Suspense></div></div>
}
