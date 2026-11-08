import { lazy, Suspense, useEffect, useState } from 'react'
import { ArrowUpRight, BrainCircuit, Check, FlaskConical, Play, Search, SlidersHorizontal, Sparkles } from 'lucide-react'
import { Badge, Button, Card } from './ui'
import type { Dataset, ModelCatalog, ModelParameter, RunResult } from '../types'
const Plot=lazy(()=>import('./Plot'))
const DataGrid=lazy(()=>import('./DataGrid'))
type Options=Record<string,unknown>
type Space=Record<string,Options>
export type Candidate={name:string;params:Options}
export type CandidatePlan={signature:string;candidates:Candidate[]}|null
export type Strategy='manual'|'range'|'optuna'
type Props={
  dataset?:Dataset;target:string;task:string;features:string[];preparation:Options;training:Options;ready:boolean
  model:string;setModel:(value:string)=>void;params:Options;setParams:(value:Options)=>void;catalog:ModelCatalog
  strategy:Strategy;setStrategy:(value:Strategy)=>void;count:number;setCount:(value:number)=>void
  space:Space;setSpace:(value:Space)=>void;plan:CandidatePlan;setPlan:(value:CandidatePlan)=>void
  loading:boolean;onRun:(candidates?:Candidate[])=>void;result:RunResult|null;onData:()=>void;onResults:()=>void
}
const help:Record<string,string>={
  fit_intercept:'Свободный член позволяет модели предсказывать ненулевой уровень при нулевых признаках. Отключайте его, если это соответствует смыслу данных.',
  alpha:'Сила регуляризации: большие значения сильнее ограничивают коэффициенты и могут уменьшать переобучение.',
  C:'Обратная сила регуляризации: меньшее C сильнее ограничивает сложность модели.',
  max_iter:'Предельное число итераций оптимизации. Если модель не сошлась, увеличьте лимит и проверьте масштабирование.',
  n_estimators:'Количество деревьев или базовых моделей. Увеличение обычно повышает время обучения.',
  max_depth:'Максимальная глубина дерева. Небольшая глубина ограничивает сложность; 0 означает отсутствие ограничения.',
  min_samples_leaf:'Минимум наблюдений в листе дерева. Большие значения сглаживают прогноз и ограничивают переобучение.',
  learning_rate:'Размер шага обучения. Малый шаг обычно требует больше деревьев или эпох.',
  learning_rate_init:'Начальный шаг оптимизации нейросети. Слишком большой шаг может мешать сходимости.',
  hidden_layer_sizes:'Число нейронов в каждом скрытом слое через запятую: 64, 32 задаёт два слоя.',
  activation:'Нелинейность скрытых слоёв. ReLU — обычная отправная точка; tanh ограничивает выход слоя диапазоном от −1 до 1.',
  n_neighbors:'Число ближайших соседей. Меньшее значение чувствительнее к локальным изменениям и шуму.',
  weights:'uniform даёт соседям равные веса; distance повышает вклад ближайших наблюдений.',
  kernel:'Ядро задаёт форму зависимостей: linear — линейную, rbf — гибкую нелинейную.',
  epsilon:'Ширина зоны нечувствительности к ошибке в SVR. Ошибки внутри этой зоны не штрафуются.',
  window:'Число последовательных строк на входе рекуррентной сети. Требуются история и разделение по времени.',
  layers:'Количество рекуррентных слоёв. Глубокая сеть требует больше данных и времени.',
  units:'Число нейронов рекуррентного слоя. Увеличение повышает ёмкость и стоимость модели.',
  dense_units:'Число нейронов полносвязного слоя после рекуррентной части.',
  dropout:'Доля отключаемых нейронов во время обучения для ограничения переобучения.',
  epochs:'Максимальное число проходов по обучающим данным.',
  batch_size:'Число наблюдений в одном шаге обновления весов.',
  optimizer:'Алгоритм обновления весов: Adam адаптирует шаги, SGD выполняет стохастический градиентный спуск.',
  l1_ratio:'Баланс L1 и L2 регуляризации: 1 соответствует L1, 0 — L2.',
  tol:'Допуск остановки оптимизации: меньшие значения задают более строгий критерий сходимости.',
}
function parameterHelp(name:string,schema:ModelParameter) {
  return help[name]??(schema.type==='choice'?'Выберите допустимое значение параметра алгоритма.':`Параметр «${schema.label??name}» управляет обучением выбранного алгоритма. Допустимый диапазон: ${schema.min??'—'}–${schema.max??'—'}.`)
}
async function request<T>(path:string,body?:unknown,signal?:AbortSignal):Promise<T>{
  const response=await fetch(`/api${path}`,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined,signal})
  const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Проверьте настройки модели и диапазоны поиска.');return data
}
function Parameter({name,schema,value,onChange}:{name:string;schema:ModelParameter;value:unknown;onChange:(v:unknown)=>void}){
  return <label className="field-label model-parameter"><span>{schema.label??name}</span>{schema.type==='choice'?<select className="field" value={String(value)} onChange={e=>onChange(schema.choices?.find(v=>String(v)===e.target.value)??e.target.value)}>{schema.choices?.map(v=><option key={String(v)} value={String(v)}>{String(v)}</option>)}</select>:schema.type==='layers'?<input className="field" value={Array.isArray(value)?value.join(', '):String(value)} onChange={e=>onChange(e.target.value.split(',').map(Number).filter(v=>Number.isInteger(v)&&v>0))} placeholder="64, 32"/>:<input className="field" type="number" min={schema.min} max={schema.max} step={schema.step??1} value={Number(value)} onChange={e=>onChange(Number(e.target.value))}/>}<small>{parameterHelp(name,schema)}</small></label>
}
function SpaceEditor({schema,space,setSpace}:{schema:Record<string,ModelParameter>;space:Space;setSpace:(v:Space)=>void}){
  const range=(name:string,label:string,defaultSpace:Options)=>{
    const value=space[name]??defaultSpace
    return <div className="model-range" key={name}><strong>{label}</strong><div className="field-row">{['min','max',...(value.type==='int'?['step']:[])].map(side=><label className="field-label" key={side}>{side==='min'?'От':side==='max'?'До':'Шаг'}<input className="field" type="number" step={value.type==='int'?1:'any'} value={Number(value[side]??1)} onChange={e=>setSpace({...space,[name]:{...value,[side]:Number(e.target.value)}})}/></label>)}</div>{value.type==='float'&&<label className="model-check"><input type="checkbox" checked={Boolean(value.log)} onChange={e=>setSpace({...space,[name]:{...value,log:e.target.checked}})}/>Логарифмическая шкала</label>}</div>
  }
  return <div className="model-space-grid">{Object.entries(schema).flatMap(([name,item])=>item.type==='layers'?[range('layer_count','Число скрытых слоёв',{type:'int',min:1,max:4,step:1}),range('layer_units','Нейронов в каждом слое',{type:'int',min:16,max:128,step:16})]:[<div className="model-range-wrapper" key={name}>{item.type==='choice'?<fieldset className="prep-multi"><legend>{item.label??name}</legend>{item.choices?.map(choice=>{const selected=(space[name]?.choices??item.choices??[]) as unknown[];return <label className="model-check" key={String(choice)}><input type="checkbox" checked={selected.includes(choice)} onChange={e=>setSpace({...space,[name]:{type:'choice',choices:e.target.checked?[...selected,choice]:selected.filter(v=>v!==choice)}})}/>{String(choice)}</label>})}</fieldset>:range(name,item.label??name,{type:item.type,min:item.min,max:item.max,step:item.step??1,log:Boolean(item.log)})}<p className="settings-note">{parameterHelp(name,item)}</p></div>])}</div>
}
function SearchTable({rows}:{rows:Options[]}){
  return <div className="model-search-table"><Suspense fallback={<p>Загрузка таблицы…</p>}><DataGrid rows={rows} columns={Object.keys(rows[0]??{}).map(field=>({field,flex:field==='Параметры'?3:1,minWidth:field==='Параметры'?300:130}))} rowHeight={46}/></Suspense></div>
}
export default function ModelTraining({dataset,target,task,features,preparation,training,ready,model,setModel,params,setParams,catalog,strategy,setStrategy,count,setCount,space,setSpace,plan,setPlan,loading,onRun,result,onData,onResults}:Props){
  const [category,setCategory]=useState('Все категории');const [query,setQuery]=useState('');const [error,setError]=useState('');const [generating,setGenerating]=useState(false);const [elapsed,setElapsed]=useState(0)
  const [details,setDetails]=useState<{description:string;theory:Record<string,unknown>;documentation:string}|null>(null)
  const schema=catalog.schemas[model]??{}
  const signature=JSON.stringify({task,model,count,params,space,seed:training.seed??42})
  const validPlan=plan?.signature===signature?plan:null
  const rows=catalog.catalog.filter(row=>catalog.models.includes(String(row['Модель'])))
  const categories=['Все категории',...new Set(rows.map(row=>String(row['Категория']??'')))]
  const filtered=rows.filter(row=>(category==='Все категории'||row['Категория']===category)&&[row['Модель'],row['Категория'],row['Библиотека']].join(' ').toLowerCase().includes(query.toLowerCase()))
  const active=rows.find(row=>row['Модель']===model)
  const metric=task==='Регрессия'?'R²':'F1 macro'
  const effectiveSpace:Space={}
  for(const [name,item] of Object.entries(schema)){
    if(item.type==='layers'){
      effectiveSpace.layer_count=space.layer_count??{type:'int',min:1,max:4,step:1}
      effectiveSpace.layer_units=space.layer_units??{type:'int',min:16,max:128,step:16}
    }else effectiveSpace[name]=space[name]??{...item}
  }
  const invalidSpace=Object.entries(effectiveSpace).find(([,item])=>item.type==='choice'?!Array.isArray(item.choices)||!item.choices.length:!Number.isFinite(Number(item.min))||!Number.isFinite(Number(item.max))||Number(item.min)>Number(item.max)||(item.type==='int'&&(!Number.isInteger(Number(item.min))||!Number.isInteger(Number(item.max))||!Number.isInteger(Number(item.step??1))||Number(item.step??1)<1))||(Boolean(item.log)&&Number(item.min)<=0))
  const countValid=Number.isInteger(count)&&count>=1&&count<=100
  const runnable=ready&&Boolean(target)&&features.length>0&&Boolean(model)&&!loading&&(strategy==='manual'||(countValid&&!invalidSpace&&(strategy!=='range'||Boolean(validPlan?.candidates.length))))
  useEffect(()=>{const controller=new AbortController();setDetails(null);if(model)request<typeof details>(`/models/details?name=${encodeURIComponent(model)}&task=${encodeURIComponent(task)}`,undefined,controller.signal).then(setDetails).catch(()=>{});return()=>controller.abort()},[model,task])
  useEffect(()=>{if(!loading){setElapsed(0);return}const started=Date.now();const timer=setInterval(()=>setElapsed(Math.floor((Date.now()-started)/1000)),1000);return()=>clearInterval(timer)},[loading])
  const generate=async()=>{
    setGenerating(true);setError('')
    try{const candidates=await request<Candidate[]>('/candidates',{task,models:[model],count,params:{...params,seed:training.seed??42},spaces:{[model]:{...effectiveSpace,__fixed__:params}}})
      const clean=candidates.map(candidate=>({name:candidate.name,params:Object.fromEntries(Object.entries(candidate.params).filter(([name])=>name in schema))}))
      setPlan({signature,candidates:clean})
    }catch(err){setError(err instanceof Error?err.message:'Не удалось сгенерировать варианты.')}finally{setGenerating(false)}
  }
  const candidateRows=validPlan?.candidates.map((candidate,i)=>({'Вариант':i+1,'Модель':candidate.name,'Параметры':JSON.stringify(candidate.params)}))??[]
  const leaderboard=result?.leaderboard??[]
  const resultSchema=catalog.schemas[result?.model??'']??{}
  const resultRows=leaderboard.map(row=>({...row,'Параметры':JSON.stringify(Object.fromEntries(Object.entries((row['Параметры']??{}) as Options).filter(([key])=>key in (catalog.schemas[String(row['Модель'])]??resultSchema))))}))
  const resultMetric=result?.task==='Классификация'?'F1 macro':'R²'
  const resultScores=leaderboard.filter(row=>typeof row[resultMetric]==='number'&&Number.isFinite(row[resultMetric]))
  const best=resultScores.reduce<Options|null>((best,row)=>!best||Number(row[resultMetric])>Number(best[resultMetric])?row:best,null)
  return <>
    {Boolean(preparation.anomaly_stage||preparation.restoration)&&<p className="prep-info">Включено восстановление архива: интерполяция и модель восстановления используют наблюдения из всего набора, включая будущие относительно разрыва. Метрики такого эксперимента не являются независимой оценкой прогноза. Для проверки прогноза отключите архивное восстановление в разделе «Данные».</p>}
    <div className="page-heading"><div><div className="eyebrow"><BrainCircuit size={14}/> ОБУЧЕНИЕ И СРАВНЕНИЕ</div><h1>Модели и подбор</h1><p>Выберите алгоритм, настройте параметры и сравните варианты по валидационной выборке.</p></div><Badge tone="blue">{task}</Badge></div>
    <div className="experiment-layout"><div className="experiment-form">
      <Card className="form-card model-training-card"><div className="form-step"><span className="step-number">01</span><div><h3>Выбор модели</h3><p>Категории объединяют алгоритмы с похожим устройством</p></div></div><div className="field-row"><label className="field-label">Категория модели<select className="field" value={category} onChange={e=>setCategory(e.target.value)}>{categories.map(c=><option key={c}>{c}</option>)}</select></label><label className="field-label">Поиск модели<div className="model-search-field"><Search size={14}/><input className="field" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Название, категория или библиотека"/></div></label></div><label className="field-label">Модель<select className="field" value={model} disabled={loading||generating} onChange={e=>{setModel(e.target.value);setSpace({});setError('')}}>{!filtered.some(row=>row['Модель']===model)&&model&&<option value={model}>{model} · текущий выбор</option>}{filtered.map(row=><option key={String(row['Модель'])}>{String(row['Модель'])}</option>)}</select></label>{!filtered.length&&<p className="settings-note">По этому фильтру доступных моделей нет. Измените категорию или поисковый запрос.</p>}<div className="model-note"><BrainCircuit size={17}/><span><strong>{model}</strong> · {String(active?.['Категория']??'')} · {String(active?.['Библиотека']??'')}</span></div>{details&&<><p className="model-explanation">{details.description||String(details.theory?.text??details.theory?.summary??details.theory?.description??'Параметры управляют сложностью модели и процессом её обучения.')}</p><details className="parameters-details"><summary>Как устроена модель и как читать параметры</summary><dl className="model-theory-notes">{([['intuition','Как работает'],['whenToUse','Когда использовать'],['assumptions','Предположения'],['strengths','Сильные стороны'],['limitations','Ограничения']] as const).map(([key,label])=>details.theory?.[key]?<div key={key}><dt>{label}</dt><dd>{String(details.theory[key])}</dd></div>:null)}</dl><a className="text-action" href={details.documentation} target="_blank" rel="noreferrer">Официальная документация<ArrowUpRight size={13}/></a></details></>}</Card>
      <Card className="form-card model-training-card"><div className="form-step"><span className="step-number">02</span><div><h3>Режим настройки</h3><p>Одна конфигурация или поиск лучших параметров</p></div></div><div className="model-strategies">{([['manual','Ручная настройка','Обучить одну модель с выбранными значениями.'],['range','Исследование диапазонов','Сначала создать список разных конфигураций, затем обучить их.'],['optuna','Автоматическая оптимизация','Optuna выбирает следующие параметры с учётом предыдущих испытаний.']] as const).map(([value,label,description])=><label key={value} className={strategy===value?'active':''}><input type="radio" name="model-strategy" value={value} checked={strategy===value} disabled={loading} onChange={()=>setStrategy(value)}/><strong>{label}</strong><small>{description}</small></label>)}</div><p className="prep-info">Во всех режимах параметры модели обучаются на train. Победитель выбирается по {metric} на validation: больше — лучше. Test оценивает итоговую модель и не участвует в подборе.</p></Card>
      <Card className="form-card model-training-card"><div className="form-step"><span className="step-number">03</span><div><h3>{strategy==='manual'?'Параметры алгоритма':'Диапазоны поиска'}</h3><p>{strategy==='manual'?'Значения для одного обучения':'Разрешённые значения для сравнения конфигураций'}</p></div><SlidersHorizontal size={17}/></div><fieldset disabled={loading||generating} className="model-fields">{strategy==='manual'?<div className="parameter-grid">{Object.entries(schema).map(([key,item])=><Parameter key={key} name={key} schema={item} value={params[key]??item.default} onChange={v=>setParams({...params,[key]:v})}/>)}</div>:<><SpaceEditor schema={schema} space={space} setSpace={setSpace}/><label className="field-label">{strategy==='range'?'Количество вариантов':'Количество испытаний Optuna'}<input className="field" type="number" min="1" max="100" value={count} onChange={e=>setCount(Number(e.target.value))}/></label><p className="settings-note">Чем больше вариантов, тем дольше поиск. Начните с 3–5. Random seed берётся из раздела «Данные»: {String(training.seed??42)}. Логарифмическая шкала удобна для регуляризации и шага обучения, которые меняются на несколько порядков.</p>{invalidSpace&&<p className="inline-error">Проверьте диапазон «{schema[invalidSpace[0]]?.label??invalidSpace[0]}»: границы должны быть согласованы, шаг — положительным целым, а список значений — непустым.</p>}{!countValid&&<p className="inline-error">Укажите целое число вариантов от 1 до 100.</p>}</>}{!Object.keys(schema).length&&<p className="settings-note">У этой модели нет дополнительных настраиваемых параметров.</p>}</fieldset></Card>
      {strategy==='range'&&<Card className="form-card model-training-card"><div className="card-heading"><div><h3>Варианты перед обучением</h3><p>Проверьте конкретные конфигурации до запуска</p></div><Button variant="secondary" disabled={loading||generating||Boolean(invalidSpace)||!countValid||!model} onClick={()=>void generate()}><Sparkles size={14}/>{generating?'Генерация…':'Сгенерировать модели'}</Button></div>{error&&<p className="inline-error">{error}</p>}{candidateRows.length?<><SearchTable rows={candidateRows}/><p className="settings-note">Сгенерировано {candidateRows.length} уникальных вариантов из запрошенных {count}. Если комбинаций меньше, повторения не добавляются. Обучатся именно показанные конфигурации.</p></>:<p className="model-explanation">{plan?'Настройки изменились. Сгенерируйте варианты заново.':'Задайте диапазоны и нажмите «Сгенерировать модели». Пока модели не обучаются.'}</p>}</Card>}
      {strategy==='optuna'&&<Card className="form-card"><h3>Как идёт оптимизация</h3><p className="model-explanation">Список заранее не фиксируется: Optuna TPE создаёт конфигурации последовательно и использует результаты validation для следующих шагов. После выполнения здесь появятся испытания, оценки и история поиска.</p></Card>}
    </div><aside className="experiment-aside"><Card className="run-summary"><div className="summary-head"><div className="run-summary-icon"><FlaskConical size={18}/></div><div><h3>Запуск обучения</h3><p>Данные настроены в разделе «Данные»</p></div></div><div className="summary-item"><span>Набор</span><strong>{dataset?.name??'Не выбран'}</strong></div><div className="summary-item"><span>Цель Y</span><strong>{target||'Не выбрана'}</strong></div><div className="summary-item"><span>Признаков X</span><strong>{features.length}</strong></div><div className="summary-item"><span>Алгоритм</span><strong>{model||'Не выбран'}</strong></div><div className="summary-item"><span>Режим</span><strong>{strategy==='manual'?'Один запуск':strategy==='range'?`${validPlan?.candidates.length??0} вариантов`:`Optuna · ${count}`}</strong></div><details className="model-data-summary"><summary>Настройки данных · только просмотр</summary><p className="settings-note">Разделение: {String(training.split??'time')==='time'?'по времени':'случайное'}. Масштабирование: {String(training.scaler??'standard')}. Горизонт: {Number(preparation.horizon??0)} строк. Восстановление: {preparation.repair?'включено':'выключено'}.</p><p className="settings-note">X: {features.join(', ')||'не выбраны'}</p></details><Button variant="ghost" onClick={onData}>Изменить данные<ArrowUpRight size={14}/></Button>{(!ready||!features.length)&&<p className="prep-info">Сначала подготовьте набор и выберите X в разделе «Данные».</p>}<Button className="run-button" disabled={!runnable||generating} onClick={()=>onRun(strategy==='range'?validPlan?.candidates:undefined)}>{loading?<><span className="spinner"/>Обучение · {elapsed} с</>:<><Play size={15}/>{strategy==='manual'?'Обучить модель':'Запустить поиск'}</>}</Button>{loading&&<p className="settings-note" role="status">{strategy==='manual'?'Модель обучается.':'Выполняется подбор конфигураций.'} Таблица и график появятся после завершения расчёта.</p>}<p className="run-hint"><Check size={13}/>Результат и конфигурация сохранятся в журнале</p></Card></aside></div>
    {result&&<Card className="model-results-card"><div className="card-heading"><div><h3>Результаты последнего подбора</h3><p>{result.model} · {result.target} · {String(result.config?.dataset??'')} · {leaderboard.length} вариантов</p></div><Button variant="secondary" onClick={onResults}>Открыть результаты<ArrowUpRight size={14}/></Button></div><p className="prep-info">Сравниваются оценки validation. Итоговое качество test: {resultMetric} = {typeof result.metrics[resultMetric]==='number'?Number(result.metrics[resultMetric]).toFixed(4):'—'}. {best?`Лучший вариант: №${best['Вариант']}.`:''}</p>{resultScores.length>0&&<Suspense fallback={<p>Загрузка истории поиска…</p>}><Plot data={[{x:resultScores.map(row=>Number(row['Вариант'])),y:resultScores.map(row=>Number(row[resultMetric])),type:'scatter',mode:'lines+markers',name:`Validation · ${resultMetric}`,line:{color:'#3978d5'},marker:{size:8}}]} layout={{autosize:true,height:300,margin:{l:65,r:25,t:35,b:50},title:{text:'История подбора · validation'},paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:document.documentElement.dataset.theme==='dark'?'#a4b1c5':'#64748b'},xaxis:{title:{text:'Номер варианта'},dtick:1},yaxis:{title:{text:resultMetric}},showlegend:false}} config={{responsive:true,displayModeBar:true}} style={{width:'100%'}}/></Suspense>}<SearchTable rows={resultRows}/><p className="settings-note">Статус и предупреждения помогают понять ошибки отдельных вариантов. Чем выше validation {resultMetric}, тем лучше результат подбора. R² может быть отрицательным. Повторный выбор по test снижает независимость итоговой оценки.</p></Card>}
  </>
}
