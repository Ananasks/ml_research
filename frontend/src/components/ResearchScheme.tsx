import { useMemo, useState } from 'react'
import { Background, Controls, MiniMap, ReactFlow, ReactFlowProvider, type Edge } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { Activity, ArrowRight, BookOpen, Check, CircleHelp, Workflow } from 'lucide-react'
import { Badge, Button, Card } from './ui'
import { nodeTypes, type ToolNode, type ToolKind } from './WorkbenchNode'
import type { RunResult } from '../types'

type JournalRecord = { id:string; created:string; model:string; task:string; target:string; config?:Record<string,unknown> }
type Page = 'overview'|'data'|'experiment'|'results'|'workflow'|'builder'|'models'|'journal'|'settings'
const labels:Record<ToolKind,string>={source:'Источник данных',profile:'Профиль данных',preprocess:'Предобработка',selection:'Цель и входы',split:'Train / validation / test',model:'Обучение модели',evaluation:'Оценка качества',explain:'Объяснимость',export:'Экспорт и отчёт'}

function toText(value:unknown){return value==null?'—':Array.isArray(value)?value.join(', '):typeof value==='object'?JSON.stringify(value):String(value)}
function toRecord(value:unknown):Record<string,unknown>{return value&&typeof value==='object'&&!Array.isArray(value)?value as Record<string,unknown>: {}}
function summaryFor(kind:ToolKind,config:Record<string,unknown>,result:RunResult){
 const params=(config.params&&typeof config.params==='object'?config.params:{}) as Record<string,unknown>
 const preparation=(config.preparation&&typeof config.preparation==='object'?config.preparation:{}) as Record<string,unknown>
 if(kind==='source')return `Набор: ${toText(config.dataset)}`
 if(kind==='profile')return 'Профиль вычислен перед подготовкой данных'
 if(kind==='selection')return `${result.target} · ${result.features.length} признаков`
 if(kind==='preprocess')return `${toText(params.imputer_strategy??'median')} · ${toText(params.scaler??'standard')}${Array.isArray(preparation.lags)&&preparation.lags.length?` · лаги ${preparation.lags.join(', ')}`:''}`
 if(kind==='split')return `${toText(params.split??'random')} · test ${Math.round(Number(params.test_size??.2)*100)}% · validation ${Math.round(Number(params.validation_size??.2)*100)}%`
 if(kind==='model')return `${result.model} · ${result.task}`
 if(kind==='explain')return `Расчёт: ${toText(config.method??'объяснение не запускалось')}`
 if(kind==='evaluation')return `${Object.keys(result.metrics??{}).length} метрик · train ${result.sizes.train??0} · test ${result.sizes.test??0}`
 return 'Результаты этого запуска доступны для экспорта'
}

function createFlow(result:RunResult,config:Record<string,unknown>){
 const rawNodes=Array.isArray(config.nodes)?config.nodes:Array.isArray(config.workflowNodes)?config.workflowNodes:[]
 const rawEdges=Array.isArray(config.edges)?config.edges:Array.isArray(config.workflowEdges)?config.workflowEdges:[]
 if(rawNodes.length){
  const nodes:ToolNode[]=rawNodes.map((raw,index)=>{
   const node=raw as ToolNode;const kind=node.data.kind
   return {...node,type:'research',position:node.position??{x:80+index*250,y:180},selected:false,data:{...node.data,label:node.data.label||labels[kind],status:'complete' as const,config:{...node.data.config,readOnly:true},summary:summaryFor(kind,{...config,...node.data.config},result)}} as ToolNode
  })
  const edges=rawEdges as Edge[]
  return {nodes,edges}
 }
 const positions:Record<ToolKind,{x:number;y:number}>={source:{x:40,y:180},profile:{x:250,y:180},selection:{x:250,y:180},preprocess:{x:465,y:180},split:{x:680,y:180},model:{x:895,y:180},explain:{x:1110,y:60},evaluation:{x:1110,y:230},export:{x:1325,y:230}}
 const configNodes:ToolNode[]=[
  {id:'source',type:'research',position:positions.source,data:{kind:'source',label:labels.source,status:'complete',config:{dataset:typeof config.dataset==='string'?config.dataset:undefined,readOnly:true}}},
  {id:'selection',type:'research',position:positions.selection,data:{kind:'selection',label:labels.selection,status:'complete',config:{target:result.target,features:result.features,readOnly:true}}},
  {id:'preprocess',type:'research',position:positions.preprocess,data:{kind:'preprocess',label:labels.preprocess,status:'complete',config:{preparation:toRecord(config.preparation),params:toRecord(config.params),readOnly:true}}},
  {id:'split',type:'research',position:positions.split,data:{kind:'split',label:labels.split,status:'complete',config:{params:toRecord(config.params),readOnly:true}}},
  {id:'model',type:'research',position:positions.model,data:{kind:'model',label:labels.model,status:'complete',config:{model:result.model,task:result.task,params:toRecord(config.params),readOnly:true}}},
  {id:'evaluation',type:'research',position:positions.evaluation,data:{kind:'evaluation',label:labels.evaluation,status:'complete',config:{readOnly:true}}},
 ]
 const edges:Edge[]=configNodes.slice(1).map((node,index)=>({id:`summary-${index}`,source:configNodes[index].id,target:node.id,type:'smoothstep',animated:false,style:{stroke:'var(--accent)',strokeWidth:2}}))
 return {nodes:configNodes.map(node=>({...node,data:{...node.data,summary:summaryFor(node.data.kind,config,result)}})),edges}
}

export default function ResearchScheme({experiment,journal,onGo}:{experiment:RunResult|null;journal:JournalRecord[];onGo:(page:Page)=>void}){
 const [selectedId,setSelectedId]=useState('')
 const journalEntry=journal.find(row=>row.id===experiment?.id)??journal.find(row=>row.model===experiment?.model&&row.task===experiment?.task&&row.target===experiment?.target)
 const config=(experiment?.config??journalEntry?.config??{}) as Record<string,unknown>
 const flow=useMemo(()=>experiment?createFlow(experiment,config):{nodes:[],edges:[]},[experiment,config])
 const selected=flow.nodes.find(node=>node.id===selectedId)??flow.nodes.find(node=>node.data.kind==='model')??flow.nodes[0]
 const datasetName=String(config.dataset??'Набор данных')
 const settings=selected?Object.entries(selected.data.config).filter(([key,value])=>key!=='readOnly'&&value!=null&&typeof value!=='function'):[ ]
 if(!experiment)return <section className="research-scheme-page"><div className="page-heading"><div><div className="eyebrow"><Workflow size={14}/> СИНХРОНИЗИРОВАННАЯ СХЕМА</div><h1>Схема исследования</h1><p>Здесь автоматически отображаются этапы последнего эксперимента и его настройки.</p></div><Badge tone="blue">Ожидает запуск</Badge></div><Card className="scheme-empty"><CircleHelp size={24}/><h3>Пока нет выполненного исследования</h3><p>Запустите обычный эксперимент или соберите его в конструкторе — схема появится автоматически.</p><div><Button onClick={()=>onGo('experiment')}>Модели и подбор</Button><Button variant="secondary" onClick={()=>onGo('builder')}>Открыть конструктор</Button></div></Card></section>
 return <section className="research-scheme-page"><div className="page-heading"><div><div className="eyebrow"><Workflow size={14}/> СИНХРОНИЗИРОВАННАЯ СХЕМА</div><h1>Схема исследования</h1><p>{datasetName} · {experiment.model} · {experiment.task} · {experiment.sizes.train??0} train / {experiment.sizes.test??0} test</p></div><div className="scheme-page-actions"><Badge tone="green"><Check size={12}/> Синхронизировано</Badge><Button variant="secondary" onClick={()=>onGo('results')}>Результаты <ArrowRight size={14}/></Button></div></div>
  <div className="scheme-summary-grid"><Card><small>Целевая колонка</small><strong>{experiment.target}</strong></Card><Card><small>Входные признаки</small><strong>{experiment.features.length}</strong><span>{experiment.features.slice(0,5).join(', ')}{experiment.features.length>5?'…':''}</span></Card><Card><small>Модель</small><strong>{experiment.model}</strong><span>{experiment.task}</span></Card>{Object.entries(experiment.metrics).slice(0,2).map(([metric,value])=><Card key={metric}><small>{metric} · test</small><strong>{value==null?'—':Number(value).toPrecision(4)}</strong></Card>)}</div>
  <div className="scheme-visual-layout"><Card className="scheme-flow-card"><div className="data-card-head"><div><div className="eyebrow"><Activity size={14}/> ВЫПОЛНЕННЫЙ ПРОЦЕСС</div><h3>{String(config.name??'Последний эксперимент')}</h3><p>Этапы и связи собраны из фактической конфигурации запуска. Схема доступна для просмотра; инструменты и их настройки изменяются в конструкторе.</p></div><Badge tone="green">{flow.nodes.length} блоков</Badge></div><div className="scheme-flow-canvas"><ReactFlowProvider><ReactFlow nodes={flow.nodes} edges={flow.edges} nodeTypes={nodeTypes} fitView fitViewOptions={{padding:.25}} nodesDraggable={false} nodesConnectable={false} elementsSelectable onNodeClick={(_,node)=>setSelectedId(node.id)} defaultEdgeOptions={{type:'smoothstep'}}><Background color="var(--flow-dot)" gap={18} size={1}/><Controls/><MiniMap pannable zoomable nodeColor={(node)=>({source:'#3978d5',profile:'#2ea9a1',preprocess:'#8963ce',model:'#16a36a',evaluation:'#e9a23b',selection:'#3978d5',split:'#e9a23b',explain:'#8963ce',export:'#2ea9a1'}[(node.data as {kind:ToolKind}).kind]??'#94a3b8')}/></ReactFlow></ReactFlowProvider></div></Card>
   <Card className="scheme-settings-card"><div className="eyebrow"><BookOpen size={14}/> НАСТРОЙКИ БЛОКА</div>{selected?<><h3>{selected.data.label}</h3><p>{selected.data.summary}</p><div className="scheme-setting-list">{settings.length?settings.map(([key,value])=><div key={key}><strong>{key}</strong><span>{toText(value)}</span></div>):<p>Отдельных параметров нет.</p>}</div></>:<div className="scheme-no-selection">Выберите блок схемы, чтобы посмотреть его настройки.</div>}<Button className="scheme-builder-cta" onClick={()=>onGo('builder')}>Изменить схему и настройки <ArrowRight size={14}/></Button></Card>
  </div><Card className="scheme-tools-note"><Workflow size={17}/><span>Все блоки остаются инструментами. Их можно добавлять, настраивать и запускать в независимом конструкторе.</span><Button variant="secondary" onClick={()=>onGo('builder')}>Перейти в конструктор</Button></Card>
 </section>
}
