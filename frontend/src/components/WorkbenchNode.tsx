import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import { Activity, BrainCircuit, Check, Circle, Database, Gauge, LoaderCircle, ScanSearch, SlidersHorizontal, TriangleAlert, Scissors, Target, FileOutput, Sparkles } from 'lucide-react'

export type ToolKind = 'source' | 'profile' | 'preprocess' | 'selection' | 'split' | 'model' | 'evaluation' | 'explain' | 'export'
export type NodeStatus = 'idle' | 'running' | 'complete' | 'error' | 'dirty'
export type ToolConfig = {
  dataset?: string
  imputer?: string
  scaler?: string
  task?: string
  target?: string
  features?: string[]
  model?: string
  params?: Record<string, unknown>
  preparation?: Record<string, unknown>
  training?: Record<string, unknown>
  strategy?: string
  count?: number
  searchSpace?: Record<string, Record<string, unknown>>
  operation?: string
  method?: string
}
export type ToolNodeData = {
  kind: ToolKind
  label: string
  config: ToolConfig
  status?: NodeStatus
  summary?: string
}
export type ToolNode = Node<ToolNodeData, 'research'>

const iconByKind = {
  source: Database,
  profile: ScanSearch,
  preprocess: SlidersHorizontal,
  model: BrainCircuit,
  evaluation: Gauge,
  selection: Target,
  split: Scissors,
  explain: Sparkles,
  export: FileOutput,
}
const colors = { source: 'blue', profile: 'cyan', preprocess: 'violet', selection:'blue',split:'amber',model:'green',evaluation:'amber',explain:'violet',export:'cyan' }
const statusLabels = { idle: 'Ожидает запуска', running: 'Выполняется', complete: 'Выполнен', error: 'Ошибка', dirty: 'Нужно пересчитать' }

export function WorkbenchNode({ data, selected }: NodeProps<ToolNode>) {
  const Icon = iconByKind[data.kind] ?? Activity
  const status = data.status ?? 'idle'
  return <div className={`work-node node-${colors[data.kind]} ${selected ? 'is-selected' : ''} state-${status}`}>
    {data.kind !== 'source' && <Handle type="target" position={Position.Left} className="work-handle" />}
    <div className="work-node-head"><span className="work-node-icon"><Icon size={16} /></span><span className="work-node-title">{data.label}</span><span className={`node-status-icon ${status}`}>{status === 'complete' ? <Check size={12} /> : status === 'running' ? <LoaderCircle size={12} className="spin" /> : status === 'error' ? <TriangleAlert size={12} /> : <Circle size={8} />}</span></div>
    <div className="work-node-description">{data.summary ?? summaryByKind[data.kind]}</div>
    <div className="work-node-status"><span className={`status-indicator ${status}`} />{statusLabels[status]}</div>
    {!['export'].includes(data.kind) && <Handle type="source" position={Position.Right} className="work-handle" />}
  </div>
}

const summaryByKind: Record<ToolKind, string> = {
  source: 'Открыть таблицу и выбрать источник данных',
  profile: 'Статистика, пропуски, распределения и корреляции',
  preprocess: 'Импутация и масштабирование внутри train',
  model: 'Обучение на выбранных признаках',
  evaluation: 'Метрики, прогнозы и диагностические графики',
  selection: 'Выбор целевой колонки и входных признаков',
  split: 'Настройка границ train, validation и test',
  explain: 'Permutation importance, SHAP или анализ последовательности',
  export: 'Архив результатов, прогнозы и отчёт',
}

export const nodeTypes = { research: WorkbenchNode }
