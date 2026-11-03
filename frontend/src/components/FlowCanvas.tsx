import { useCallback, useEffect } from 'react'
import {
  Background, Controls, MiniMap, ReactFlow, ReactFlowProvider,
  type Connection, type Edge, type NodeChange, type EdgeChange, type Node,
  useReactFlow,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { nodeTypes, type ToolNode } from './WorkbenchNode'

type Props = {
  nodes: ToolNode[]
  edges: Edge[]
  onNodesChange: (changes: NodeChange<ToolNode>[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void
  onNodeClick: (node: ToolNode) => void
  onNodeDoubleClick: (node: ToolNode) => void
  onDropTool: (kind: string, position: { x: number; y: number }) => void
}

function FlowCanvasInner(props: Props) {
  const { screenToFlowPosition, fitView } = useReactFlow<ToolNode, Edge>()
  useEffect(() => {
    const frame = requestAnimationFrame(() => { void fitView({ padding: .22 }) })
    return () => cancelAnimationFrame(frame)
  }, [fitView, props.nodes.length])
  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    const kind = event.dataTransfer.getData('application/mlrl-tool')
    if (!kind) return
    props.onDropTool(kind, screenToFlowPosition({ x: event.clientX, y: event.clientY }))
  }, [props, screenToFlowPosition])
  const onDragOver = useCallback((event: React.DragEvent) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move' }, [])

  return <ReactFlow<ToolNode, Edge>
    nodes={props.nodes}
    edges={props.edges}
    nodeTypes={nodeTypes}
    onNodesChange={props.onNodesChange}
    onEdgesChange={props.onEdgesChange}
    onConnect={props.onConnect}
    onNodeClick={(_, node) => props.onNodeClick(node)}
    onNodeDoubleClick={(_, node) => props.onNodeDoubleClick(node)}
    onDrop={onDrop}
    onDragOver={onDragOver}
    fitView
    fitViewOptions={{ padding: .28 }}
    deleteKeyCode={['Backspace', 'Delete']}
    connectionLineStyle={{ stroke: 'var(--accent)', strokeWidth: 2 }}
    defaultEdgeOptions={{ type: 'smoothstep', animated: true, style: { stroke: 'var(--accent)', strokeWidth: 1.7 } }}
  >
    <Background color="var(--flow-dot)" gap={18} size={1} />
    <Controls />
    <MiniMap pannable zoomable nodeColor={(node: Node) => ({ source: '#3978d5', profile: '#2ea9a1', preprocess: '#8963ce', model: '#16a36a', evaluation: '#e9a23b', selection:'#3978d5', split:'#e9a23b', explain:'#8963ce', export:'#2ea9a1' }[(node.data as { kind?: string }).kind ?? ''] ?? '#94a3b8')} />
  </ReactFlow>
}

export default function FlowCanvas(props: Props) {
  return <ReactFlowProvider><FlowCanvasInner {...props} /></ReactFlowProvider>
}
