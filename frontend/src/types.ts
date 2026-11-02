import type { ColDef } from 'ag-grid-community'
export type Dataset = { id: string; name: string; file: string; rows: number; columns: number; columnNames: string[]; timeColumns: string[] }
export type Preview = { columns: ColDef[]; rows: Record<string, unknown>[]; totalRows: number; chart: { name: string; points: { x: number | string; y: number | null }[] } }
export type ModelParameter = { label: string; type: string; min?: number; max?: number; step?: number; log?: boolean; default: unknown; choices?: unknown[] }
export type ModelCatalog = { models: string[]; catalog: Record<string, unknown>[]; schemas: Record<string, Record<string, ModelParameter>> }
export type RunResult = { id: string; model: string; task: string; target: string; features: string[]; metrics: Record<string, number | null>; trainMetrics: Record<string, number | null>; validation?: Record<string, number | null>; sizes: Record<string, number>; leaderboard: Record<string, unknown>[]; predictions: Record<string, unknown>[]; training: Record<string, unknown>[] }
