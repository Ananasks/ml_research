import { AllCommunityModule, ModuleRegistry, themeQuartz, type ColDef } from 'ag-grid-community'
import { AgGridReact } from 'ag-grid-react'

ModuleRegistry.registerModules([AllCommunityModule])

type Props = {
  rows: Record<string, unknown>[]
  columns: ColDef[]
  rowHeight?: number
  headerHeight?: number
  className?: string
  defaultColDef?: ColDef
}

export default function DataGrid({ rows, columns, rowHeight = 44, headerHeight = 42, className, defaultColDef = { sortable: true, filter: true, resizable: true } }: Props) {
  return <div className={className}><AgGridReact<Record<string, unknown>> theme={themeQuartz} rowData={rows} columnDefs={columns} defaultColDef={defaultColDef} rowHeight={rowHeight} headerHeight={headerHeight} animateRows /></div>
}
