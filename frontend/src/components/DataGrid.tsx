import { AllCommunityModule, ModuleRegistry, themeQuartz, type ColDef } from 'ag-grid-community'
import { AgGridReact } from 'ag-grid-react'
import {formatDateTime,isDateTime} from '../lib/dateTime'

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
  const displayColumns=columns.map(column=>!column.valueFormatter&&column.field&&( /datetime|timestamp|date|time|дата|время/i.test(column.field)||rows.slice(0,10).some(row=>isDateTime(row[column.field!])))?{...column,valueFormatter:({value}:{value:unknown})=>formatDateTime(value)}:column)
  return <div className={className} style={{height:'100%',width:'100%'}}><AgGridReact<Record<string, unknown>> theme={themeQuartz} rowData={rows} columnDefs={displayColumns} defaultColDef={defaultColDef} rowHeight={rowHeight} headerHeight={headerHeight} animateRows /></div>
}
