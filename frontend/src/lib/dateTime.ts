/** Display timestamps in UTC; preserve the original values used for calculations. */
export function isDateTime(value:unknown):boolean{return value instanceof Date||typeof value==='string'&&/^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$/.test(value)}
export function formatDateTime(value:unknown):string{
 if(value==null)return ''
 if(!isDateTime(value))return String(value)
 let iso=value instanceof Date?(Number.isNaN(value.getTime())?'':value.toISOString()):String(value)
 if(typeof value==='string'&&/(?:Z|[+-]\d{2}:?\d{2})$/.test(value)){
  const date=new Date(value)
  if(!Number.isNaN(date.getTime()))iso=date.toISOString()
 }
 const match=/^(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?/.exec(iso)
 return match?`${match[2]}:${match[3]}:${match[4]??'00'} ${match[1]}`:String(value)
}
