import type {Series,TecAnalysis} from './TecWorkspace'
export type RadioCase='measured'|'clarke'|'predicted'
export const caseLabels:Record<RadioCase,string>={measured:'1 · Реальный ПЭС',clarke:'2 · Реальный ПЭС + Кларк',predicted:'3 · Прогноз ПЭС'}
export function comparisonTimes(analysis:TecAnalysis|null):string[]{
 if(!analysis)return []
 const {measured,predicted}=analysis.series
 if(measured&&predicted){const stamps=new Set(predicted.time);return measured.time.filter(t=>stamps.has(t))}
 return (measured??predicted)?.time??[]
}
export function sliceSeries(series:Series,times:Set<string>):Series{
 const indices=series.time.flatMap((t,i)=>times.has(t)?[i]:[])
 return Object.fromEntries(Object.entries(series).map(([key,values])=>[key,indices.map(i=>values[i])])) as Series
}
export function sliceAnalysis(analysis:TecAnalysis|null,times:string[]):TecAnalysis|null{
 if(!analysis)return null
 const allowed=new Set(times)
 const series=Object.fromEntries(Object.entries(analysis.series).map(([name,s])=>[name,sliceSeries(s,allowed)])) as TecAnalysis['series']
 const a=series.measured,b=series.predicted
 let comparison:TecAnalysis['comparison']=null
 if(a&&b){comparison={};for(const [name,field,unit] of [['stec','stec_tecu','TECU'],['delay','delay_ns','ns'],['phase','phase_rad','rad']] as const){
 const predicted=new Map(b.time.map((t,i)=>[t,b[field][i]]));const pairs=a.time.flatMap((t,i)=>predicted.has(t)?[[a[field][i],predicted.get(t)!]]:[])
 const n=pairs.length,mean=n?pairs.reduce((v,[x])=>v+x,0)/n:0
 const squared=pairs.reduce((v,[x,y])=>v+(y-x)**2,0),total=pairs.reduce((v,[x])=>v+(x-mean)**2,0),nonzero=pairs.filter(([x])=>Math.abs(x)>1e-8)
 comparison[name]={count:n,unit,mae:n?pairs.reduce((v,[x,y])=>v+Math.abs(y-x),0)/n:null,rmse:n?Math.sqrt(squared/n):null,bias:n?pairs.reduce((v,[x,y])=>v+y-x,0)/n:null,mape_percent:nonzero.length?100*nonzero.reduce((v,[x,y])=>v+Math.abs((y-x)/x),0)/nonzero.length:null,r2:n>1&&total>0?1-squared/total:null}
 }}
 return {...analysis,series,residual:analysis.residual?sliceSeries(analysis.residual,allowed):null,comparison,alignment:{...analysis.alignment,common_rows:a&&b?times.length:0}}
}

export type StudySettings={start:string;end:string;count:number;method:'last'|'first'|'even'|'random';minPoints:number;seed:number}
export type StudyDay={day:string;count:number;expected:number;coverage:number;incomplete:boolean;eligible:boolean}
export type ResearchSample={time:string;measured:number|null;predicted:number|null;measured_rate:number;predicted_rate:number}
export function studyDays(times:string[],minPoints:number):StudyDay[]{
 const sorted=[...times].sort(),deltas=sorted.slice(1).map((t,i)=>(Date.parse(t)-Date.parse(sorted[i]))/1000).filter(v=>v>0).sort((a,b)=>a-b)
 const step=deltas.length?deltas[Math.floor(deltas.length/2)]:86400
 const expected=Math.max(1,Math.round(86400/step))
 return [...new Set(sorted.map(t=>t.slice(0,10)))].map(day=>{const points=sorted.filter(t=>t.startsWith(day)),coverage=Math.min(1,(Date.parse(points.at(-1)!)-Date.parse(points[0])+step*1000)/86400000);return {day,count:points.length,expected,coverage,incomplete:points.length<expected||coverage<.95,eligible:points.length>=minPoints}})
}
export function selectStudyDays(days:StudyDay[],settings:StudySettings):string[]{
 const candidates=days.filter(d=>d.eligible&&(!settings.start||d.day>=settings.start)&&(!settings.end||d.day<=settings.end)).map(d=>d.day)
 const count=Math.max(0,Math.min(candidates.length,Math.floor(settings.count)))
 if(settings.method==='last')return candidates.slice(candidates.length-count)
 if(settings.method==='first')return candidates.slice(0,count)
 if(settings.method==='even')return count<2?candidates.slice(0,count):Array.from({length:count},(_,i)=>candidates[Math.round(i*(candidates.length-1)/(count-1))])
 let seed=settings.seed>>>0
 const shuffled=[...candidates]
 for(let i=shuffled.length-1;i>0;i--){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const j=Math.floor(seed/4294967296*(i+1));[shuffled[i],shuffled[j]]=[shuffled[j],shuffled[i]]}
 return shuffled.slice(0,count).sort()
}
export function researchSamples(analysis:TecAnalysis|null,times:string[]):ResearchSample[]{
 const a=analysis?.series.measured,b=analysis?.series.predicted
 const am=new Map(a?.time.map((t,i)=>[t,i])),bm=new Map(b?.time.map((t,i)=>[t,i]))
 return times.map(time=>{const i=am.get(time)??-1,j=bm.get(time)??-1;return {time,measured:i>=0?a!.stec_tecu[i]:null,predicted:j>=0?b!.stec_tecu[j]:null,measured_rate:i>=0?a!.rate_tecu_s[i]??0:0,predicted_rate:j>=0?b!.rate_tecu_s[j]??0:0}})
}

/** Apply Data's calendar test window to the saved held-out observations, never create new dates. */
export function dataTestAnalysis(analysis:TecAnalysis|null,training:Record<string,unknown>):TecAnalysis|null{
 if(!analysis||training.size_mode!=='days')return analysis
 const start=String(training.day_start??''),end=String(training.day_end??'')
 const times=comparisonTimes(analysis).filter(t=>(!start||t.slice(0,10)>=start)&&(!end||t.slice(0,10)<=end))
 const days=[...new Set(times.map(t=>t.slice(0,10)))].sort()
 const count=Number(training.test_days??1)
 const kept=Number.isInteger(count)&&count>0?new Set(days.slice(-count)):new Set<string>()
 return sliceAnalysis(analysis,times.filter(t=>kept.has(t.slice(0,10))))
}
