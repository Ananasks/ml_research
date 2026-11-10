import type {PhysicalSeries} from './radioTypes'
export function profileByTime(series:PhysicalSeries){
 const groups=new Map<number,number[]>()
 series.time.forEach((t,i)=>{const date=new Date(t),hour=date.getUTCHours()+date.getUTCMinutes()/60+date.getUTCSeconds()/3600;groups.set(hour,[...(groups.get(hour)??[]),series.stec_tecu[i]])})
 const quantile=(a:number[],q:number)=>{const position=(a.length-1)*q,i=Math.floor(position);return a[i]+(a[Math.min(i+1,a.length-1)]-a[i])*(position-i)}
 return [...groups].sort(([a],[b])=>a-b).map(([hour,values])=>{const sorted=[...values].sort((a,b)=>a-b);return {hour,count:values.length,mean:values.reduce((a,b)=>a+b,0)/values.length,q25:quantile(sorted,.25),q75:quantile(sorted,.75),min:sorted[0],max:sorted.at(-1)!}})
}
