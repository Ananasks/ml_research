import type {Result} from './radioTypes'
export const wrappedPhase=(result:Result)=>result.diagnostics.real.map((v,i)=>Math.atan2(result.diagnostics.imag[i],v))
export function sampleSummary(values:number[]){const mean=values.reduce((a,b)=>a+b,0)/Math.max(1,values.length);return {mean,variance:values.reduce((s,v)=>s+(v-mean)**2,0)/Math.max(1,values.length)}}
export function channelComparison(reference:Result,result:Result){
 const a=reference.diagnostics,b=result.diagnostics
 if(a.time.length!==b.time.length||a.time.some((t,i)=>Math.abs(t-b.time[i])>1e-9))return null
 const realPhase=wrappedPhase(reference),predPhase=wrappedPhase(result)
 const score=(errors:number[],truth:number[])=>{const n=errors.length,mean=sampleSummary(truth).mean,sse=errors.reduce((v,e)=>v+e*e,0),total=truth.reduce((v,x)=>v+(x-mean)**2,0);return {count:n,rmse:Math.sqrt(sse/n),mae:errors.reduce((v,e)=>v+Math.abs(e),0)/n,r2:total>1e-20?1-sse/total:null}}
 const phaseError=predPhase.map((v,i)=>Math.atan2(Math.sin(v-realPhase[i]),Math.cos(v-realPhase[i])))
 return {amplitude:score(b.amplitude.map((v,i)=>v-a.amplitude[i]),a.amplitude),phase:score(phaseError,realPhase),power:score(b.amplitude.map((v,i)=>v*v-a.amplitude[i]**2),a.amplitude.map(v=>v*v)),phase_error:phaseError}
}
