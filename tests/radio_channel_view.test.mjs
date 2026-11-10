import assert from 'node:assert/strict'
import {test} from 'node:test'
import {channelComparison} from '../frontend/src/components/channelView.ts'
import {profileByTime} from '../frontend/src/components/processView.ts'
const result=phase=>({diagnostics:{time:[0,1],amplitude:[1,1],real:[Math.cos(phase),Math.cos(phase)],imag:[Math.sin(phase),Math.sin(phase)]}})
test('phase errors use circular differences and constant amplitude has no R²',()=>{
 const scores=channelComparison(result(Math.PI-.01),result(-Math.PI+.01))
 assert.ok(Math.abs(scores.phase.rmse-.02)<1e-12)
 assert.equal(scores.amplitude.rmse,0)
 assert.equal(scores.amplitude.r2,null)
 const shifted=result(0);shifted.diagnostics.time=[0,2]
 assert.equal(channelComparison(result(0),shifted),null)
})
test('UTC daily profile uses empirical quantiles without invented dates',()=>{
 const profile=profileByTime({time:['2024-01-01T00:00:00Z','2024-01-02T03:00:00+03:00','2024-01-02T01:00:00Z'],stec_tecu:[10,20,30]})
 assert.equal(profile.length,2)
 assert.deepEqual(profile[0],{hour:0,count:2,mean:15,q25:12.5,q75:17.5,min:10,max:20})
})
