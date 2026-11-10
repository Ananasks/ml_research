import assert from 'node:assert/strict'
import {test} from 'node:test'
import {comparisonTimes,sliceAnalysis} from '../frontend/src/components/radioPeriod.ts'
const series=(time,values)=>({time,tec:values,stec_tecu:values,delay_ns:values,phase_rad:values,
 path_m:values,dispersion_ns:values,coherence_bandwidth_mhz:values,delay_dispersion_s2:values,
 rate_tecu_s:values,doppler_hz:values})
test('periods use common UTC observations and keep every array aligned',()=>{
 const a=series(['2024-01-01T00:00:00Z','2024-01-02T00:00:00Z','2024-01-02T01:00:00Z'],[1,2,3])
 const b=series(a.time.slice(1),[4,5])
 const source={series:{measured:a,predicted:b},residual:null,alignment:{common_rows:2,unmatched:{}},comparison:null}
 assert.deepEqual(comparisonTimes(source),b.time)
 const sliced=sliceAnalysis(source,[b.time[0]])
 assert.deepEqual(sliced.series.measured.stec_tecu,[2])
 assert.deepEqual(sliced.series.predicted.stec_tecu,[4])
 assert.equal(sliced.comparison.stec.rmse,2)
 assert.equal(sliced.comparison.stec.count,1)
 assert.deepEqual(a.stec_tecu,[1,2,3])
})
test('no overlap stays empty and a single-source period remains usable',()=>{
 const a=series(['2024-01-01T00:00:00Z'],[1]),b=series(['2024-01-02T00:00:00Z'],[2])
 assert.deepEqual(comparisonTimes({series:{measured:a,predicted:b}}),[])
 assert.deepEqual(comparisonTimes({series:{predicted:b}}),b.time)
 assert.equal(sliceAnalysis(null,[]),null)
})

test('sampling respects actual days, minimum coverage and seeded selection',async()=>{
 const {studyDays,selectStudyDays}=await import('../frontend/src/components/radioPeriod.ts')
 const dates=Array.from({length:6},(_,i)=>`2024-01-0${i+1}T00:00:00Z`)
 dates.push('2024-01-01T01:00:00Z')
 const days=studyDays(dates,1),settings={start:'2024-01-02',end:'2024-01-06',count:3,method:'last',minPoints:1,seed:42}
 assert.deepEqual(selectStudyDays(days,settings),['2024-01-04','2024-01-05','2024-01-06'])
 assert.deepEqual(selectStudyDays(days,{...settings,method:'even'}),['2024-01-02','2024-01-04','2024-01-06'])
 assert.deepEqual(selectStudyDays(days,{...settings,method:'random'}),selectStudyDays(days,{...settings,method:'random'}))
 assert.deepEqual(selectStudyDays(studyDays(dates,2),{...settings,start:'2024-01-01',count:1}),['2024-01-01'])
 assert.equal(selectStudyDays(days,{...settings,count:0}).length,0)
})

test('Data owns the held-out date window and maximum test days',async()=>{
 const {dataTestAnalysis,comparisonTimes}=await import('../frontend/src/components/radioPeriod.ts')
 const times=['2024-01-01T00:00:00Z','2024-01-02T00:00:00Z','2024-01-03T00:00:00Z','2024-01-04T00:00:00Z']
 const analysis={series:{measured:series(times,[1,2,3,4]),predicted:series(times,[2,3,4,5])},residual:null,alignment:{common_rows:4},config:{}}
 assert.equal(dataTestAnalysis(analysis,{size_mode:'percent'}),analysis)
 const chosen=dataTestAnalysis(analysis,{size_mode:'days',day_start:'2024-01-01',day_end:'2024-01-03',test_days:2})
 assert.deepEqual(comparisonTimes(chosen),times.slice(1,3))
 assert.deepEqual(comparisonTimes(dataTestAnalysis(analysis,{size_mode:'days',test_days:0})),[])
 assert.deepEqual(comparisonTimes(analysis),times)
})
