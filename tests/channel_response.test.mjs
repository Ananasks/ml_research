import assert from 'node:assert/strict'
import {test} from 'node:test'
import {requireChannelTransmission} from '../frontend/src/components/channelResponse.ts'

test('legacy channel response produces an actionable error before rendering',()=>{
  const legacy={config:{model:'clarke'},metrics:{mean_power:1},time_s:[0]}
  assert.throws(()=>requireChannelTransmission(legacy), /Перезапустите сервер API/)
  for(const transmission of [null,undefined,42,'old']) {
    assert.throws(()=>requireChannelTransmission({...legacy,transmission}), /прежний формат/)
  }
})
test('signal-path response remains available to the renderer',()=>{
  const value={transmission:{generator:'ofdm'}}
  assert.doesNotThrow(()=>requireChannelTransmission(value))
})
