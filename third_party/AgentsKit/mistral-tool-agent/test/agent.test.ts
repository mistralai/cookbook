import assert from 'node:assert/strict'
import test from 'node:test'
import { createWeatherAgent } from '../src/agent.js'

function eventStream(events: unknown[]) {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n'
  return new Response(body, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  })
}

test('Mistral calls the typed tool and uses its result', async (context) => {
  const originalFetch = globalThis.fetch
  const requests: Array<{ url: string; body: Record<string, unknown> }> = []
  let turn = 0

  globalThis.fetch = async (input, init) => {
    requests.push({
      url: String(input),
      body: JSON.parse(String(init?.body)) as Record<string, unknown>,
    })
    turn += 1
    if (turn === 1) {
      return eventStream([
        {
          choices: [{
            delta: {
              tool_calls: [{
                index: 0,
                id: 'weather-1',
                function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
              }],
            },
          }],
        },
      ])
    }
    return eventStream([
      { choices: [{ delta: { content: 'Paris is 18°C.' } }] },
    ])
  }
  context.after(() => { globalThis.fetch = originalFetch })

  const result = await createWeatherAgent('test-key').run('Weather in Paris?')

  assert.equal(result.content, 'Paris is 18°C.')
  assert.equal(result.steps, 2)
  assert.deepEqual(
    result.toolCalls.map(({ name, status, result: value }) => ({ name, status, value })),
    [{
      name: 'get_weather',
      status: 'complete',
      value: '{"city":"Paris","available":true,"temperatureCelsius":18}',
    }],
  )
  assert.equal(requests[0]?.url, 'https://api.mistral.ai/v1/chat/completions')
  assert.equal((requests[0]?.body.tools as Array<{ function: { name: string } }>)[0]?.function.name, 'get_weather')
  const secondTurnMessages = requests[1]?.body.messages as Array<{ role: string; content: string }>
  assert.equal(secondTurnMessages.at(-1)?.role, 'tool')
  assert.equal(secondTurnMessages.at(-1)?.content, '{"city":"Paris","available":true,"temperatureCelsius":18}')
})
