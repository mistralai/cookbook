import 'dotenv/config'
import { createWeatherAgent } from './agent.js'

const apiKey = process.env.MISTRAL_API_KEY
if (!apiKey) {
  throw new Error('Set MISTRAL_API_KEY in your environment or .env file.')
}

const agent = createWeatherAgent(apiKey, process.env.MISTRAL_MODEL)
const question = process.argv.slice(2).join(' ') || 'What is the temperature in Paris?'
const result = await agent.run(question)

console.log(result.content)
console.log(JSON.stringify({
  steps: result.steps,
  tools: result.toolCalls.map(({ name, status }) => ({ name, status })),
  durationMs: result.durationMs,
}, null, 2))
