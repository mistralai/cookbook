import { mistral } from '@agentskit/adapters'
import type { ToolDefinition } from '@agentskit/core'
import { createRuntime } from '@agentskit/runtime'

const cityTemperatures: Record<string, number> = {
  lisbon: 24,
  paris: 18,
  tokyo: 27,
}

function cityFrom(args: Record<string, unknown>) {
  if (typeof args.city !== 'string' || args.city.trim() === '') {
    throw new Error('city must be a non-empty string')
  }
  return args.city
}

export const getWeather: ToolDefinition = {
  name: 'get_weather',
  description: 'Return the current temperature for a supported city.',
  schema: {
    type: 'object',
    properties: {
      city: { type: 'string', description: 'City name.' },
    },
    required: ['city'],
    additionalProperties: false,
  },
  execute: (args) => {
    const city = cityFrom(args)
    const temperatureCelsius = cityTemperatures[city.toLowerCase()]
    if (temperatureCelsius === undefined) {
      return { city, available: false }
    }
    return { city, available: true, temperatureCelsius }
  },
}

export function createWeatherAgent(apiKey: string, model = 'mistral-small-latest') {
  return createRuntime({
    adapter: mistral({ apiKey, model }),
    tools: [getWeather],
    maxSteps: 3,
    systemPrompt: [
      'You answer weather questions for Lisbon, Paris, and Tokyo.',
      'Always call get_weather before answering.',
      'If the city is unavailable, say so plainly.',
    ].join(' '),
  })
}
