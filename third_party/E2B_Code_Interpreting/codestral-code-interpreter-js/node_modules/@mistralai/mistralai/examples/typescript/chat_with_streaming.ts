import MistralClient from '@mistralai/mistralai';

const apiKey = process.env.MISTRAL_API_KEY;

const client = new MistralClient(apiKey);

const responseInterface = '{"best": string, "reasoning": string}';
const chatStreamResponse = client.chatStream({
  model: 'open-mistral-7b',
  responseFormat: {type: 'json_object'},
  messages: [{
    role: 'user', content: `
    What is the best French cheese?
    Answer in ${responseInterface} format`,
  }],
});

console.log('Chat Stream:');
for await (const chunk of chatStreamResponse) {
  if (chunk.choices[0].delta.content !== undefined) {
    const streamText = chunk.choices[0].delta.content;
    process.stdout.write(streamText);
  }
}
