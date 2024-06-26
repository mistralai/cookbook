import MistralClient from '@mistralai/mistralai';

const apiKey = process.env.MISTRAL_API_KEY;

const client = new MistralClient(apiKey);

const chatStreamResponse = client.chatStream({
  model: 'mistral-tiny',
  messages: [{role: 'user', content: 'What is the best French cheese?'}],
});

console.log('Chat Stream:');
for await (const chunk of chatStreamResponse) {
  if (chunk.choices[0].delta.content !== undefined) {
    const streamText = chunk.choices[0].delta.content;
    process.stdout.write(streamText);
  }
}
