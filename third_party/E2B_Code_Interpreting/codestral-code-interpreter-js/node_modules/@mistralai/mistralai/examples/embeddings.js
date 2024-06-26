import MistralClient from '@mistralai/mistralai';

const apiKey = process.env.MISTRAL_API_KEY;

const client = new MistralClient(apiKey);

const input = [];
for (let i = 0; i < 1; i++) {
  input.push('What is the best French cheese?');
}

const embeddingsBatchResponse = await client.embeddings({
  model: 'mistral-embed',
  input: input,
});

console.log('Embeddings Batch:', embeddingsBatchResponse.data);
