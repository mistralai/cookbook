import MistralClient from '@mistralai/mistralai';

const apiKey = process.env.MISTRAL_API_KEY;

const client = new MistralClient(apiKey);

const listModelsResponse = await client.listModels();

listModelsResponse.data.forEach((model) => {
  console.log('Model:', model);
});
