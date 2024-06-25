import jest from 'jest-mock';

/**
 * Mock the fetch function
 * @param {*} status
 * @param {*} payload
 * @return {Object}
 */
export function mockFetch(status, payload) {
  return jest.fn(() =>
    Promise.resolve({
      json: () => Promise.resolve(payload),
      text: () => Promise.resolve(JSON.stringify(payload)),
      status,
      ok: status >= 200 && status < 300,
    }),
  );
}

/**
 * Mock fetch stream
 * @param {*} status
 * @param {*} payload
 * @return {Object}
 */
export function mockFetchStream(status, payload) {
  const asyncIterator = async function* () {
    while (true) {
      // Read from the stream
      const value = payload.shift();
      // Exit if we're done
      if (!value) return;
      // Else yield the chunk
      yield value;
    }
  };

  return jest.fn(() =>
    Promise.resolve({
      // body is a ReadableStream of the objects in payload list
      body: asyncIterator(),
      status,
      ok: status >= 200 && status < 300,
    }),
  );
}

/**
 * Mock models list
 * @return {Object}
 */
export function mockListModels() {
  return {
    object: 'list',
    data: [
      {
        id: 'mistral-medium',
        object: 'model',
        created: 1703186988,
        owned_by: 'mistralai',
        root: null,
        parent: null,
        permission: [
          {
            id: 'modelperm-15bebaf316264adb84b891bf06a84933',
            object: 'model_permission',
            created: 1703186988,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: false,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: '*',
            group: null,
            is_blocking: false,
          },
        ],
      },
      {
        id: 'mistral-small-latest',
        object: 'model',
        created: 1703186988,
        owned_by: 'mistralai',
        root: null,
        parent: null,
        permission: [
          {
            id: 'modelperm-d0dced5c703242fa862f4ca3f241c00e',
            object: 'model_permission',
            created: 1703186988,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: false,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: '*',
            group: null,
            is_blocking: false,
          },
        ],
      },
      {
        id: 'mistral-tiny',
        object: 'model',
        created: 1703186988,
        owned_by: 'mistralai',
        root: null,
        parent: null,
        permission: [
          {
            id: 'modelperm-0e64e727c3a94f17b29f8895d4be2910',
            object: 'model_permission',
            created: 1703186988,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: false,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: '*',
            group: null,
            is_blocking: false,
          },
        ],
      },
      {
        id: 'mistral-embed',
        object: 'model',
        created: 1703186988,
        owned_by: 'mistralai',
        root: null,
        parent: null,
        permission: [
          {
            id: 'modelperm-ebdff9046f524e628059447b5932e3ad',
            object: 'model_permission',
            created: 1703186988,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: false,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: '*',
            group: null,
            is_blocking: false,
          },
        ],
      },
    ],
  };
}

/**
 * Mock chat completion object
 * @return {Object}
 */
export function mockChatResponsePayload() {
  return {
    id: 'chat-98c8c60e3fbf4fc49658eddaf447357c',
    object: 'chat.completion',
    created: 1703165682,
    choices: [
      {
        finish_reason: 'stop',
        message: {
          role: 'assistant',
          content: 'What is the best French cheese?',
        },
        index: 0,
      },
    ],
    model: 'mistral-small-latest',
    usage: {prompt_tokens: 90, total_tokens: 90, completion_tokens: 0},
  };
}

/**
 * Mock chat completion stream
 * @return {Object}
 */
export function mockChatResponseStreamingPayload() {
  const encoder = new TextEncoder();
  const firstMessage =
    [encoder.encode('data: ' +
    JSON.stringify({
      id: 'cmpl-8cd9019d21ba490aa6b9740f5d0a883e',
      model: 'mistral-small-latest',
      choices: [
        {
          index: 0,
          delta: {role: 'assistant'},
          finish_reason: null,
        },
      ],
    }) +
    '\n\n')];
  const lastMessage = [encoder.encode('data: [DONE]\n\n')];

  const dataMessages = [];
  for (let i = 0; i < 10; i++) {
    dataMessages.push(encoder.encode(
      'data: ' +
        JSON.stringify({
          id: 'cmpl-8cd9019d21ba490aa6b9740f5d0a883e',
          object: 'chat.completion.chunk',
          created: 1703168544,
          model: 'mistral-small-latest',
          choices: [
            {
              index: i,
              delta: {content: `stream response ${i}`},
              finish_reason: null,
            },
          ],
        }) +
        '\n\n'),
    );
  }

  return firstMessage.concat(dataMessages).concat(lastMessage);
}

/**
 * Mock embeddings response
 * @param {number} batchSize
 * @return {Object}
 */
export function mockEmbeddingResponsePayload(batchSize = 1) {
  return {
    id: 'embd-98c8c60e3fbf4fc49658eddaf447357c',
    object: 'list',
    data:
      [
        {
          object: 'embedding',
          embedding: [-0.018585205078125, 0.027099609375, 0.02587890625],
          index: 0,
        },
      ] * batchSize,
    model: 'mistral-embed',
    usage: {prompt_tokens: 90, total_tokens: 90, completion_tokens: 0},
  };
}

/**
 * Mock embeddings request payload
 * @return {Object}
 */
export function mockEmbeddingRequest() {
  return {
    model: 'mistral-embed',
    input: 'embed',
  };
}
