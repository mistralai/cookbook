declare module "@mistralai/mistralai" {
  export interface ModelPermission {
    id: string;
    object: "model_permission";
    created: number;
    allow_create_engine: boolean;
    allow_sampling: boolean;
    allow_logprobs: boolean;
    allow_search_indices: boolean;
    allow_view: boolean;
    allow_fine_tuning: boolean;
    organization: string;
    group: string | null;
    is_blocking: boolean;
  }

  export interface Model {
    id: string;
    object: "model";
    created: number;
    owned_by: string;
    root: string | null;
    parent: string | null;
    permission: ModelPermission[];
  }

  export interface ListModelsResponse {
    object: "list";
    data: Model[];
  }

  export interface Function {
    name: string;
    description: string;
    parameters: object;
  }

  export interface FunctionCall {
    name: string;
    arguments: string;
  }

  export interface ToolCalls {
    id: string;
    function: FunctionCall;
  }

  export interface ResponseFormat {
    type: "json_object";
  }

  export interface TokenUsage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  }

  export interface ChatCompletionResponseChoice {
    index: number;
    message: {
      role: string;
      content: string;
      tool_calls: null | ToolCalls[];
    };
    finish_reason: string;
  }

  export interface ChatCompletionResponseChunkChoice {
    index: number;
    delta: {
      role?: string;
      content?: string;
      tool_calls?: ToolCalls[];
    };
    finish_reason: string;
  }

  export interface ChatCompletionResponse {
    id: string;
    object: "chat.completion";
    created: number;
    model: string;
    choices: ChatCompletionResponseChoice[];
    usage: TokenUsage;
  }

  export interface ChatCompletionResponseChunk {
    id: string;
    object: "chat.completion.chunk";
    created: number;
    model: string;
    choices: ChatCompletionResponseChunkChoice[];
    usage: TokenUsage | null;
  }

  export interface Embedding {
    id: string;
    object: "embedding";
    embedding: number[];
  }

  export interface EmbeddingResponse {
    id: string;
    object: "list";
    data: Embedding[];
    model: string;
    usage: TokenUsage;
  }

  export type Message =
    | {
        role: "system" | "user" | "assistant";
        content: string | string[];
      }
    | {
        role: "tool";
        content: string | string[];
        name: string;
        tool_call_id: string;
      };

  export interface Tool {
    type: "function";
    function: Function;
  }

  export interface ChatRequest {
    model: string;
    messages: Array<Message>;
    tools?: Array<Tool>;
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    randomSeed?: number;
    /**
     * @deprecated use safePrompt instead
     */
    safeMode?: boolean;
    safePrompt?: boolean;
    toolChoice?: "auto" | "any" | "none";
    responseFormat?: ResponseFormat;
  }

  export interface CompletionRequest {
    model: string;
    prompt: string;
    suffix?: string;
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    randomSeed?: number;
    stop?: string | string[];
  }

  export interface ChatRequestOptions {
    signal?: AbortSignal;
  }

  class MistralClient {
    apiKey: string;
    endpoint: string;
    maxRetries: number;
    timeout: number;

    constructor(
      apiKey?: string,
      endpoint?: string,
      maxRetries?: number,
      timeout?: number
    );

    listModels(): Promise<ListModelsResponse>;

    chat(
      request: ChatRequest,
      options?: ChatRequestOptions
    ): Promise<ChatCompletionResponse>;

    chatStream(
      request: ChatRequest,
      options?: ChatRequestOptions
    ): AsyncGenerator<ChatCompletionResponseChunk, void>;

    completion(
        request: CompletionRequest,
        options?: ChatRequestOptions
    ): Promise<ChatCompletionResponse>;

    completionStream(
        request: CompletionRequest,
        options?: ChatRequestOptions
    ): AsyncGenerator<ChatCompletionResponseChunk, void>;


    embeddings(options: {
      model: string;
      input: string | string[];
    }): Promise<EmbeddingResponse>;
  }

  export default MistralClient;
}
