"""LLM service for question-answering evaluation."""

import os
import re
import time
from typing import Any, Literal

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


# Instructions matching the official FinMME eval script
INSTRUCTIONS = {
    "single_choice": (
        "Please answer this single choice question about the document. "
        "The document content is:\n{markdown}\n\n"
        "Please answer the question directly. "
        "The answer MUST be of the following format: 'Answer: $ANSWER' "
        "(without quotes) where $ANSWER is the answer to the problem "
        "(the single letter of the correct answer, A, B, C, D, etc.)."
    ),
    "multiple_choice": (
        "Please answer this multiple choice question about the document. "
        "The document content is:\n{markdown}\n\n"
        "Please answer the question directly. "
        "The answer MUST be of the following format: 'Answer: $ANSWER' "
        "(without quotes) where $ANSWER is the answer to the problem "
        "(the letter(s) of the correct answer(s), split by ',')."
    ),
    "numerical": (
        "Please answer this numerical question about the document. "
        "The unit of the answer is {unit}. "
        "The document content is:\n{markdown}\n\n"
        "Please answer the question directly. "
        "The answer MUST be of the following format: 'Answer: $ANSWER' "
        "(without quotes) where $ANSWER is the answer to the problem "
        "(digit number only, without unit or any other text)."
    ),
    "free_text": (
        "Please answer this question about the document. "
        "The document content is:\n{markdown}\n\n"
        "Please answer the question directly and concisely. "
        "If the question asks about a checkbox or selection state, answer 'Yes' or 'No'. "
        "If the question asks which items are selected/checked, list only the selected items separated by commas. "
        "The answer MUST be of the following format: 'Answer: $ANSWER' "
        "(without quotes) where $ANSWER is your answer."
    ),
}


def extract_result(res: str, question_type: str = "") -> str:
    """
    Extract answer from response.

    For free_text questions, captures everything after "Answer:" to end of line
    (supporting multi-word answers). For other types, captures only the first token
    (matching official FinMME eval behavior).

    :param res: Response string from LLM
    :param question_type: Question type (used to select extraction strategy)
    :return: Extracted answer string
    """
    if not res:
        return ""
    if question_type == "free_text":
        # Capture everything after "Answer:" to end of line
        match = re.search(r"(?i)Answer\s*:\s*(.+?)(?:\n|$)", res)
        return match.group(1).strip() if match else ""
    else:
        # Original FinMME behavior: capture first token only
        match = re.search(r"(?i)Answer\s*:\s*([^\s\n]+)", res)
        return match.group(1) if match else ""


def normalize_response(response: str) -> str:
    """
    Normalize response using the same logic as official FinMME eval.

    :param response: Raw response string
    :return: Normalized response string
    """
    return (
        response.replace("**", "")
        .replace(":", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


class QALLMService:
    """Service for calling LLM to answer questions based on markdown content."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5-mini",
        provider: Literal["openai", "google"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize the QA LLM service.

        :param api_key: API key (default: from OPENAI_API_KEY or GOOGLE_GENAI_API_KEY env var)
        :param model: Model name to use (default: "gpt-5-2025-08-07")
        :param provider: Provider to use ("openai" or "google").
            If None, auto-detect from model name
        :param temperature: Temperature for generation (default: 0.0 for deterministic).
            Some models may not support this parameter
        :param max_tokens: Maximum tokens in response (default: 512).
            Some models may not support this parameter
        :param max_retries: Maximum number of retry attempts (default: 3)
        :param retry_delay: Base delay between retries in seconds (default: 2.0)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Determine provider
        if provider is None:
            # Auto-detect from model name
            if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3-"):
                provider = "openai"
            elif model.startswith("gemini-") or "gemini" in model.lower():
                provider = "google"
            else:
                # Default to OpenAI for unknown models
                provider = "openai"

        self.provider = provider

        # Determine if model supports temperature and max_tokens parameters
        # Some models (like certain GPT-5 variants) may not support these
        self._supports_temperature = self._model_supports_temperature(model)
        self._supports_max_tokens = self._model_supports_max_tokens(model)

        # Initialize client based on provider
        if provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package is required for OpenAI provider. Install it with: pip install openai")
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for OpenAI provider. "
                    "Set it or pass api_key parameter."
                )
            self.client = OpenAI(api_key=api_key)
        elif provider == "google":
            if genai is None:
                raise ImportError(
                    "google-genai package is required for Google provider. Install it with: pip install google-genai"
                )
            if api_key is None:
                api_key = os.getenv("GOOGLE_GENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_GENAI_API_KEY environment variable is required for Google provider. "
                    "Set it or pass api_key parameter."
                )
            self.client = genai.Client(api_key=api_key)  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def answer_question(
        self,
        markdown: str,
        question: str,
        question_type: str,
        options: str = "",
        unit: str = "",
    ) -> str:
        """
        Answer a question based on markdown content, following the official FinMME format.

        :param markdown: Markdown content to use as context
            (replaces verified_caption/related_sentences)
        :param question: Question to answer
        :param question_type: Type of question ("single_choice", "multiple_choice", or "numerical")
        :param options: Options string for choice questions (default: "")
        :param unit: Unit string for numerical questions (default: "")
        :return: Predicted answer string (extracted and normalized)
        :raises RuntimeError: If API call fails after retries
        """
        # Get question-type-specific instruction template
        if question_type not in INSTRUCTIONS:
            # Fallback to multiple_choice if unknown
            question_type = "multiple_choice"

        # Build prompt following official FinMME format
        if question_type == "numerical":
            prompt = INSTRUCTIONS[question_type].format(unit=unit, markdown=markdown)
            prompt += f"\nQuestion: {question}"
        else:
            prompt = INSTRUCTIONS[question_type].format(markdown=markdown)
            prompt += f"\nQuestion: {question}"
            if options:
                prompt += f"\nOptions: {options}"

        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response_text = self._call_openai(prompt)
                elif self.provider == "google":
                    response_text = self._call_google(prompt)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

                if response_text:
                    # Extract and normalize answer
                    extracted = extract_result(response_text, question_type)
                    if extracted:
                        normalized = normalize_response(extracted)
                        return normalized.strip()

                    # If extraction failed, try normalizing the whole response
                    normalized = normalize_response(response_text)
                    return normalized.strip()

                # If no text in response, try again
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue

                raise RuntimeError("Empty response from LLM")

            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a retryable error
                is_retryable = any(
                    keyword in error_str for keyword in ["503", "overloaded", "rate limit", "timeout", "429"]
                )

                if is_retryable and attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)
                    continue

                # If not retryable or max retries reached, raise
                raise RuntimeError(f"Failed to get answer from LLM: {e}") from e

        raise RuntimeError(f"Failed to get answer after {self.max_retries} attempts")

    def _model_supports_temperature(self, model: str) -> bool:
        """
        Check if the model supports temperature parameter.

        GPT-5 models do not support temperature parameter.

        :param model: Model name
        :return: True if model supports temperature, False otherwise
        """
        # GPT-5 models don't support temperature
        if model.startswith("gpt-5"):
            return False
        return True

    def _model_supports_max_tokens(self, model: str) -> bool:
        """
        Check if the model supports max_tokens parameter.

        GPT-5 models do not support max_tokens parameter.

        :param model: Model name
        :return: True if model supports max_tokens, False otherwise
        """
        # GPT-5 models don't support max_tokens
        if model.startswith("gpt-5"):
            return False
        return True

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        # Build request parameters conditionally based on model support
        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        # Only include temperature if model supports it
        if self._supports_temperature:
            request_params["temperature"] = self.temperature

        # Only include max_tokens if model supports it
        if self._supports_max_tokens:
            request_params["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**request_params)

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content  # type: ignore[no-any-return]

        return ""

    def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API."""
        contents = [types.Content(parts=[types.Part.from_text(text=prompt)])]

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        response = self.client.models.generate_content(  # type: ignore[attr-defined]
            model=self.model,
            contents=contents,
            config=config,
        )

        if response.text:
            return response.text  # type: ignore[no-any-return]

        return ""
