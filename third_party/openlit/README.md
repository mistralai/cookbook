![](https://github.com/openlit/.github/blob/main/profile/assets/wide-logo-no-bg.png?raw=true)

# Analyze your Mistral AI Models with OpenLIT

## What is OpenLIT?
[**OpenLIT**](https://github.com/openlit/openlit) is an an open source project that helps developers build and manage AI apps in production, effectively helping them improve accuracy. As a self-hosted solution, it enables developers to experiment with LLMs, manage and version prompts, securely manage API keys, and provide safeguards against prompt injection and jailbreak attempts. It also includes built-in **OpenTelemetry-native** observability and evaluation for the complete GenAI stack (LLMs, vector databases, Agents and GPUs).

## Why use Tracing to gain Observability into an LLM Application?
- Capture the complete context of execution, including API calls, context, prompts, parallelism, and more
- Monitor model usage and associated costs
- Gather user feedback effectively
- Detect and identify low-quality outputs

## OpenLIT and Mistral AI Integration Cookbooks
These guides offer detailed instructions for integrating OpenLIT with Mistral AI using Python. By following these steps, you will learn how to effectively analyze and trace interactions with Mistral's language models, improving the transparency, debuggability, and performance monitoring of your AI-powered applications.

| Guides | Description |
|-------|-------------|
| [1. Cookbook: Monitoring Mistral AI with OpenTelemetry (Python)](cookbook_mistral_opentelemetry.ipynb) | This cookbook will cover the process of integrating OpenLIT with the Mistral SDK. A straightforward guide demonstrates how adding a single line of code can seamlessly enable OpenLIT to track various metrics, including cost, tokens, prompts, responses, and all chat/completion activities from the Mistral SDK using OpenTelemetry.|

## Feedback and Community
If you have any feedback or requests, please create a GitHub [Issue](https://github.com/openlit/openlit/issues) or share your idea with the community on [Slack](https://join.slack.com/t/openlit/shared_invite/zt-2etnfttwg-TjP_7BZXfYg84oAukY8QRQ).
