# Portkey + Mistral Integration

This cookbook demonstrates how to effectively use Mistral models through Portkey's AI Gateway. Learn how to integrate Mistral AI with enterprise-grade reliability, observability, and performance features.

<img width="400" src="https://raw.githubusercontent.com/siddharthsambharia-portkey/Portkey-Product-Images/main/full-white-text.png" alt="portkey">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16Z1AlAsW_d_mB5UK1EbailGt4laUU7E9?usp=sharing)



## Features

- 🚀 Unified API Gateway for Mistral models
- 📊 Full-stack observability with 40+ metrics
- 💾 Semantic & simple caching strategies
- 🔄 Automated retries & model fallbacks
- 🛡️ AI Guardrails & content safety controls

## Getting Started

1. Install the required packages:
```bash
pip install portkey-ai
```

2. Set up your API keys in environment variables:
```python
export PORTKEY_API_KEY="your_portkey_key"
export MISTRAL_VIRTUAL_KEY="your_mistral_key"  # Get from Portkey dashboard
```

## Examples

The cookook in the same folder contains several examples demonstrating different features:

- Simple chat completion setup
- Load balancing and Fallback deployments
- Metadata tracking and request tracing
- Semantic caching implementation


## Usage

Basic chat completion with Mistral:
```python
from portkey_ai import Portkey

portkey = Portkey(
    api_key="your_portkey_key",
    virtual_key="your_mistral_key"
)

completion = portkey.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model="mistral-large-latest"
)
print(completion.choices[0].message.content)
```

## Features Overview

### Production-Grade Controls
- **Load Balancing**: Distribute traffic between Mistral models/versions
- **Semantic Retries**: Auto-rewrite failed prompts for better results
- **Guardrails**: Prevent PII leaks, block prompt injections
- **Virtual Keys**: Secure API key management with budget controls

### Enterprise Observability
- Real-time cost tracking
- Custom metadata tagging
- Request tracing with Trace IDs
- Audit logs with full history

### Team Collaboration
- Version-controlled prompt library
- Shared routing configurations
- RBAC permissions
- Central monitoring dashboard

## Support

- [Portkey AI Mistral Documentation](https://portkey.sh/mistral)
- [Portkey Wesbite](https://portkey.ai/)
- [Join Portkey Discord](https://discord.gg/SqX9epQKNR)
