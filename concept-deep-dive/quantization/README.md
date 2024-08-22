# Quantization

**Quantization** is a process that plays a crucial role in Large Language Models. These vast neural networks are primarily composed of raw values, known as `weights` or parameters. The higher the number of parameters, the larger the model, and the more knowledge it can extract and retain, but also the more **resources** it requires to run them. These big models can range from a few million parameters to billions.

## Inference and Resource Requirements

When we want to run an LLM, also called **Inference**, we need to load the entire model, including all of its parameters, to perform the necessary computations. This requirement scales linearly with RAM/VRAM, meaning that **larger models necessitate more memory**.

## The Role of Quantization

This is where quantization comes into play. The parameters that store all the knowledge and decision-making capabilities of our models are stored in a certain number of bits. For example, the models we have open-sourced are usually in BF16, stored within 16 bits of precision.

Here's a simplified representation of how a value in BF16 is stored:

| Bit Position | 15 | 14-7 | 6-0 |
|--------------|:--:|:--:|:--:|
| Component    | sign| exponent | fraction |

The goal of quantization is to reduce the precision required for each parameter, or weight, without significantly impacting the model's performance. This is not always a simple truncation process. There are different methods of quantization that aim to minimize the impact of reducing precision.

By quantizing the model's values to use fewer bits on average (such as 16 bits, 8 bits, 6 bits, 4 bits, or even lower), we can store the models more efficiently on disk and in memory. This makes them more accessible and less resource-intensive to run.

## Estimate Memory Requirements

One might wonder how to estimate how much memory will be required and how to calculate it. We can estimate by using this simple formula:
- required_bytes = n_parameters * bytes_per_parameter

Let's apply this formula to different data types for comparison for our 7.3B model!

| Data Type | Bytes | Range | N° Different Values | Memory |
|--------------|:--:|:--:|:--:|:--:|
| FP32   | 4 | -1.18e38 : 3.4e38 | 2^32 | 29.2 GB |
| FP16   | 2 | -65k : 65k | 2^16 | 14.6 GB |
| BF16   | 2 | -3.39e38 : 3.39e38 | 2^16 | 14.6 GB |
| FP8 (E5M2) | 1 | -57k : 57k | 256 | 7.3 GB |
| INT8   | 1 | -128 : 127 | 256 | 7.3 GB |
| INT4   | 0.5 | -8 : 7 | 16 | 3.65 GB |

Crucial to note that this is only the memory for loading inference, without taking into account the context size (sequence length) and computation.

## Mistral Models

Making use of the same previous formula, lets compute a rough estimate for the required Memory for different data types for each model.

| Model | Params | BF16 | FP8 | INT4 |
|--------------|:--:|:--:|:--:|:--:|
| Mistral 7B   | 7.3B | 14.6 GB | 7.3 GB | 3.65 GB |
| Mathstral 7B | 7.3B | 14.6 GB | 7.3 GB | 3.65 GB |
| Codestral Mamba 7B   | 7.3B | 14.6 GB | 7.3 GB | 3.65 GB |
| Mistral Nemo 12B | 12.2B | 24.4 GB | 12.2 GB | 6.1 GB |
| Mixtral 8x7B   | 46.7B | 93.4 GB | 46.7 GB | 23.35 GB |
| Codestral 22B | 22.2B | 44.4 GB | 22.2 GB | 11.1 GB |
| Mixtral 8x22B   | 140.6B | 281.2 GB | 140.6 GB | 70.3 GB |
| Mistral Large 2407  | 123B | 246 GB | 123 GB | 61.5 GB |

## Quantization Formats

While it is common practice to release weights in BF16 or FP16 due to hardware optimized for inference at that precision, the community has favored 8 bits as the precision loss is minimal or even null. And special techniques such as Training with Quantization Awareness can allow inference at FP8 losslessly! See Mistral Nemo 12B.

As mentioned previously, quantization is often not a simple truncation to fit in fewer bits. There are a lot of different formats and precisions.

Among them we have:

<details>
<summary><b>Bits-and-bytes</b></summary>

<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/concept-deep-dive/quantization/methods/bnb.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Bits-and-bytes is a very fast and straightforward approach to quantization, quantizing while loading. However, speed and quality are not optimal, useful for quick quantization and loading of models.
</details>

<details>
<summary><b>GGUF</b></summary>

<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/concept-deep-dive/quantization/methods/gguf.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Previously GGML, GGUF is favored by a lot of the community for its ability to run efficiently on CPU and Apple devices, offloading to a GPU if available! Making it a good choice for local testing and deployment.
</details>

<details>
<summary><b>GPTQ</b></summary>

<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/concept-deep-dive/quantization/methods/gptq.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

While GGUF focuses on CPU, GPTQ is oriented towards GPU inference performance by reducing errors with a calibration dataset.
</details>

<details>
<summary><b>AWQ</b></summary>

<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/concept-deep-dive/quantization/methods/awq.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

AWQ is also oriented towards GPU, it bases itself on the fact that ~1% of weights actually contribute significantly to the model's accuracy, and hence these must be treated delicately by using a dataset to analyze the activation distributions during inference and identify those important and critical weights.
</details>

<details>
<summary><b>EXL2</b></summary>

<a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/concept-deep-dive/quantization/methods/exl2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

A more recent format based on the GPTQ optimization method but with mixed quantization levels. It achieves an average desired bitrate with smaller errors than GPTQ while keeping the same or similar bitrate. Can have a slightly higher VRAM usage but better inference speed and quality.
</details>

*Other formats: HQQ, AQLM, EETQ, Marlin...*

Among these formats, the most popular and widely used precisions are between 4 bits and 8 bits. While 8 bits has the best accuracy, 6 bits still keeps a lot of the smarts of the model without degrading the quality too much. And only below 4 bits of precision do we start to see a very high impact on most models, degrading their abilities considerably. While there are still use cases for models with lower bitrates than 4 bits, their quality can vary immensely on the models, optimization methods, and formats.

## Inference & Quantized Models

Not all inference engines support all possible formats; some are highly specialized and optimized for specific formats, while others aim to generalize and support all kinds.  


Among these engines, we have:
- **[VLLM](https://github.com/vllm-project/vllm)**: One of the oldest and most standard engines, supporting GPTQ, AWQ, INT4, INT8, and FP8.
- **[Transformers](https://github.com/huggingface/transformers)**: A popular library for running LLMs, supporting GPTQ, AWQ, bnb, EETQ, AQLM, torchao, HQQ, FP8 and [more](https://huggingface.co/docs/transformers/main/en/quantization/overview).
- **[Exllamav2](https://github.com/turboderp/exllamav2)**: Mostly for GPTQ and EXLV2 formats.
- **[Text Generation Inference](https://github.com/huggingface/text-generation-inference)**: A production ready inference and deployment library for large language models, supporting GPTQ, AWQ, bnb, EXL2, FP8, Marlin, Medusa (Faster Decoding) and [more](https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization).
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)**/**[ollama](https://github.com/ollama/ollama)**: Good options for GGUF inference.
- **[Aphrodite](https://github.com/PygmalionAI/aphrodite-engine)**: A big generalized engine for production with support for AWQ, Bitsandbytes, EXL2, GGUF, GPTQ, and many others.
