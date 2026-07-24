# Concept Deep Dive: An Overview of Prompt Optimization

Early successes with prompt-based learning demonstrated that the phrasing of tasks can drastically affect a language model (LLM)'s performance. In their seminal work, Brown et al. (2020) argue that carefully crafted text prompts could induce LLMs to perform new tasks without any parameter update. Together with empirical findings related to the effectiveness of LLMs across a wide variety of tasks, this spurred interest in prompt engineering. Yet, designing prompts by hand proved to be a cumbersome and error-prone effort, requiring expertise and extensive trial-and-error (Liu et al., 2023).

One line of early work focused on automated search for prompts in a continuous setting. In particular, Shin et al. (2020) introduced *AutoPrompt*, a gradient-guided search algorithm to find input trigger phrases that coax models into desired behaviors.

Later, Ma et al. (2023) followed with a discrete approach to prompt optimization, automatically generating prompts by mining corpora and paraphrasing existing prompts.

These approaches demonstrated that automatic prompt discovery is feasible and effective. Gradient-based prompt tuning methods have also emerged, including techniques such as prefix-tuning (Li et al., 2021) and prompt tuning (Lester et al., 2021). Both approaches learn continuous representations (prompt embeddings) prepended to inputs, updating only a small number of parameters and leaving the main model unchanged at inference time.

A gradient-free approach which does not require access to the model internals—it only operates on the input tokens given to the model—is RLPrompt (Deng et al., 2022), which formulates prompt design as a reinforcement learning (RL) problem.

These techniques are suited for scenarios where model internals are inaccessible and resulted in unintuitive yet effective prompt strings.

# A Taxonomy of Prompt Optimization

## Retrieval-Augmented Prompting

Retrieval-augmented prompting incorporates external information (e.g., documents, examples) into the prompt. This includes Retrieval-Augmented Generation (RAG), which appends retrieved context to the prompt (Lewis et al., 2020), and example-based prompting, which selects demonstrations from a dataset (Rubin et al., 2021). While effective, these methods depend on retrieval infrastructure as well as correctly-labeled data, hindering adoption in cases where either of these assumptions is violated.

## Continuous Prompt Tuning

Continuous prompt tuning methods represent prompts as trainable vectors. Clearly, this assumes having access to the model internals, as these trainable vectors are then fed to the transformer layers for conditional generation. Trainable vectors can either be changed partially (*prefix tuning*) or entirely (*prompt tuning*).
Prefix-tuning (Li et al., 2021) and prompt tuning (Lester et al., 2021) perform prompt optimization by first embedding the candidate prompt into a vector, and then updating it, enabling efficient adaptation. These methods are gradient-based and assume access to model internals, making them non-applicable to API-only models.

## Black-Box Discrete Optimization

Black-box techniques, such as manual search, evolutionary strategies, and reinforcement learning, work without gradient access and are thus equally usable for both open and closed-weights models.
In particular, RLPrompt (Deng et al., 2022) uses a policy network trained via reinforcement learning (RL) to generate prompts based on reward signals. Though effective, these methods are often sample-inefficient and computationally expensive.

### Diving into Self-Supervised Prompt Optimization (SPO)

In their work, Xiang et al. (2025) propose Self-Supervised Prompt Optimization (SPO), which *(i)* eliminates the need for ground truth by using the LLM to evaluate its own outputs and *(ii)* combines larger and smaller models in a self-loop for greater sample efficiency and speed, optimizing prompts using as little as three examples.

SPO runs an iterative loop where model(s) act as a prompt executor (carrying out a task given a prompt \( p_k, \phi(p_k) \in \mathcal{T} \)), evaluator (mapping the model's output to a scalar metric of performance \( e: \mathcal{T} \mapsto \mathbb{R} , e(\phi(p_k)) \in \mathbb{R} \)), and optimizer (optimizing \( p_{k+1} \) with \( e(p_{k+1}) \geq e(p_k) \)).
Through pairwise comparisons between subsequent prompts and refinement, SPO improves prompts \( p_k \) without external supervision.
SPO proved highly efficient and applicable to both open- and closed-ended tasks. Unlike gradient-based or RL methods, it does not require labeled data or model internals. It is well-suited for black-box settings and maintains interpretability since it works in natural language space. Further, SPO occupies a promising spot in the PO landscape, combining the strengths of black-box and self-supervised learning while minimizing their limitations, it points towards the promising direction of enabling models to autonomously optimize their own prompts using internal feedback loops.

# Insights on Prompt Optimization

**Sample Efficiency** Gradient-based methods tend to be more sample-efficient than black-box optimization, though recent work like SPO (Xiang et al., 2025) demonstrates efficient optimization using only a few samples per iteration. Sample efficiency is crucial for many applications, especially when concerned with larger models/slower inference settings.

**Dependence on Labeled Data** Continuous tuning and RL often require labeled data, while methods like SPO avoid this by relying on model self-evaluation. In this, SPO effectively allows for prompt optimization without relying on previously collected data, and is thus preferable when limited information is available to optimize prompts differently.

**Generalization and Transfer** Most prompt optimization techniques tend to be task-specific in practice. Some prompts (e.g., “Let's think step by step”) generalize better. RL-discovered prompts sometimes transfer across models (Deng et al., 2022).

**Robustness** Prompt performance can be brittle with respect to phrasing. Continuous prompts are less interpretable, and discrete ones may be sensitive to small changes. Methods like prompt ensembling help mitigate this.

**Computational Cost** Manual prompt engineering is computationally cheap, but proves brittle and labor-intensive. Gradient-based tuning proves to be more compute-efficient, while RL and search-based methods are costly due to the number of required queries of the method to iterate and improve on the prompt. SPO-like approaches seem promising due to *(i)* limited number of queries and *(ii)* disuse of model internals at test-time.

### References
```bash
@article{brown2020language,
  title={Language models are few-shot learners},
  author={Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={1877--1901},
  year={2020}
}

@article{liu2023pre,
  title={Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing},
  author={Liu, Pengfei and Yuan, Weizhe and Fu, Jinlan and Jiang, Zhengbao and Hayashi, Hiroaki and Neubig, Graham},
  journal={ACM computing surveys},
  volume={55},
  number={9},
  pages={1--35},
  year={2023},
  publisher={ACM New York, NY}
}

@article{ma2023prompt,
  title={Is prompt-based finetuning always better than vanilla finetuning? insights from cross-lingual language understanding},
  author={Ma, Bolei and Nie, Ercong and Schmid, Helmut and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2307.07880},
  year={2023}
}

@article{shin2020autoprompt,
  title={Autoprompt: Eliciting knowledge from language models with automatically generated prompts},
  author={Shin, Taylor and Razeghi, Yasaman and Logan IV, Robert L and Wallace, Eric and Singh, Sameer},
  journal={arXiv preprint arXiv:2010.15980},
  year={2020}
}

@article{li2021prefix,
  title={Prefix-tuning: Optimizing continuous prompts for generation},
  author={Li, Xiang Lisa and Liang, Percy},
  journal={arXiv preprint arXiv:2101.00190},
  year={2021}
}

@article{lester2021power,
  title={The power of scale for parameter-efficient prompt tuning},
  author={Lester, Brian and Al-Rfou, Rami and Constant, Noah},
  journal={arXiv preprint arXiv:2104.08691},
  year={2021}
}

@article{deng2022rlprompt,
  title={Rlprompt: Optimizing discrete text prompts with reinforcement learning},
  author={Deng, Mingkai and Wang, Jianyu and Hsieh, Cheng-Ping and Wang, Yihan and Guo, Han and Shu, Tianmin and Song, Meng and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2205.12548},
  year={2022}
}

@article{rubin2021learning,
  title={Learning to retrieve prompts for in-context learning},
  author={Rubin, Ohad and Herzig, Jonathan and Berant, Jonathan},
  journal={arXiv preprint arXiv:2112.08633},
  year={2021}
}

@article{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and others},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}

@article{xiang2025self,
  title={Self-Supervised Prompt Optimization},
  author={Xiang, Jinyu and Zhang, Jiayi and Yu, Zhaoyang and Teng, Fengwei and Tu, Jinhao and Liang, Xinbing and Hong, Sirui and Wu, Chenglin and Luo, Yuyu},
  journal={arXiv preprint arXiv:2502.06855},
  year={2025}
}

```
