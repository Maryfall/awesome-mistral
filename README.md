# Awesome Mistral [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Last Updated](https://img.shields.io/github/last-commit/samouraiworld/awesome-mistral)

> A curated list of awesome resources, tools, libraries, and projects for the Mistral AI ecosystem.

Mistral AI is a Paris-based AI company building open-weight, high-performance large language models. Founded in 2023, Mistral has quickly become a leading force in open-source AI, offering models that rival proprietary alternatives while remaining accessible to developers worldwide.

This repository maps and curates the entire Mistral.ai ecosystem for AI engineers, researchers, startup founders, and open-source contributors.

**Legend:**
- 🧠 Official Mistral AI
- 🌍 Community project
- 🧪 Experimental

---

## Contents

- [Why Mistral?](#why-mistral)
- [Official Mistral Resources](#official-mistral-resources)
- [Models](#models)
- [Community Fine-Tuned Models](#community-fine-tuned-models)
- [SDKs & APIs](#sdks--apis)
- [Inference & Deployment](#inference--deployment)
- [Fine-Tuning & Training](#fine-tuning--training)
- [Model Merging & Quantization](#model-merging--quantization)
- [Agents & Orchestration](#agents--orchestration)
- [Tooling & Dev Experience](#tooling--dev-experience)
- [Community Projects](#community-projects)
- [Demos & Examples](#demos--examples)
- [Tutorials & Guides](#tutorials--guides)
- [Benchmarks & Evaluation](#benchmarks--evaluation)
- [Research & Papers](#research--papers)
- [Talks & Media](#talks--media)
- [Ecosystem & Community](#ecosystem--community)
- [Contributing](#contributing)
- [License](#license)

---

## Why Mistral?

Mistral AI offers a compelling alternative in the LLM landscape:

| Aspect | Mistral Advantage |
|--------|-------------------|
| **Open Weights** | Models like Mistral 7B and Mixtral are fully open-weight, enabling local deployment, fine-tuning, and full control |
| **Efficiency** | Mistral 7B outperforms Llama 2 13B; Mixtral 8x7B matches GPT-3.5 with only 12.9B active parameters |
| **European Sovereignty** | Paris-based company offering GDPR-compliant, EU-hosted API options |
| **Cost Efficiency** | Competitive API pricing; open models enable free self-hosting |
| **Innovation** | Pioneered efficient MoE architectures and sliding window attention in open models |

---

## Official Mistral Resources

- 🧠 [Mistral AI](https://mistral.ai) – Official company website with product information and announcements.
- 🧠 [Mistral AI Documentation](https://docs.mistral.ai) – Comprehensive API documentation, guides, and model specifications.
- 🧠 [Mistral AI Console](https://console.mistral.ai) – Web interface for API key management and model access.
- 🧠 [Mistral AI GitHub](https://github.com/mistralai) – Official GitHub organization with 22+ repositories.
- 🧠 [mistral-inference](https://github.com/mistralai/mistral-inference) ⭐ 10k+ – Official inference library for running Mistral models.
- 🧠 [mistral-finetune](https://github.com/mistralai/mistral-finetune) ⭐ 3k+ – Official lightweight LoRA-based fine-tuning library.
- 🧠 [Mistral Cookbook](https://github.com/mistralai/cookbook) ⭐ 2k+ – Official notebooks and examples for common use cases.
- 🧠 [mistral-common](https://github.com/mistralai/mistral-common) – Official tokenization and pre-processing library.
- 🧠 [Platform Docs Public](https://github.com/mistralai/platform-docs-public) – Open-source documentation repository.

---

## Models

### Flagship Models (API)

| Model | Parameters | Context | Best For |
|-------|------------|---------|----------|
| **Mistral Large** | 123B | 128k | Complex reasoning, multilingual, code generation |
| **Mistral Medium** | — | 32k | Balanced performance-to-cost ratio |
| **Mistral Small** | 24B | 128k | Low-latency, cost-sensitive applications |

### Open-Weight Models

- 🧠 [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) – Compact 7B model outperforming Llama 2 13B on most benchmarks.
- 🧠 [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) – Latest instruction-tuned variant with function calling.
- 🧠 [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) – Sparse MoE with 46.7B total / 12.9B active parameters.
- 🧠 [Mixtral 8x7B Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) – Instruction-tuned MoE variant.
- 🧠 [Mixtral 8x22B](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1) – Large-scale MoE with 141B total / 39B active parameters.

### Specialized Models

- 🧠 **Codestral** – Code-specialized model for 80+ programming languages.
- 🧠 **Devstral** – Developer-focused model for coding assistance and software development.
- 🧠 **Pixtral** – Multimodal model with vision capabilities.
- 🧠 **Mathstral** – Mathematics-specialized for reasoning and problem-solving.

---

## Community Fine-Tuned Models

High-quality community fine-tunes built on Mistral base models:

### Instruction & Chat

- 🌍 [OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) – GPT-4 quality instruction-tuned by Teknium.
- 🌍 [Zephyr-7B-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) – DPO-trained by HuggingFace H4, outperforms 70B on MT-Bench.
- 🌍 [Nous-Hermes-2-Mistral-7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO) – DPO-enhanced with strong benchmark scores.
- 🌍 [Hermes-2-Pro-Mistral-7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) – Function calling and JSON mode specialist.
- 🌍 [OpenChat-3.5-0106](https://huggingface.co/openchat/openchat-3.5-0106) – C-RLFT trained, ChatGPT-comparable performance.
- 🌍 [Dolphin-2.8-Mistral-7B](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02) – Uncensored model by Eric Hartford.

### Specialized

- 🌍 [MistralLite](https://huggingface.co/amazon/MistralLite) – AWS-optimized with 32k context window.
- 🌍 [Mistral-7B-OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca) – Trained on OpenOrca dataset.
- 🌍 [WizardMath-7B-V1.1](https://huggingface.co/WizardLM/WizardMath-7B-V1.1) – Math-specialized Mistral fine-tune.

### Quantized Model Collections

- 🌍 [TheBloke](https://huggingface.co/TheBloke) – Extensive GGUF/AWQ/GPTQ quantized model repository.
- 🌍 [bartowski](https://huggingface.co/bartowski) – High-quality GGUF quantizations.

---

## SDKs & APIs

### Official SDKs

- 🧠 [client-python](https://github.com/mistralai/client-python) – Official Python client library.
- 🧠 [@mistralai/mistralai](https://www.npmjs.com/package/@mistralai/mistralai) – Official TypeScript/JavaScript SDK.

### Community SDKs

- 🌍 [mistral.rs](https://github.com/EricLBuehler/mistral.rs) – Blazingly fast Rust inference with ISQ, LoRA, quantization.
- 🌍 [mistral-go](https://github.com/Gage-Technologies/mistral-go) – Go client for Mistral AI API.
- 🌍 [@ai-sdk/mistral](https://www.npmjs.com/package/@ai-sdk/mistral) – Vercel AI SDK provider.
- 🌍 [@langchain/mistralai](https://www.npmjs.com/package/@langchain/mistralai) – LangChain.js integration.

### Official Libraries

- 🧠 [mistral-common](https://github.com/mistralai/mistral-common) – Tokenization and pre-processing.
- 🧠 [mistral-vibe](https://github.com/mistralai/mistral-vibe) ⭐ 2.5k+ – Minimal CLI coding agent.

---

## Inference & Deployment

### High-Performance Inference

- 🌍 [vLLM](https://github.com/vllm-project/vllm) ⭐ 35k+ – High-throughput with PagedAttention. Excellent Mistral support.
- 🌍 [Text Generation Inference](https://github.com/huggingface/text-generation-inference) – Hugging Face's production inference server.
- 🌍 [llama.cpp](https://github.com/ggerganov/llama.cpp) ⭐ 70k+ – CPU/GPU inference with GGUF quantization.
- 🌍 [ExLlamaV2](https://github.com/turboderp/exllamav2) – Fast inference with EXL2 quantization.
- 🌍 [SGLang](https://github.com/sgl-project/sglang) – Fast serving with RadixAttention.

### Local Inference

- 🌍 [Ollama](https://ollama.com) ⭐ 100k+ – Simple CLI for local Mistral models.
- 🌍 [LM Studio](https://lmstudio.ai) – Desktop GUI for local LLMs.
- 🌍 [Jan](https://jan.ai) – Open-source ChatGPT alternative running locally.
- 🌍 [GPT4All](https://gpt4all.io) – Local inference with Mistral support.
- 🌍 [Msty](https://msty.app) – Desktop app for running local LLMs.

### Cloud & Container Deployment

- 🌍 [LocalAI](https://github.com/mudler/LocalAI) ⭐ 25k+ – OpenAI-compatible local API server.
- 🌍 [SkyPilot](https://github.com/skypilot-org/skypilot) – Run on any cloud with cost optimization.
- 🧪 [MLC LLM](https://github.com/mlc-ai/mlc-llm) – Universal deployment across hardware backends.

---

## Fine-Tuning & Training

### Fine-Tuning Frameworks

- 🧠 [mistral-finetune](https://github.com/mistralai/mistral-finetune) – Official LoRA fine-tuning library.
- 🌍 [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) – Streamlined LoRA/QLoRA/full fine-tuning.
- 🌍 [Unsloth](https://github.com/unslothai/unsloth) ⭐ 20k+ – 2-5x faster fine-tuning, 80% less memory.
- 🌍 [Hugging Face PEFT](https://github.com/huggingface/peft) – Parameter-Efficient Fine-Tuning.
- 🌍 [Hugging Face TRL](https://github.com/huggingface/trl) – RLHF and DPO training.
- 🌍 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) ⭐ 35k+ – Unified fine-tuning framework.
- 🌍 [torchtune](https://github.com/pytorch/torchtune) – PyTorch-native fine-tuning.

### Training Infrastructure

- 🌍 [DeepSpeed](https://github.com/microsoft/DeepSpeed) – Distributed training optimization.
- 🌍 [Hugging Face Accelerate](https://github.com/huggingface/accelerate) – Simple distributed training.

---

## Model Merging & Quantization

### Model Merging

- 🌍 [MergeKit](https://github.com/arcee-ai/mergekit) ⭐ 5k+ – Toolkit for merging LLMs (SLERP, TIES, DARE).
- 🌍 [LazyMergeKit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb) – Colab notebook for easy merging.

### Quantization Tools

- 🌍 [llama.cpp](https://github.com/ggerganov/llama.cpp) – GGUF quantization (Q4, Q5, Q8).
- 🌍 [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) – GPTQ quantization.
- 🌍 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) – AWQ quantization.
- 🌍 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) – 4-bit and 8-bit quantization.
- 🌍 [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) – Quantization format specification.

---

## Agents & Orchestration

### Agent Frameworks

- 🌍 [LangChain](https://github.com/langchain-ai/langchain) ⭐ 95k+ – LLM app framework with native Mistral support.
- 🌍 [LlamaIndex](https://github.com/run-llama/llama_index) ⭐ 37k+ – Data framework for RAG with Mistral.
- 🌍 [CrewAI](https://github.com/crewAIInc/crewAI) ⭐ 20k+ – Multi-agent orchestration.
- 🌍 [AutoGen](https://github.com/microsoft/autogen) ⭐ 35k+ – Microsoft's multi-agent framework.
- 🌍 [Semantic Kernel](https://github.com/microsoft/semantic-kernel) – Microsoft's AI orchestration SDK.
- 🌍 [Haystack](https://github.com/deepset-ai/haystack) – End-to-end NLP framework.
- 🌍 [PydanticAI](https://github.com/pydantic/pydantic-ai) – Type-safe AI agent framework.

### Function Calling & Structured Output

- 🧠 [Mistral Function Calling](https://docs.mistral.ai/capabilities/function_calling/) – Native function calling docs.
- 🌍 [Instructor](https://github.com/jxnl/instructor) ⭐ 8k+ – Structured outputs with Pydantic.
- 🌍 [Outlines](https://github.com/outlines-dev/outlines) ⭐ 10k+ – Guaranteed structured generation.
- 🌍 [Marvin](https://github.com/prefecthq/marvin) – AI functions with type hints.

---

## Tooling & Dev Experience

### IDE Extensions & Code Assistants

- 🧠 [Zed Extensions](https://github.com/mistralai/zed-extensions) – Official Mistral for Zed editor.
- 🌍 [Continue](https://github.com/continuedev/continue) ⭐ 20k+ – Open-source AI code assistant (VSCode/JetBrains).
- 🌍 [Tabby](https://github.com/TabbyML/tabby) ⭐ 22k+ – Self-hosted GitHub Copilot alternative.
- 🌍 [Aider](https://github.com/paul-gauthier/aider) ⭐ 20k+ – AI pair programming in terminal.
- 🌍 [Cody](https://github.com/sourcegraph/cody) – AI coding assistant with codebase context.

### Development Tools

- 🌍 [LiteLLM](https://github.com/BerriAI/litellm) ⭐ 15k+ – Unified API for 100+ LLMs.
- 🌍 [Promptfoo](https://github.com/promptfoo/promptfoo) ⭐ 5k+ – LLM evaluation and red-teaming.
- 🌍 [Langfuse](https://github.com/langfuse/langfuse) ⭐ 7k+ – Open-source LLM observability.
- 🌍 [Phoenix](https://github.com/Arize-ai/phoenix) – ML observability for LLM apps.
- 🌍 [Weights & Biases](https://wandb.ai) – Experiment tracking with LLM support.

---

## Community Projects

### Chat Interfaces

- 🌍 [Open WebUI](https://github.com/open-webui/open-webui) ⭐ 50k+ – Self-hosted ChatGPT-like UI.
- 🌍 [LibreChat](https://github.com/danny-avila/LibreChat) ⭐ 20k+ – Multi-model chat interface.
- 🌍 [Lobe Chat](https://github.com/lobehub/lobe-chat) ⭐ 50k+ – Modern extensible chat framework.
- 🌍 [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui) – Open-source ChatGPT clone.
- 🌍 [BetterChatGPT](https://github.com/ztjhz/BetterChatGPT) – Enhanced chat interface.

### RAG & Knowledge Management

- 🌍 [PrivateGPT](https://github.com/zylon-ai/private-gpt) ⭐ 55k+ – Private document Q&A.
- 🌍 [Danswer](https://github.com/danswer-ai/danswer) ⭐ 12k+ – Enterprise Q&A over internal docs.
- 🌍 [Quivr](https://github.com/QuivrHQ/quivr) ⭐ 37k+ – Personal knowledge base.
- 🌍 [Khoj](https://github.com/khoj-ai/khoj) – AI second brain.
- 🌍 [LocalGPT](https://github.com/PromtEngineer/localGPT) – Chat with documents locally.

### Specialized Applications

- 🌍 [Fabric](https://github.com/danielmiessler/fabric) ⭐ 25k+ – AI augmentation framework.
- 🌍 [GPT Researcher](https://github.com/assafelovic/gpt-researcher) ⭐ 15k+ – Autonomous research agent.
- 🌍 [OpenDevin](https://github.com/OpenDevin/OpenDevin) ⭐ 35k+ – AI software engineer.

---

## Demos & Examples

### Official Examples

- 🧠 [Mistral Cookbook](https://github.com/mistralai/cookbook) – RAG, function calling, embeddings, agents.
- 🧠 [Fine-Tuning Guide](https://docs.mistral.ai/capabilities/finetuning/) – Official fine-tuning documentation.
- 🧠 [API Examples](https://docs.mistral.ai/api/) – Complete API reference with examples.

### Community Examples

- 🌍 [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) – Curated LLM resources including Mistral.
- 🌍 [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates) – Production-ready templates.

---

## Tutorials & Guides

### Getting Started

- 🧠 [Mistral Quickstart](https://docs.mistral.ai/getting-started/quickstart/) – Official getting started guide.
- 🧠 [Model Selection Guide](https://docs.mistral.ai/getting-started/models/) – Choosing the right model.
- 🌍 [Run Mistral Locally](https://ollama.com/library/mistral) – Ollama setup guide.

### Fine-Tuning Tutorials

- 🧠 [Official Fine-Tuning](https://docs.mistral.ai/capabilities/finetuning/) – Mistral's fine-tuning guide.
- 🌍 [Axolotl Mistral Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/mistral) – Config examples.
- 🌍 [QLoRA Guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes) – 4-bit fine-tuning.
- 🌍 [Unsloth Tutorial](https://github.com/unslothai/unsloth#mistral) – Fast Mistral fine-tuning.

### RAG & Applications

- 🧠 [RAG with Mistral](https://docs.mistral.ai/guides/rag/) – Official RAG guide.
- 🌍 [LlamaIndex + Mistral](https://docs.llamaindex.ai/en/stable/examples/llm/mistralai/) – RAG with LlamaIndex.
- 🌍 [LangChain + Mistral](https://python.langchain.com/docs/integrations/llms/mistralai/) – LangChain integration.

---

## Benchmarks & Evaluation

### Leaderboards

- 🌍 [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) – Hugging Face benchmarks.
- 🌍 [Chatbot Arena](https://lmarena.ai/) – Human preference rankings.
- 🌍 [Artificial Analysis](https://artificialanalysis.ai/) – LLM quality and speed benchmarks.

### Evaluation Frameworks

- 🌍 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) – EleutherAI's eval framework.
- 🌍 [HELM](https://github.com/stanford-crfm/helm) – Stanford's holistic evaluation.
- 🌍 [OpenCompass](https://github.com/open-compass/opencompass) – Comprehensive LLM evaluation.

### Code Benchmarks

- 🌍 [HumanEval](https://github.com/openai/human-eval) – Code generation benchmark.
- 🌍 [BigCodeBench](https://github.com/bigcode-project/bigcodebench) – Comprehensive code evaluation.
- 🌍 [EvalPlus](https://github.com/evalplus/evalplus) – Rigorous code evaluation.

---

## Research & Papers

### Mistral Technical Reports

- 🧠 [Mistral 7B](https://arxiv.org/abs/2310.06825) – Foundational 7B architecture paper.
- 🧠 [Mixtral of Experts](https://arxiv.org/abs/2401.04088) – Sparse MoE architecture.

### Related Research

- 🌍 [Sliding Window Attention](https://arxiv.org/abs/2004.05150) – Longformer attention mechanism.
- 🌍 [LoRA](https://arxiv.org/abs/2106.09685) – Low-Rank Adaptation paper.
- 🌍 [QLoRA](https://arxiv.org/abs/2305.14314) – Quantized LoRA for efficient fine-tuning.
- 🌍 [DPO](https://arxiv.org/abs/2305.18290) – Direct Preference Optimization.
- 🌍 [Mixture of Experts](https://arxiv.org/abs/1701.06538) – MoE foundations.

---

## Talks & Media

### Official Channels

- 🧠 [Mistral AI Blog](https://mistral.ai/news/) – Official announcements.
- 🧠 [Mistral AI Discord](https://discord.gg/mistralai) – Official community server.
- 🧠 [Mistral AI Twitter/X](https://twitter.com/MistralAI) – Official updates.

### Conferences & Talks

- 🌍 [Hugging Face YouTube](https://www.youtube.com/@HuggingFace) – Tutorials with Mistral.
- 🌍 [AI Explained](https://www.youtube.com/@aiexplained-official) – Technical breakdowns.

---

## Ecosystem & Community

### Cloud Providers

- 🌍 [Azure AI](https://azure.microsoft.com/en-us/products/ai-studio/) – Mistral on Azure AI Studio.
- 🌍 [AWS Bedrock](https://aws.amazon.com/bedrock/) – Mistral via Amazon Bedrock.
- 🌍 [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai) – Mistral on GCP.
- 🌍 [Groq](https://groq.com/) – Ultra-fast Mistral inference.
- 🌍 [Together AI](https://together.ai/) – Mistral model hosting.
- 🌍 [Replicate](https://replicate.com/) – Run Mistral via API.

### Community Hubs

- 🌍 [Hugging Face Hub](https://huggingface.co/mistralai) – Official model repository.
- 🧠 [Mistral Discord](https://discord.gg/mistralai) – Official community.
- 🌍 [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) – Local LLM community.
- 🌍 [r/MistralAI](https://www.reddit.com/r/MistralAI/) – Mistral-focused subreddit.

### Partnerships

- 🧠 [Microsoft Azure Partnership](https://azure.microsoft.com/en-us/blog/microsoft-and-mistral-ai-announce-new-partnership-to-accelerate-ai-innovation-and-introduce-mistral-large-first-on-azure/) – Strategic Azure partnership.
- 🧠 [La Plateforme](https://console.mistral.ai/) – Mistral's cloud platform.

---

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

### Quick Guidelines

1. Ensure all links point to real, existing resources
2. Use consistent formatting: `- 🧠/🌍/🧪 [Name](url) – Brief description.`
3. Prefer high-signal, actively maintained projects
4. Include star counts for major projects (⭐ 10k+)

---

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

This work is licensed under [CC0 1.0 Universal](LICENSE).
