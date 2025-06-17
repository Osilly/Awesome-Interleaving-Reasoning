<div align="center">
    <img alt="logo" src="./images/logo.webp" style="height: 200px;" />
</div>

<div align="center">

# Awesome Interleaving Reasoning

</div>


<p align="center">
    <img src="./images/overview.png" width="95%" height="95%">
</p>


With the release of [OpenAI o1](https://openai.com/o1/) and [Deepseek-R1](https://arxiv.org/abs/2501.12948), reasoning models have yielded remarkably promising results and garnered significant attention from the research community. This development signals that reasoning models represent a critical advancement toward Artificial General Intelligence (AGI). The standard reasoning paradigm can be formally defined as:

- **Standard Reasoning**: The model conducts a comprehensive intermediate reasoning phase prior to generating the final response. This intermediate reasoning typically manifests as unstructured textual content, with the entire inference process constituting a single atomic operation.

Recently, the introduction of [OpenAI o3](https://openai.com/index/introducing-o3-and-o4-mini/), [Deep research](https://openai.com/index/introducing-deep-research/), [Zochi](https://github.com/IntologyAI/Zochi/blob/main/Zochi_Technical_Report.pdf), and [BAGEL](https://arxiv.org/abs/2505.14683) has established an alternative reasoning formulation, which we designate as **Interleaving Reasoning**. In contrast to standard reasoning, Interleaving Reasoning is characterized by multi-turn interactions and exhibits sophisticated reasoning dynamics. This reasoning modality has empirically demonstrated superior accuracy in addressing complex problems. Consequently, we posit that Interleaving Reasoning potentially constitutes the **Next-Generation Reasoning Systems for AGI**. We propose a taxonomy of Interleaving Reasoning that encompasses the following categories:

- **Multimodal Interleaving Reasoning**: The model's inference process operates on diverse information modalities (e.g., textual, visual, auditory, video). This involves an intricately interleaved execution of modality-specific information processing and cross-modal reasoning. Examples: [OpenAi o3](https://openai.com/index/introducing-o3-and-o4-mini/), [DeepEyes](https://arxiv.org/abs/2505.14362).
- **Multi-Round Acting Interleaving Reasoning**: The system achieves task completion through iterative interactions (actions) with the environment. Each action is either predicated upon or performed in conjunction with a reasoning-driven inference step, establishing an interleaved execution of action and inference processes. Examples: [Deep research](https://openai.com/index/introducing-deep-research/), [Search-R1](https://arxiv.org/abs/2503.09516), [ReTool](https://arxiv.org/abs/2504.11536), [UI-TARS](https://arxiv.org/abs/2501.12326), [ReAct](https://arxiv.org/abs/2210.03629).
- **Multi-Agent Interleaving Reasoning**: In a multi-agent system, multiple agents, such as LLMs and MLLMs, engage in collaborative or competitive dynamics via a paradigm of interleaved reasoning. This implies that agents either alternate in contributing discrete reasoning steps, share intermediate conclusions to establish a shared cognitive state, and subsequently build upon this foundation, or their respective inferential processes exhibit mutual influence. Examples: [Society of Minds](https://arxiv.org/abs/2305.14325), [Zochi](https://github.com/IntologyAI/Zochi/blob/main/Zochi_Technical_Report.pdf), [MetaGPT](https://arxiv.org/abs/2309.07870).
- **Unified Understanding and Generation Interleaving Reasoning**: The model's reasoning capabilities are not confined to producing solely unimodal outputs. Instead, it strategically generates multimodal content (e.g., textual and visual elements) as an integral intermediate step within its intrinsic processes of comprehension and problem-solving. Example: [GoT](https://arxiv.org/abs/2503.10639), [T2I-R1](https://arxiv.org/abs/2505.00703), [BAGEL](https://arxiv.org/abs/2505.14683).

> It is imperative to establish precise categorical boundaries:
>
> - While **Multimodal Interleaving Reasoning** could conceivably be subsumed within the **Multi-Round Acting Interleaving Reasoning** paradigm, we formally define Multimodal Interleaving Reasoning as necessitating the direct incorporation of multi-modal information streams during the reasoning process. This information typically derives from the processing of input modalities, as exemplified by [OpenAi o3](https://openai.com/index/introducing-o3-and-o4-mini/), which extracts visual information and integrates it into text-based reasoning workflows.
> - The fundamental distinction between **Multi-Round Acting Interleaving Reasoning** and **Multi-Agent Interleaving Reasoning** lies in their architectural composition: Multi-Round Acting Interleaving Reasoning typically employs a single LLM/MLLM to perform reasoning and determine subsequent actions. Conversely, Multi-Agent Interleaving Reasoning leverages multiple LLM/MLLM entities that collaboratively contribute to reasoning steps.
> - The differentiation between **Unified Understanding and Generation Interleaving Reasoning** and **Multimodal Interleaving Reasoning** resides in their information processing mechanisms. Unified Understanding and Generation Interleaving Reasoning utilizes an unified understanding and generation model capable of directly generating multimodal outputs during the reasoning process. In contrast, Multimodal Interleaving Reasoning typically sources its multimodal information from external systems or processes.

We aim to provide the community with a comprehensive and timely synthesis of this fascinating and promising field, as well as some insights into it. This repository provides valuable reference for researchers in the field of Interleaving Reasoning, please start your exploration! 

***This work is in progress!***

---

<font size=5><center><b> Table of Contents </b> </center></font>

- [Our Group](#our-group)
  - [Originators](#originators)
  - [Members](#members)
- [Our Activities](#our-activities)
- [Standard Reasoning Examples](#standard-reasoning-examples)
- [Awesome Interleaving Reasoning Papers](#awesome-interleaving-reasoning-papers)
  - [Multimodal Interleaving Reasoning](#multimodal-interleaving-reasoning)
  - [Multi-Round Acting Interleaving Reasoning](#multi-round-acting-interleaving-reasoning)
  	- [Search](#search)
  	- [Code](#code)
  	- [UI](#ui)
  	- [Complex acting](##complex-acting)
  	- [Others](#others)
  - [Multi-Agent Interleaving Reasoning](#multi-agent-interleaving-reasoning)
  	- [Debate](#debate)
  	- [Coordination](#coordination)
  - [Unified Understanding and Generation Interleaving Reasoning](#unified-understanding-and-generation-interleaving-reasoning)
  	- [Generation](#generation)
  	- [Understanding](#understanding) 	
- [Awesome Datasets](#awesome-datasets)


---

## Our Group

### Originators

<p align="left">
  &nbsp;&nbsp;
  <img src="./images/wenxuan_new.png" width="100" style="border-radius: 50%;">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./images/zhenfei_new.png" width="100" style="border-radius: 50%;">
  <br>
  <a href="https://scholar.google.com/citations?user=6Ys6HgsAAAAJ&hl=en">
    <b>Wenxuan Huang</b>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=ngPR1dIAAAAJ&hl=en">
    <b>Zhenfei Yin</b>
  </a>
  <br>
  &nbsp;&nbsp;
  ECNU&CUHK
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  USYD&Oxford
</p>


### Members




## Our Activities
---

🔥🔥🔥 **ICCV 2025 Workshop on Multi-Modal Reasoning for Agentic Intelligence (MMRAgi-2025)**  
<p align="center">
    <img src="./images/MMRAgi.png" width="30%" height="30%">
</p>


<font size=7><div align='center' >We organised **[ICCV 2025 Workshop MMRAgi](https://agent-intelligence.github.io/agent-intelligence)**! <br> Submission DDL: Proceeding Track: 24 June 2025, 23:59 AoE, Non-Proceeding Track: 24 July 2025, 23:59 AoE.  </div></font>


---

🔥🔥🔥 **Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models**  
<p align="center">
    <img src="./images/vision_r1.png" width="80%" height="80%">
</p>

<font size=7><div align='center' > [[📖 arXiv Paper](https://arxiv.org/abs/2503.06749)] [[🌟 GitHub](https://github.com/Osilly/Vision-R1)![Star](https://img.shields.io/github/stars/Osilly/Vision-R1.svg?style=social&label=Star)] [[🤗 Vision-R1-cold Dataset](https://huggingface.co/datasets/Osilly/Vision-R1-cold)] [[🤗 Vision-R1-7B](https://huggingface.co/Osilly/Vision-R1-7B)]</div></font>  

<font size=7><div align='center' > This is the first paper to explore how to effectively use RL for MLLMs and introduce Vision-R1, a reasoning MLLM that leverages cold-start initialization and RL training to incentivize reasoning capability.  </div></font>

---

🔥🔥🔥 **DeepEyes: Incentivizing “Thinking with Images” via Reinforcement Learning**  
<p align="center">
    <img src="./images/deepeyes.png" width="80%" height="80%">
</p>

<font size=7><div align='center' > [[📖 arXiv Paper](https://arxiv.org/abs/2505.14362)] [[🌟 GitHub](https://github.com/Visual-Agent/DeepEyes)![Star](https://img.shields.io/github/stars/Visual-Agent/DeepEyes.svg?style=social&label=Star)] [[🤗 Dataset](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)] [[🤗 DeepEyes-7B](https://huggingface.co/ChenShawn/DeepEyes-7B)]</div></font>

<font size=7><div align='center' > The first opensource "o3-like" interleaving reasoning MLLM with "Thinking with Images". They don’t just see an image, they can integrate visual information directly into the reasoning chain.  </div></font>

## Standard Reasoning Examples

* [OpenAI o1] [Introducing OpenAI o1](https://openai.com/o1/)

* [DeepSeek-R1] [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) [[🤗Models](https://huggingface.co/deepseek-ai/DeepSeek-R1)] [[💻Code](https://github.com/deepseek-ai/DeepSeek-R1)]

* [Kimi-k1.5] [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) [[💻Code](https://github.com/MoonshotAI/Kimi-k1.5)]

* [QVQ-Max] [QVQ-Max: Think with Evidence](https://qwenlm.github.io/blog/qvq-max-preview/)

* [Vision-R1] [Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](https://arxiv.org/abs/2501.12948) [[🤗Models](https://huggingface.co/Osilly/Vision-R1-7B)] [[🤗Datasets](https://huggingface.co/datasets/Osilly/Vision-R1-cold)] [[💻Code](https://github.com/Osilly/Vision-R1)]

## Awesome Interleaving Reasoning Papers

> ***PR Temporal***
>
> * [**RL**] [2505] [DeepEyes] [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362) [[🌐Project](https://visual-agent.github.io/)] [[🤗Models](https://huggingface.co/ChenShawn/DeepEyes-7B)] [[🤗Datasets](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)] [[💻Code](https://github.com/Visual-Agent/DeepEyes)]
>
> *You can select your categories in [**Pretrain**, **SFT**, **RL**, **Prompt**, **Position paper**, **Survey paper**] and so on. Furthermore, you can combine them, for example,  **SFT+RL**.*

### Multimodal Interleaving Reasoning

> Definition: The model's inference process operates on diverse information modalities (e.g., textual, visual, auditory, video). This involves an intricately interleaved execution of modality-specific information processing and cross-modal reasoning.

* [2504] [OpenAI o3] [Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)

* [**RL**] [2505] [DeepEyes] [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362) [[🌐Project](https://visual-agent.github.io/)]  [[🤗Models](https://huggingface.co/ChenShawn/DeepEyes-7B)]  [[🤗Datasets](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)] [[💻Code](https://github.com/Visual-Agent/DeepEyes)]

* [**SFT**] [2312] [V\*] [V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/abs/2312.14135) [[🌐Project](https://vstar-seal.github.io)] [[💻Code](https://github.com/penghao-wu/vstar)]

* [**Prompt**] [2211] [VISPROG] [Visual Programming: Compositional visual reasoning without training](https://arxiv.org/abs/2211.11559) [[🌐Project](https://prior.allenai.org/projects/visprog)] [[💻Code](https://github.com/allenai/visprog)]

* [**Prompt**] [2406] [Sketchpad] [Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models](https://arxiv.org/abs/2406.09403) [[🌐Project](https://visualsketchpad.github.io)] [[💻Code](https://github.com/Yushi-Hu/VisualSketchpad)]

* [**Prompt**] [2505] [VAT] [Visual Abstract Thinking Empowers Multimodal Reasoning](https://arxiv.org/abs/2502.17422) [[💻Code](https://github.com/THUNLP-MT/VAT)]

* [**SFT**] [2403] [Visual COT] [Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning](https://arxiv.org/abs/2403.16999) [[🌐Project](https://hao-shao.com/projects/viscot.html)] [[💻Code](https://github.com/deepcs233/Visual-CoT)] [[🤗Models](https://huggingface.co/collections/deepcs233/viscot-65fe883e2a0cdd3c59fc5d63)] [[🤗Datasets](https://huggingface.co/datasets/deepcs233/Visual-CoT)]

* [**SFT**] [2501] [MVoT] [Imagine while Reasoning in Space: Multimodal Visualization-of-Thought](https://arxiv.org/pdf/2501.07542) [[🌐Project](https://thegenerality.com/agi/)] [[💻Code](https://github.com/chengzu-li/MVoT)] [[🤗Models](https://huggingface.co/collections/deepcs233/viscot-65fe883e2a0cdd3c59fc5d63)] [[🤗Datasets](https://huggingface.co/datasets/deepcs233/Visual-CoT)]

* [**SFT**] [2503] [CoT-VLA] [CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](https://arxiv.org/abs/2503.22020) [[🌐Project](https://cot-vla.github.io/)]

* [**SFT**] [2505] [v1] [Don’t Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation](https://arxiv.org/pdf/2505.18842) [[💻Code](https://github.com/jun297/v1)] [[🤗Models](https://huggingface.co/kjunh/v1-7B)]

* [**SFT**] [2505] [MathCoder-VL] [MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](https://arxiv.org/abs/2505.10557) [[💻Code](https://github.com/mathllm/MathCoder)] [[🤗Models](https://huggingface.co/collections/MathLLMs/mathcoder-vl-68263a5d0b71cac81b6568b4)] [[🤗Datasets](https://huggingface.co/datasets/MathLLM/MathCodeInstruct)]

* [**SFT+RL**] [2505] [OpenThinkIMG] [OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/abs/2505.08617) [[💻Code](https://github.com/zhaochen0110/OpenThinkIMG)]

* [**SFT+RL**] [2505] [Visual-ARFT] [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/pdf/2505.14246) [[💻Code](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT)] [[🤗Models](https://huggingface.co/collections/laolao77/visual-arft-682c601d0e35ac6470adfe9f)] [[🤗Datasets](https://huggingface.co/datasets/laolao77/MAT)]

* [**SFT+RL**] [2505] [CoF] [Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL](https://arxiv.org/pdf/2505.15436) [[🌐Project](https://cof-reasoning.github.io/)] [[💻Code](https://github.com/xtong-zhang/Chain-of-Focus)] [[🤗Models](https://huggingface.co/collections/laolao77/visual-arft-682c601d0e35ac6470adfe9f)] [[🤗Datasets](https://huggingface.co/datasets/laolao77/MAT)]

* [**SFT+RL**] [2505] [Pixel Reasoner] [Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning](https://arxiv.org/abs/2505.15966) [[🌐Project](https://tiger-ai-lab.github.io/Pixel-Reasoner/)] [[💻Code](https://github.com/TIGER-AI-Lab/Pixel-Reasoner)] [[🤗Models](https://huggingface.co/TIGER-Lab/PixelReasoner-RL-v1)] [[🤗Datasets](https://huggingface.co/collections/TIGER-Lab/pixel-reasoner-682fe96ea946d10dda60d24e)]

* [**SFT+RL**] [2505] [V-Triune] [One RL to See Them All: Visual Triple Unified Reinforcement Learning](https://arxiv.org/abs/2505.18129) [[💻Code](https://github.com/MiniMax-AI/One-RL-to-See-Them-All)] [[🤗Models](https://huggingface.co/One-RL-to-See-Them-All)] [[🤗Datasets](https://huggingface.co/datasets/One-RL-to-See-Them-All/Orsta-Data-47k)]

* [**SFT+RL**] [2505] [ViGoRL] [Grounded Reinforcement Learning for Visual Reasoning](https://arxiv.org/abs/2505.23678) [[🌐Project](https://visually-grounded-rl.github.io/)]

* [**SFT+RL**] [2505] [VLM-R\3] [VLM-R\3: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.16192)

* [**RL**] [2505] [Ground-R1] [Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.20272)

* [**RL**] [2505] [GRIT] [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/pdf/2505.15436) [[🌐Project](https://grounded-reasoning.github.io/)] [[💻Code](https://github.com/eric-ai-lab/GRIT)] [[🤗Models](https://huggingface.co/yfan1997/GRIT-20-Qwen2.5-VL-3B)]


### Multi-Round Acting Interleaving Reasoning

> Definition: The system achieves task completion through iterative interactions (actions) with the environment. Each action is either predicated upon or performed in conjunction with a reasoning-driven inference step, establishing an interleaved execution of action and inference processes.

#### Search

* [**SFT+RL**] [2505] [R1-Searcher++:] [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005) [[💻Code](https://github.com/RUCAIBox/R1-Searcher-plus)]


* [**RL**] [2503] [ReSearch] [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470) [[🤗Models](https://huggingface.co/agentrl/ReSearch-Qwen-7B-Instruct)]  [[🤗Datasets](https://huggingface.co/datasets/agentrl/ReCall-data)] [[💻Code](https://github.com/Agent-RL/ReCall)]

* [**RL**] [2503] [R1-Searcher] [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) [[🤗Models](https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL)]  [[🤗Datasets](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki)] [[💻Code](https://github.com/RUCAIBox/R1-Searcher)]

* [**RL**] [2503] [Search-R1] [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) [[💻Code](https://github.com/PeterGriffinJin/Search-R1)]

* [2502] [Deep research] [Introducing deep research](https://openai.com/index/introducing-deep-research/)

* [**RL**] [2112] [WebGPT] [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) [[🤗Datasets](https://huggingface.co/datasets/openai/webgpt_comparisons)]

#### Code

* [**SFT+RL**] [2504] [ReTool] [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) [[🌐Project](https://retool-rl.github.io/)] [[💻Code](https://github.com/penghao-wu/vstar) [[🤗Models](https://huggingface.co/JoeYing/ReTool-Qwen-32B)]  [[🤗Datasets](https://huggingface.co/datasets/JoeYing/ReTool-SFT)] [[💻Code](https://github.com/ReTool-RL/ReTool)]

* [**SFT**] [2410] [MathCoder2] [MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code](https://arxiv.org/abs/2410.08196) [[🤗Models](https://huggingface.co/MathGenie/MathCoder2-Llama-3-8B)]  [[🤗Datasets](https://github.com/mathllm/MathCoder2?tab=readme-ov-file#data-processing)] [[💻Code](https://github.com/mathllm/MathCoder2)]

* [**SFT**] [2312] [VPD] [Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models](https://arxiv.org/abs/2312.03052)

* [**SFT**] [2310] [MathCoder] [MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning](https://arxiv.org/abs/2310.03731) [[🤗Models](https://huggingface.co/MathLLM/MathCoder-L-7B)]  [[🤗Datasets](https://huggingface.co/datasets/MathLLM/MathCodeInstruct)] [[💻Code](https://github.com/mathllm/MathCoder)]

* [**SFT**] [2309] [ToRA] [ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving](https://arxiv.org/abs/2309.17452) [[🌐Project](https://microsoft.github.io/ToRA/)]  [[🤗Models](https://huggingface.co/llm-agents)]  [[🤗Datasets](https://huggingface.co/datasets/llm-agents/CriticBench)] [[💻Code](https://github.com/microsoft/ToRA)]


#### UI

* [**Pretrain+SFT+RL**] [2501] [UI-TARS] [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326) [[🌐Project](https://seed-tars.com/)] [[🤗Models](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)] [[💻Code](https://github.com/bytedance/UI-TARS)]

* [**Prompt**] [2304] [DroidBot-GPT] [DroidBot-GPT: GPT-powered UI Automation for Android](https://arxiv.org/abs/2304.07061) [[💻Code](https://github.com/MobileLLM/DroidBot-GPT)]


#### Complex acting

* [**RL**] [2505] [GiGPO] [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978) [[💻Code](https://github.com/langfengQ/verl-agent)]

* [**Prompt**] [2409] [AWM] [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) [[🌐Project](https://visual-agent.github.io/)] [[💻Code](https://github.com/zorazrw/agent-workflow-memory)]

* [**Prompt**] [2210] [ReAct] [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) [[🌐Project](https://react-lm.github.io/)] [[💻Code](https://github.com/ysymyth/ReAct)]

#### Others



### Multi-Agent Interleaving Reasoning

> Definition: In a multi-agent system, multiple agents, such as LLMs and MLLMs, engage in collaborative or competitive dynamics via a paradigm of interleaved reasoning. This implies that agents either alternate in contributing discrete reasoning steps, share intermediate conclusions to establish a shared cognitive state, and subsequently build upon this foundation, or their respective inferential processes exhibit mutual influence.

#### Debate

* [**RL**] [2411] [ACC-Collab] [ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration](https://arxiv.org/abs/2411.00053) [[💻Code](https://github.com/LlenRotse/ACC-Collab)]

* [**Prompt**] [2409] [GroupDebate] [GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion](https://arxiv.org/abs/2409.14051)

* [**Position paper**] [2311] [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) [[💻Code](https://github.com/instadeepai/DebateLLM)]

* [**Prompt**] [2305] [Society of Minds] [improving factuality and reasoning in language models through multiagent debate](https://arxiv.org/abs/2305.14325) [[💻Code](https://github.com/composable-models/llm_multiagent_debate)]

#### Coordination

* [2503] [Zochi] [Zochi Technical Report](https://github.com/IntologyAI/Zochi/blob/main/Zochi_Technical_Report.pdf) [[🌐Project](https://www.intology.ai/blog/zochi-tech-report)] [[💻Code](https://github.com/IntologyAI/Zochi)]

* [**Prompt**] [2309] [MetaGPT] [Agents: An Open-source Framework for Autonomous Language Agents](https://arxiv.org/abs/2309.07870) [[🌐Project](https://aiwaves-cn.github.io/agents/)] [[💻Code](https://github.com/aiwaves-cn/agents)]

* [**Prompt**] [2308] [AgentVerse] [AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors](https://arxiv.org/abs/2308.10848) [[💻Code](https://github.com/OpenBMB/AgentVerse)]

* [**Prompt**] [2308] [MetaGPT] [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) [[💻Code](https://github.com/FoundationAgents/MetaGPT)]


### Unified Understanding and Generation Interleaving Reasoning

> Definition: The model's reasoning capabilities are not confined to producing solely unimodal outputs. Instead, it strategically generates multimodal content (e.g., textual and visual elements) as an integral intermediate step within its intrinsic processes of comprehension and problem-solving. 

#### Generation

* [**Pretrain+SFT**] [2505] [BAGEL] [Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683) [[🌐Project](https://bagel-ai.org/)] [[🤗Models](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)] [[🤗Datasets](https://github.com/ByteDance-Seed/Bagel/blob/main/TRAIN.md#data-prepration)] [[💻Code](https://github.com/bytedance-seed/BAGEL)]

* [**RL**] [2505] [T2I-R1] [T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT](https://arxiv.org/abs/2505.00703) [[🤗Models](https://huggingface.co/CaraJ/T2I-R1)] [[💻Code](https://github.com/CaraJ7/T2I-R1)]

* [**SFT**] [2503] [GoT] [GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/abs/2503.10639) [[🤗Models](https://huggingface.co/LucasFang/GoT-6B)] [[🤗Datasets](https://github.com/rongyaofang/GoT#released-datasets)] [[💻Code](https://github.com/rongyaofang/GoT)]


#### Understanding

## Awesome Datasets



