<div align="center">
    <img alt="logo" src="./images/logo.webp" style="height: 200px;" />
</div>

<div align="center">

# Awesome Interleaving Reasoning

</div>


<p align="center">
    <img src="./images/overview.png" width="95%" height="95%">
</p>


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
> *You can select your class in [**Pretrain**, **SFT**, **RL**, **Prompt**] and so on. Furthermore, you can combine them, for example,  **SFT+RL**.*

### Multimodal Interleaving Reasoning

> Definition: The model's inference process operates on diverse information modalities (e.g., textual, visual, auditory, video). This involves an intricately interleaved execution of modality-specific information processing and cross-modal reasoning.

* [**SFT+RL**] [2505] [OpenThinkIMG] [OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/abs/2505.08617) [[💻Code](https://github.com/Visual-Agent/DeepEyes)]

* [**RL**] [2505] [DeepEyes] [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362) [[🌐Project](https://visual-agent.github.io/)]  [[🤗Models](https://huggingface.co/ChenShawn/DeepEyes-7B)]  [[🤗Datasets](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)] [[💻Code](https://github.com/Visual-Agent/DeepEyes)]

* [**SFT**] [2312] [V\*] [V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/abs/2312.14135) [[🌐Project](https://vstar-seal.github.io)] [[💻Code](https://github.com/penghao-wu/vstar)]

### Multi-Round Acting Interleaving Reasoning

> Definition: The system achieves task completion through iterative interactions (actions) with the environment. Each action is either predicated upon or performed in conjunction with a reasoning-driven inference step, establishing an interleaved execution of action and inference processes.

#### Search

* [**SFT+RL**] [2505] [R1-Searcher++:] [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005) [[💻Code](https://github.com/RUCAIBox/R1-Searcher-plus)]


* [**RL**] [2503] [ReSearch] [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470) [[🤗Models](https://huggingface.co/agentrl/ReSearch-Qwen-7B-Instruct)]  [[🤗Datasets](https://huggingface.co/datasets/agentrl/ReCall-data)] [[💻Code](https://github.com/Agent-RL/ReCall)]

* [**RL**] [2503] [R1-Searcher] [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) [[🤗Models](https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL)]  [[🤗Datasets](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki)] [[💻Code](https://github.com/RUCAIBox/R1-Searcher)]

* [**RL**] [2503] [Search-R1] [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) [[💻Code](https://github.com/PeterGriffinJin/Search-R1)]

* [Deep research] [2502] [Introducing deep research](https://openai.com/index/introducing-deep-research/)

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

* [**Position paper**] [2311] [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371)


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



