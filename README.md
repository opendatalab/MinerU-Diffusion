<p align="center">
  <img src="assets/banner.png" alt="MinerU-Diffusion" width="100%">
</p>

# MinerU-Diffusion

<p align="center">
  <img src="https://img.shields.io/badge/⚡_Fully_Async-yellow?style=for-the-badge" alt="Fully Async" />
  <img src="https://img.shields.io/badge/💰_Zero_API_or_Zero_GPU-blue?style=for-the-badge" alt="Zero API or Zero GPU" />
  <img src="https://img.shields.io/badge/🤖_Personalized-success?style=for-the-badge" alt="Personalized" />
  <img src="https://img.shields.io/badge/🛠️_Auto_Optimization-orange?style=for-the-badge" alt="Auto" />
  <img src="https://img.shields.io/badge/💬_Language_Feedback-purple?style=for-the-badge" alt="Language Feedback" />
  <img src="https://img.shields.io/badge/🧠_Hybrid_RL-red?style=for-the-badge" alt="Hybrid RL" />
  <img src="https://img.shields.io/badge/🌍_Real_World_Agentic_RL-green?style=for-the-badge" alt="General Agentic RL" />
  <br><br>
  <a href="./MinerU-Diffusion-V1.pdf"><img src="https://img.shields.io/badge/📄_Tech_Report-red?style=flat-square" alt="Tech Report" /></a>
  <a href="https://yinjjiew.github.io/projects/openclawrl1"><img src="https://img.shields.io/badge/Blog-Page-blue?style=flat-square" alt="OpenClaw-RL Blog" /></a>
  <a href="https://openclaw.ai"><img src="https://img.shields.io/badge/OpenClaw-Plugin-orange?style=flat-square" alt="OpenClaw Plugin" /></a>
  <a href="https://github.com/sgl-project/sglang"><img src="https://img.shields.io/badge/SGLang-Supported-purple?style=flat-square" alt="SGLang Supported" /></a>
  <a href="https://github.com/GeeeekExplorer/nano-vllm"><img src="https://img.shields.io/badge/Nano--vLLM-Supported-yellow?style=flat-square" alt="Nano-vLLM Supported" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License MIT" /></a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/a58aacad-3c1d-47aa-bbd1-cf8c5f36de6f" controls width="200"></video>
</p>



## 📰 News

- **[2026/3/20]** 🔥 We release **MinerU-Diffusion-V1** — a diffusion-based framework for document OCR that
replaces autoregressive decoding with block-level parallel diffusion decoding.

---

## 💡 TL;DR

> **MinerU-Diffusion** reframes document OCR as an inverse rendering problem and replaces slow, error-prone autoregressive decoding with parallel diffusion decoding.

By introducing block-wise diffusion, uncertainty-driven curriculum learning, it achieves up to 3.2× faster decoding while improving robustness and reducing reliance on language priors.

<p align="center">
  <img src="assets/train.png"  alt="Overview"  width="600">
</p>





> **Highlights:** MinerU-Diffusion maintains a strong accuracy–efficiency trade-off, achieving 2.12× speedup with 99.9% and 3.01× speedup with 98.8% relative accuracy.
> 
<details>
<summary><b>🌈 Features</b></summary>

### Fully Asynchronous 4-Component Architecture
OpenClaw-RL decouples **agent serving**, **rollout collection**, **PRM/judge evaluation**, and **policy training** into independent async loops. None of them block one another: the model continues serving requests while training runs in the background, and judging happens concurrently with new interactions.

### Self-Hosted & Private by Design
The entire stack, including the **policy model**, **judge/PRM**, and **trainer**, runs on **your own infrastructure**. Conversation data stays within your system, and no third-party model API is required.

### From Feedback to Gradient — Automatically
You do not need to manually label data. The system automatically:
- Organizes multi-turn interactions into session-aware training trajectories
- Classifies API messages into **main-line** (trainable) vs. **side** (non-trainable) turns
- Uses the next user, environment, or tool feedback as a natural "next-state" signal
- Runs PRM/judge evaluation asynchronously, with majority voting when needed for more robust scoring
- Submits ready samples to the trainer as they become available

### Three Optimization Methods in One Framework

**Binary RL (GRPO):** A Process Reward Model scores each turn based on next-state feedback. The scalar reward is then used with GRPO advantage estimation and a PPO-style clipped surrogate loss.

**On-Policy Distillation (OPD):** When the next state reveals useful hindsight, a judge model extracts a textual hint. This hint augments the original prompt to create an enhanced teacher, whose token-level log-probability gap with the student becomes a directional advantage signal richer than any scalar reward.

**Combination Method:** OpenClaw-RL further combines Binary RL and OPD in a unified training recipe, leveraging the dense scalar supervision of Binary RL together with the richer token-level directional signal from OPD. This combination achieves stronger and more robust optimization than either method alone.

### From Personal Agents to Real-World Agentic RL
The same framework supports both personalized OpenClaw optimization and scalable RL for **terminal**, **GUI**, **SWE**, and **tool-call** agents in real-world settings.



</details>

---



## 🎯 Roadmap

Our long-term goal is to **build efficient and reliable diffusion-based decoding for document OCR**. 

- ✅ **Release MinerU-Diffusion-V1:** A diffusion-based framework for document OCR that replaces autoregressive decoding with block-level parallel diffusion decoding.
- ✅ Support SGLang to accommodate diffusion computation.
- ⬜ Release MinerU-Diffusion-V2: More Small, More Faster, More Elegant, More Powerful!
- ⬜ Release Training Code
