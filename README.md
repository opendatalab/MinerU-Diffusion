<p align="center">
  <img src="assets/banner.png" alt="MinerU-Diffusion" width="100%">
</p>

# MinerU-Diffusion

<p align="center">
  <img src="https://img.shields.io/badge/✨_Diffusion_Decoding-darkgreen?style=for-the-badge" alt="Diffusion Decoding" />
  <img src="https://img.shields.io/badge/⚡_Fast_Inference-yellow?style=for-the-badge" alt="Fast Inference" />
  <img src="https://img.shields.io/badge/🧩_Block--wise_Parallel-blue?style=for-the-badge" alt="Block-wise Parallel" />
  <img src="https://img.shields.io/badge/📄_Robust_OCR-red?style=for-the-badge" alt="Robust OCR" />
  <img src="https://img.shields.io/badge/🏗️_Layout_Aware-orange?style=for-the-badge" alt="Layout Aware" />
  <img src="https://img.shields.io/badge/🚀_SGLang_Ready-purple?style=for-the-badge" alt="SGLang Ready" />
  <img src="https://img.shields.io/badge/🤗_2.5B_Model-brightgreen?style=for-the-badge" alt="2.5B Model" />
  <br><br>
  <a href="./docs/MinerU-Diffusion-V1.pdf"><img src="https://img.shields.io/badge/📄_Tech_Report-red?style=flat-square" alt="Tech Report" /></a>
  <a href="https://huggingface.co/opendatalab/MinerU-Diffusion-V1-0320-2.5B"><img src="https://img.shields.io/badge/🤗_Model-HuggingFace-yellow?style=flat-square" alt="Model" /></a>
  <a href="https://yinjjiew.github.io/projects/openclawrl1"><img src="https://img.shields.io/badge/Blog-Page-blue?style=flat-square" alt="OpenClaw-RL Blog" /></a>
  <a href="https://github.com/sgl-project/sglang"><img src="https://img.shields.io/badge/SGLang-Supported-purple?style=flat-square" alt="SGLang Supported" /></a>
  <a href="https://github.com/GeeeekExplorer/nano-vllm"><img src="https://img.shields.io/badge/Nano--DVLM-Adapted-yellow?style=flat-square" alt="Nano-DVLM Adapted" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License MIT" /></a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/a58aacad-3c1d-47aa-bbd1-cf8c5f36de6f" controls width="200"></video>
</p>



## 📰 News

- **[2026/3/20]** 🔥 We release **MinerU-Diffusion-V1** — a diffusion-based framework for document OCR that
replaces autoregressive decoding with block-level parallel diffusion decoding.

## 🎯 Roadmap

Our long-term goal is to **build efficient and reliable diffusion-based decoding for document OCR**. 

- ✅ **Release MinerU-Diffusion-V1:** A diffusion-based framework for document OCR that replaces autoregressive decoding with block-level parallel diffusion decoding.
- ✅ Support [SGLang](https://github.com/sgl-project/sglang) to accommodate diffusion computation.
- ✅ Complete the [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) adaptation used by our `nano_dvlm` engine for single-GPU inference.
- ⬜ Release MinerU-Diffusion-V2: More Small, More Faster, More Elegant, More Powerful!
- ⬜ Release Training Code

---

## 💡 TL;DR

> **MinerU-Diffusion** reframes document OCR as an inverse rendering problem and replaces slow, error-prone autoregressive decoding with parallel diffusion decoding.

By introducing block-wise diffusion, uncertainty-driven curriculum learning, it achieves up to 3.2× faster decoding while improving robustness and reducing reliance on language priors.

<p align="center">
  <img src="assets/train.png"  alt="Overview"  width="600">
</p>





> **Highlights:** MinerU-Diffusion maintains a strong accuracy–efficiency trade-off, achieving 2.12× speedup with 99.9% and 3.01× speedup with 98.8% relative accuracy.
> 
## 📈 Performance

<p align="center">
  <img src="assets/performance_tradeoff.jpeg" alt="Performance Trade-off" width="775">
</p>

MinerU-Diffusion provides a flexible accuracy-throughput trade-off through threshold control. Compared with MinerU2.5, it achieves up to **3.26x** TPS, while also offering practical operating points such as **2.12x speedup with 99.9% relative accuracy** and **3.01x speedup with 98.8% relative accuracy**.

## 🗂️ Repository Layout

```text
MinerU-Diffusion/
├── assets/
│   ├── banner.png
│   ├── performance_tradeoff.jpeg
│   └── train.png
├── docs/
│   └── MinerU-Diffusion-V1.pdf
├── engines/
│   ├── hf/
│   ├── nano_dvlm/
│   └── sglang/
├── mineru_diffusion/
│   ├── configuration_mineru_diffusion.py
│   ├── modeling_mineru_diffusion.py
│   └── processing_mineru_diffusion.py
├── scripts/
│   ├── run_inference.py
│   └── run_inference.sh
├── LICENSE
└── README.md
```

## 🚀 Inference

Replace `MODEL_PATH` and `IMAGE_PATH` with your own paths before running.

### HF Engine

```bash
cd /mnt/shared-storage-user/mineru2-shared/niujunbo/niujunbo_dev/MinerU-Diffusion
ENGINE=hf \
MODEL_PATH=/path/to/MinerU-Diffusion-model \
IMAGE_PATH=/path/to/input-image.png \
bash scripts/run_inference.sh
```

### Nano-DVLM Engine

```bash
cd /mnt/shared-storage-user/mineru2-shared/niujunbo/niujunbo_dev/MinerU-Diffusion
ENGINE=nano_dvlm \
MODEL_PATH=/path/to/MinerU-Diffusion-model \
IMAGE_PATH=/path/to/input-image.png \
bash scripts/run_inference.sh
```

## 🤝 Acknowledgement

This work is heavily built on the following open-source models:

[MinerU](https://github.com/opendatalab/mineru), [Qwen2-VL](https://github.com/QwenLM/Qwen3-VL), [SDAR](https://github.com/JetAstra/SDAR), and [LLaDA](https://github.com/ML-GSAI/LLaDA).

These acceleration methods (engines):

[SGLang](https://github.com/sgl-project/sglang), [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) as the upstream basis for our `nano_dvlm` adaptation, and [jetengine](https://github.com/Labman42/JetEngine/tree/0ddc55ad3fb712b6374515b78d656f420e1a7243),

and theoretical foundations:

[MDLM](https://arxiv.org/pdf/2406.07524), [DiffuLLaMA](https://arxiv.org/abs/2410.17891), [Block Diffusion](https://arxiv.org/abs/2503.09573).

## 📚 Citation

If you find our paper and code useful in your research, please consider giving a star and citation.

```bibtex
@article{mineru_diffusion,
  title={MinerU-Diffusion Technical Report},
  author={MinerU-Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---
