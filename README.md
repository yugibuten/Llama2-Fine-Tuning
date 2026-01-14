# Llama 2 Fine-Tuning using LoRA & QLoRA

This repository contains a comprehensive implementation for fine-tuning the **Llama-2-7b** model using Parameter-Efficient Fine-Tuning (PEFT) techniques.

## 🚀 Overview

Fine-tuning Large Language Models (LLMs) usually requires massive computational resources. This project demonstrates how to fine-tune Llama 2 on a single consumer-grade GPU (like the T4 in Google Colab) by using **QLoRA** (4-bit quantization) to minimize VRAM usage.

## 📊 Dataset

The model is fine-tuned on a subset of the **OpenAssistant Guanaco** dataset, which is designed to improve the instruction-following capabilities of chat models.

* **Original Dataset:** [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

## 🏗️ Project Architecture & Important Steps

The notebook is structured into logical blocks to ensure efficient training. Here are the most critical components:

### 1. Environment & GPU Setup
Before running, the notebook guides you to enable the **T4 GPU** in Colab. It then installs essential libraries:
* `bitsandbytes`: For 4-bit quantization.
* `peft`: For Parameter-Efficient Fine-Tuning.
* `trl`: For the Supervised Fine-tuning (SFT) trainer.

### 2. 4-bit Quantization (The "Q" in QLoRA)
This is the most important cell for memory efficiency. We load the model in 4-bit using the `BitsAndBytesConfig`. This reduces the model size significantly, allowing it to fit on a 16GB GPU.

### 3. LoRA Configuration
Instead of training all billions of parameters, we use **LoRA** to target specific modules (like `q_proj` and `v_proj`). 
* **Rank (r):** Set to 64.
* **Alpha:** Set to 16.
* **Dropout:** 0.1 to prevent overfitting.

### 4. SFTTrainer Implementation
We use the `SFTTrainer` from the `trl` library. This high-level API handles the training loop, dataset formatting, and integration with the PEFT model automatically.

## 🛠️ How to Use

### 1. Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yugibuten/Llama2-Fine-Tuning/blob/main/Lllama2(LoRA_&_QLoRA).ipynb)

### 2. Hugging Face Access
Llama 2 is a gated model. You must:
1.  Request access on the [Meta Website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
2.  Accept the terms on the [Hugging Face Model Page](https://huggingface.co/meta-llama/Llama-2-7b-hf).
3.  Enter your Hugging Face **Write Token** when prompted in the notebook.

## 📈 Monitoring
The project is configured to log training metrics to **TensorBoard**, allowing you to visualize the loss curve in real-time within the notebook.

## 📄 License
MIT License - see the [LICENSE](LICENSE) file for details.
