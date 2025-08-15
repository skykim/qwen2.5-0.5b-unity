# qwen2.5-0.5b-unity

Multilingual On-Device SLM: Qwen2.5-0.5B Instruct in Unity Inference Engine

## Overview

This repository provides a lightweight, cross-platform inference engine optimized for [Qwen2.5-0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), a state-of-the-art small language model (SLM) from Alibaba Cloud's Qwen2.5 series. Qwen2.5-0.5B delivers exceptional performance in a compact size, making it perfect for on-device applications.

## Features

- ✅ Support for 29 languages
- ✅ Qwen2Tokenizer implemented in C#
- ✅ On-device processing: No internet connection required
- ✅ Quantized model: Uint8 (648MB)

## Requirements

- **Unity**: `6000.0.50f1`
- **Inference Engine**: `2.3.0`

## Architecture

### 1. Qwen2Tokenizer in C#

BPE-based tokenizer ported from HuggingFace [Qwen2 Tokenizer](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2) Python implementation to C#.

### 2. Qwen2.5-0.5B Instruct (ONNX)

You can download the KV-cache-optimized architecture using optimum-cli, or easily download a pre-built ONNX file from [onnx-community](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct).

## Getting Started

### 1. Project Setup

- Clone or download this repository
- Unzip the provided [StreamingAssets.zip](https://drive.google.com/file/d/1TgPjfP3bvquwNfPSCc5c0nnHQzOGVS-s/view?usp=sharing) file and place its contents into the `/Assets/StreamingAssets` directory in your project

### 2. Run the Demo Scene

- Open the `/Assets/Scenes/SLMScene.unity` scene in the Unity Editor
- Run the scene to see the SLM in action

## Demo

Experience qwen2.5-0.5b-unity in action! Check out our demo showcasing the model's capabilities:

[![Piper Unity](https://img.youtube.com/vi/BHbWHjJmgU8/0.jpg)](https://www.youtube.com/watch?v=BHbWHjJmgU8)

## Links

- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Onnx Community: Qwen2.5-0.5B-Instruct](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct)
- [Qwen2 Tokenizer](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2)

## License

Qwen2.5 is licensed under the Apache 2.0 License.
