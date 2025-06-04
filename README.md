# 🔁 Neural Machine Translation using Transformers (English ↔ Hindi)

This project implements a Transformer-based Neural Machine Translation (NMT) model for translating between English and Hindi, inspired by the landmark paper:  
**[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**.

## 🚀 Overview

- ✅ Built from scratch in PyTorch  
- ✅ Custom tokenizer using HuggingFace `tokenizers`  
- ✅ Dataset: [cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi)  
- ✅ Implements multi-head attention, positional encoding, and encoder-decoder structure  
- ✅ Trained on 90K+ sentence pairs with manual + interval-based checkpoint saving  
- ✅ Translates live from English ➡ Hindi using a CLI

---

## 🧠 Architecture

- Based on the **original Transformer encoder-decoder**:
  - **Encoder**: Self-attention + Feedforward layers
  - **Decoder**: Masked self-attention + Encoder attention
  - **Positional Encoding** to inject sequence information
- Implemented with modular, readable `model.py` for clarity and reusability
---

## 🛠️ How to Use

### 1. Install Requirements
torch>=2.0.0
tokenizers>=0.13.3
datasets>=2.18.0
tqdm>=4.66.0
tensorboard>=2.15.0

pip install -r requirements.txt

Optional (if you're using GPU on Mac)
If you're using Apple M1/M2 GPU acceleration (MPS):

# Optional - Apple Silicon optimized build (PyTorch)
# Uncomment if using Apple M1/M2
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps

Input Sentence (English) ─► Tokenizer ─► Positional Encoding ─► Encoder ─┐
▼
Decoder ◄──── Masked Attention
◄──── Target Tokens (Hindi)
▼
Linear + Softmax ─► Prediction


- **Encoder**: Multi-head self-attention + feedforward
- **Decoder**: Masked self-attention, encoder-decoder attention, and feedforward
- **Loss**: CrossEntropy with label smoothing and `[PAD]` masking

## 🧪 Sample Results

| English Input             | Hindi Translation           |
|--------------------------|-----------------------------|
| What is your name?       | आपका नाम क्या है ?         |
| My name is Gaurav        | मेरा नाम गौरव है           |
| Today is a good day      | आज एक अच्छा दिन है        |
| Hello                    | नमस्ते                      |


🔧 Configuration
Defined in config.py:

python
{
  "batch_size": 16,
  "num_epochs": 1,
  "lr": 5e-4,
  "seq_len": 128,
  "d_model": 256,
  "datasource": "cfilt/iitb-english-hindi",
  "lang_src": "en",
  "lang_tgt": "hi"
}

📈 Training
To train the model from scratch:

python3 train.py
Tokenizers are built and cached

Model weights are saved at 10% intervals and once at the end of training

TensorBoard logs are written to runs/

📤 Inference
Once training is complete:

python3 translate.py

📁 Project Structure
├── config.py              # All hyperparameters and paths
├── train.py               # Training script
├── translate.py           # Inference script
├── model.py               # Transformer architecture
├── dataset.py             # Data pipeline and collators
├── weights/               # Saved model checkpoints
├── runs/                  # TensorBoard logs
└── tokenizer_en.json / tokenizer_hi.json
🧠 Citation & Reference
Vaswani et al. (2017): Attention Is All You Need
https://arxiv.org/abs/1706.03762
