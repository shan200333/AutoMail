<div align="center">

<img src="https://img.shields.io/badge/-%E2%9C%89%EF%B8%8F%20AutoMail-0037A4?style=for-the-badge&labelColor=000d3a&color=0037A4" alt="AutoMail" height="45"/>

# AutoMail — Automated Email Reply Generation

**A comparative study of Retrieval, Graph Neural Networks, and LLM Fine-Tuning for automated email reply generation on the Enron Email Dataset**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.4x-FFD21E?style=flat-square)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)](LICENSE)

<br/>

| 🟢 SBERT Baseline | 🟣 GNN GraphSAGE | 🔵 MMR Retrieval | 🔴 Phi-3.5 QLoRA |
|:-:|:-:|:-:|:-:|
| Recall@1 `0.115` | Recall@1 `0.003` ⚠️ | ROUGE-1 `0.412` | Generative LLM |
| MRR `0.177` | Over-smoothing finding | Diversity `0.923` | 3.8B params, 8GB VRAM |

<br/>

[📋 Overview](#-overview) • [🚀 Quick Start](#-quick-start) • [📊 Results](#-results) • [🏗️ Architecture](#️-architecture) • [📁 Structure](#-project-structure) • [⚙️ Installation](#️-installation) • [🖥️ Demo](#️-demo) • [👥 Team](#-team)

</div>

---

## 📋 Overview

AutoMail is an end-to-end email reply generation system built on the **Enron Email Dataset** (517,401 real corporate emails). The project implements and rigorously compares **four technically distinct AI approaches** for generating contextually appropriate email replies — providing empirically grounded insights into the trade-offs between retrieval, graph-based learning, and generative paradigms.

### Why AutoMail?

> Professionals spend ~28% of their working week on email. Most automated systems only produce short, generic responses like *"Sounds good!"* — AutoMail generates full-length, contextually relevant replies.

### Key Contributions

- 📦 **48,184 email-reply pairs** extracted from 517,401 raw emails using novel subject-thread reconstruction (no `In-Reply-To` headers available)
- ⚠️ **GNN over-smoothing discovery** — first quantified demonstration of this failure mode in cross-modal email retrieval (Recall@1 degraded 96.9%)
- 🎯 **ROUGE-1 of 0.412** with diversity score 0.923 via K-Means + MMR cluster-based generation
- 💻 **LLM fine-tuning on consumer GPU** — Phi-3.5-mini (3.8B params) adapted with QLoRA within 8 GB VRAM
- 🖥️ **Unified Streamlit demo** comparing all 4 models side-by-side in real time

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/automail.git
cd automail

# 2. Create environment
conda create -n automail-env python=3.10
conda activate automail-env

# 3. Install
pip install -r requirements.txt

# 4. Place dataset
# Download emails.csv from https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
# Put it in the project root

# 5. Run pipeline (Parts 1–6 in order)
jupyter notebook

# 6. Launch demo (after running all parts)
streamlit run automail_demo.py
```

> **Windows users:** See [Windows-Specific Fixes](#-windows-specific-fixes) before running Part 6.

---

## 📊 Results

### Retrieval Evaluation *(n = 2,000 held-out pairs)*

| Metric | 🟢 SBERT Baseline | 🟣 GNN GraphSAGE | Δ |
|--------|:-----------------:|:----------------:|:-:|
| **Recall@1** | **0.1150** | 0.0035 | ↓ 0.1115 |
| Recall@3 | **0.2190** | 0.0140 | ↓ 0.2050 |
| Recall@5 | **0.2435** | 0.0200 | ↓ 0.2235 |
| Recall@10 | **0.2665** | 0.0285 | ↓ 0.2380 |
| **MRR** | **0.1695** | 0.0133 | ↓ 0.1562 |

> **⚠️ GNN Negative Finding:** The 5-NN email similarity graph causes GraphSAGE to homogenise email embeddings via neighbourhood aggregation — destroying the discriminative signal SBERT embeddings already possess. The learnable blend parameter α converged to 0.496 but could not overcome the structural over-smoothing. Training loss *did* decrease (0.35 → 0.12), confirming the model trains correctly — the failure is architectural, not a bug.

### Generation Evaluation

| Metric | 🔵 MMR Retrieval *(n=500)* | 🔴 Phi-3.5 QLoRA *(n=200)* |
|--------|:---:|:---:|
| **ROUGE-1** | **0.4122** | *(add your score)* |
| ROUGE-2 | **0.3054** | *(add your score)* |
| ROUGE-L | **0.3691** | *(add your score)* |
| **Diversity** | **0.9228** | — |
| Inference | **< 0.1s** | ~20s |

### Embedding Quality

```
SBERT Sanity Check
──────────────────────────────────────────
Paired cosine similarity  :  0.439 ± 0.276
Random cosine similarity  :  0.169 ± 0.116
Signal-to-Noise Ratio     :  2.59×  ✅
```

### Complete Model Comparison

| Model | Task | Best Metric | Score | Notes |
|-------|------|------------|-------|-------|
| 🟢 SBERT Baseline | Retrieval | Recall@1 / MRR | `0.115` / `0.177` | Best retrieval; fastest |
| 🟣 GNN GraphSAGE | Retrieval | Recall@1 / MRR | `0.003` / `0.013` | ⚠️ Over-smoothing |
| 🔵 MMR Retrieval | Generation | ROUGE-1 | `0.412` | Best overall; real-time |
| 🔴 Phi-3.5 QLoRA | Generation | ROUGE-1 | *(your score)* | Fluent; generative |
| Gmail Smart Reply | Industrial | Short replies | N/A | Proprietary; brief only |

---

## 🏗️ Architecture

### Six-Stage Pipeline

```
emails.csv  (517,401 emails, ~1.7 GB)
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  PART 1 — Data Preparation                              │
│  Parse RFC-2822 → 7-step body cleaning                  │
│  Subject normalisation → chronological pairing          │
│  Output: email_reply_pairs.csv  (48,184 pairs)          │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  PART 2 — Embedding Generation                          │
│  Word2Vec Skip-gram  : 128d, 33,152 vocab               │
│  SBERT MiniLM-L6-v2  : 384d, L2-normalised, GPU         │
│  SNR = 2.59× (paired vs random cosine sim)              │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  PART 3 — Bipartite Graph                               │
│  G = (E ∪ R, W)  |  τ = 0.30  |  29,321 edges          │
│  Interactive Plotly HTML + static matplotlib            │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  PART 4 — GNN GraphSAGE                                 │
│  5-NN cosine graph → SAGEConv(384→256→128)              │
│  Residual skip + learnable α + triplet loss             │
│  ⚠️ Finding: over-smoothing (Recall@1: 0.115 → 0.003)  │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  PART 5 — K-Means + MMR Generation                      │
│  50 clusters (MiniBatchKMeans) + reply template banks   │
│  MMR: score = 0.4×relevance − 0.6×redundancy           │
│  5 diverse candidates per query                         │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  PART 6 — Phi-3.5 QLoRA Fine-Tuning                    │
│  Phi-3.5-mini-instruct (3.8B) + 4-bit NF4              │
│  LoRA: r=16, α=32, ~27M trainable (0.71%)              │
│  1 epoch, bf16, ~2.5 hrs on RTX 4060                   │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  automail_demo.py — Streamlit App                       │
│  All 4 models | live input | side-by-side comparison    │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

<details>
<summary><b>Step 1 — Prerequisites</b></summary>

- Python 3.10 (Anaconda recommended)
- NVIDIA GPU, CUDA 12.x
- 8 GB+ VRAM (for Part 6 and demo)
- 16 GB+ RAM
- ~15 GB free disk space

</details>

<details>
<summary><b>Step 2 — Create environment</b></summary>

```bash
conda create -n automail-env python=3.10
conda activate automail-env
```

</details>

<details>
<summary><b>Step 3 — Install PyTorch + PyG</b></summary>

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

</details>

<details>
<summary><b>Step 4 — Install all other dependencies</b></summary>

```bash
pip install transformers==4.44.0 accelerate bitsandbytes peft datasets trl
pip install sentence-transformers gensim nltk
pip install pandas numpy scikit-learn networkx plotly matplotlib
pip install rouge-score streamlit
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

</details>

<details>
<summary><b>Step 5 — Download dataset</b></summary>

1. Go to [Kaggle — Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
2. Download `emails.csv` (~1.7 GB)
3. Place it in the project root directory

</details>

---

## 🖥️ Demo

After running all 6 notebooks:

```bash
conda activate automail-env
streamlit run automail_demo.py
# Opens at http://localhost:8501
```

The demo loads all saved model artefacts — **no retraining needed**.

### Sample inputs to try

<details>
<summary><b>📧 Trading / Price Query</b></summary>

```
Hi,

I wanted to follow up on the gas price contract we discussed last week.
Could you confirm the delivery point and whether the price is fixed at $3.25
or indexed to the daily market? We need this finalized before end of day Friday.

Thanks, Mark
```

</details>

<details>
<summary><b>📧 Meeting Request</b></summary>

```
John,

Can we schedule a call this week to go over the quarterly numbers?
I have reviewed the report and have a few questions about the Houston figures.
Tuesday or Wednesday afternoon works for me.

Let me know, Sarah
```

</details>

<details>
<summary><b>📧 Document Approval</b></summary>

```
Team,

Please review the attached contract and send me your comments by Thursday.
Legal needs the signed copy no later than next Monday. If you have any issues
with the terms, flag them immediately.

Regards, David
```

</details>

---

## 📁 Project Structure

```
automail/
│
├── 📓 Notebooks
│   ├── part1_data_prep.ipynb          # Email parsing + pair extraction
│   ├── part2_embeddings.ipynb         # Word2Vec + SBERT embeddings
│   ├── part3_bipartite_graph.ipynb    # Graph construction + visualisation
│   ├── part4_gnn.ipynb                # GraphSAGE training + evaluation
│   ├── part5_multireply.ipynb         # K-Means + MMR generation
│   └── part6_llm_finetune.ipynb       # Phi-3.5 QLoRA fine-tuning
│
├── 🖥️  automail_demo.py               # Streamlit demo (all 4 models)
│
├── 📁 embeddings/                     # Generated by Parts 1–4
│   ├── word2vec.model
│   ├── email_sbert.npy               # (48184 × 384)
│   ├── reply_sbert.npy               # (48184 × 384)
│   ├── email_gnn.npy                 # (48184 × 128)
│   ├── reply_proj.npy                # (48184 × 128)
│   └── tokens.pkl
│
├── 📁 presentation/                   # Generated by Parts 4–6
│   ├── models/
│   │   ├── gnn_model.pt
│   │   ├── reply_projector.pt
│   │   ├── kmeans.pkl
│   │   ├── cluster_bank.pkl
│   │   └── phi3_lora/                # LoRA adapter + tokeniser (~50 MB)
│   └── results/
│       ├── part4_gnn_results.json
│       ├── part5_results.json
│       └── part6_results.json
│
├── 📄 email_reply_pairs.csv           # 48,184 pairs (generated by Part 1)
├── 📄 emails.csv                      # ← Download separately (1.7 GB)
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🔧 Windows-Specific Fixes

<details>
<summary><b>Fix 1 — TRL UnicodeDecodeError on import</b></summary>

**Error:** `RuntimeError: UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`

**Cause:** TRL reads internal Jinja templates using Windows `cp1252` encoding.

**Fix:** Add this at the top of Part 6 notebook, before any TRL import:

```python
import pathlib
_orig = pathlib.Path.read_text
def _utf8(self, encoding=None, errors=None):
    return _orig(self, encoding=encoding or 'utf-8', errors=errors)
pathlib.Path.read_text = _utf8
```

</details>

<details>
<summary><b>Fix 2 — KeyError: 'type' in rope_scaling</b></summary>

**Error:** `KeyError: 'type'` during model loading

**Cause:** Stale cached HuggingFace model files with outdated `rope_scaling` format.

**Fix:**

```python
import shutil, os
stale = os.path.expanduser(r"~\.cache\huggingface\modules\transformers_modules\microsoft")
if os.path.exists(stale):
    shutil.rmtree(stale)

# Also patch config before loading:
from transformers import AutoConfig
config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
if hasattr(config, "rope_scaling") and "type" not in config.rope_scaling:
    config.rope_scaling["type"] = "longrope"
```

</details>

<details>
<summary><b>Fix 3 — OSError 1455: Paging file too small</b></summary>

**Error:** `OSError: The paging file is too small for this operation to complete`

**Cause:** Windows virtual memory exhausted loading 2.4 GB model weights.

**Fix:**
1. `Win + R` → `sysdm.cpl` → Advanced → Performance → Settings
2. Advanced → Virtual Memory → Change
3. Set C: drive custom size: Initial = **8192**, Maximum = **16384** (MB)
4. Restart PC

Also add `low_cpu_mem_usage=True` to `from_pretrained()`.

</details>

<details>
<summary><b>Fix 4 — RuntimeError: BFloat16 gradient scaler</b></summary>

**Error:** `RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`

**Cause:** `fp16=True` conflicts with bfloat16 internal compute in 4-bit quantisation.

**Fix:** In `SFTConfig`, replace `fp16=True` with `bf16=True`.

</details>

<details>
<summary><b>Fix 5 — AttributeError: seen_tokens</b></summary>

**Error:** `AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'`

**Cause:** Stale remote model code cached from older transformers version.

**Fix:** Use `trust_remote_code=False` in `from_pretrained()`.

</details>

<details>
<summary><b>Fix 6 — TypeError: max_seq_length</b></summary>

**Error:** `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`

**Cause:** Newer TRL renamed the parameter.

**Fix:** Replace `max_seq_length=` with `max_length=` in `SFTConfig`.

</details>

---

## 🧰 Tech Stack

| Category | Tool | Version | Purpose |
|----------|------|---------|---------|
| Language | Python | 3.10 | All pipeline notebooks |
| Deep Learning | PyTorch | 2.x | GPU-accelerated training |
| GNN | PyTorch Geometric | 2.x | GraphSAGE, SAGEConv |
| LLM | HuggingFace Transformers | 4.44.x | Phi-3.5 loading |
| Fine-Tuning | PEFT + TRL | latest | QLoRA adapters + SFTTrainer |
| Quantisation | BitsAndBytes | 0.41+ | 4-bit NF4 |
| Embeddings | sentence-transformers | 2.x | SBERT all-MiniLM-L6-v2 |
| Word Embeddings | Gensim | 4.x | Word2Vec Skip-gram |
| Clustering | scikit-learn | 1.x | MiniBatchKMeans, NearestNeighbors |
| Graph | NetworkX | 3.x | Bipartite graph construction |
| Visualisation | Plotly + Matplotlib | 5.x | Interactive + static charts |
| Evaluation | rouge-score | — | ROUGE-1/2/L F1 |
| Demo | Streamlit | 1.x | Web interface |

---

## 📚 References

1. A. Kannan et al., *"Smart Reply: Automated Response Suggestion for Email,"* KDD, 2016.
2. N. Reimers & I. Gurevych, *"Sentence-BERT,"* EMNLP, 2019.
3. W. L. Hamilton et al., *"Inductive Representation Learning on Large Graphs (GraphSAGE),"* NeurIPS, 2017.
4. Q. Li et al., *"Deeper Insights into GCNs (Over-Smoothing),"* AAAI, 2018.
5. E. J. Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models,"* ICLR, 2022.
6. T. Dettmers et al., *"QLoRA: Efficient Finetuning of Quantized LLMs,"* NeurIPS, 2023.
7. Microsoft Research, *"Phi-3 Technical Report,"* arXiv:2404.14219, 2024.
8. J. Carbonell & J. Goldstein, *"The Use of MMR,"* SIGIR, 1998.
9. P. Lewis et al., *"Retrieval-Augmented Generation,"* NeurIPS, 2020.
10. B. Klimt & Y. Yang, *"The Enron Corpus,"* ECML, 2004.

---

## 👥 Team

<div align="center">

| | Name | Roll No. | Contribution |
|:-:|------|----------|-------------|
| 👤 | **Anshul Rohru** | 221030085 | Parts 1–2: Data prep, email parsing, embedding generation |
| 👤 | **Palak Mutreja** | 221030350 | Part 3: Bipartite graph, visualisation, literature survey |
| 👤 | **Shantanu Tripathi** | 221031064 | Part 4: GNN GraphSAGE, triplet loss, over-smoothing analysis |
| 👤 | **Kartikeya Sharma** | 221031051 | Parts 5–6: MMR generation, Phi-3.5 QLoRA, Streamlit demo |

**Supervisor:** Dr. Rakesh Kanji — Assistant Professor-SG, Dept. of CSE & IT, JUIT Waknaghat

</div>

---

## 🏛️ Institution

<div align="center">

**Jaypee University of Information Technology**
Department of Computer Science & Engineering and Information Technology
Waknaghat, Solan — 173234, Himachal Pradesh, India

*Major Project II (18B19CI891) | AY 2025–26 | Group No.: G-50*

---

*Made with ❤️ at JUIT Waknaghat*

</div>
