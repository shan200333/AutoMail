"""
AutoMail — Final Demo (All 4 Models)
======================================
Run with:
    conda activate flan-env
    streamlit run automail_demo.py

Models shown:
  1. SBERT Baseline   — nearest-neighbour retrieval (Part 4)
  2. GNN GraphSAGE    — GNN-refined retrieval (Part 4)
  3. MMR Retrieval    — cluster + diversity (Part 5)
  4. Phi-3.5 LLM      — fine-tuned generative model (Part 6)
"""

# ── UTF-8 fix for Windows TRL (before any transformers import) ─────────────
import pathlib
_orig = pathlib.Path.read_text
def _utf8(self, encoding=None, errors=None):
    return _orig(self, encoding=encoding or "utf-8", errors=errors)
pathlib.Path.read_text = _utf8

import os, json, pickle, shutil, time, warnings, gc
os.environ["PYTHONUTF8"]              = "1"
os.environ["PYTHONIOENCODING"]        = "utf-8"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "AutoMail — All Models Demo",
    page_icon  = "✉️",
    layout     = "wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.model-box {
    background: #1a1d23;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 5px solid;
}
.box-sbert { border-color: #06D6A0; }
.box-gnn   { border-color: #8338EC; }
.box-mmr   { border-color: #3A86FF; }
.box-phi   { border-color: #FF6B6B; }
.model-label {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.pill {
    display: inline-block;
    background: #2a2d35;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.73rem;
    margin-right: 5px;
    margin-bottom: 8px;
    color: #aaa;
}
.reply-text {
    font-size: 0.91rem;
    line-height: 1.65;
    color: #e0e0e0;
    white-space: pre-wrap;
}
.rank-label { color: #777; font-size: 0.76rem; margin-bottom: 4px; }
.header-grad {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg,#06D6A0,#3A86FF,#FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
PAIRS_CSV  = "email_reply_pairs.csv"
EMB_DIR    = "embeddings"
PRES_DIR   = "presentation"
MODEL_DIR  = f"{PRES_DIR}/models/phi3_lora"
LOG_DIR    = f"{PRES_DIR}/results"
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_dataset_and_embeddings():
    df          = pd.read_csv(PAIRS_CSV)
    email_sbert = np.load(f"{EMB_DIR}/email_sbert.npy").astype(np.float32)
    reply_sbert = np.load(f"{EMB_DIR}/reply_sbert.npy").astype(np.float32)
    gnn_path    = f"{EMB_DIR}/email_gnn.npy"
    proj_path   = f"{EMB_DIR}/reply_proj.npy"
    email_gnn   = np.load(gnn_path).astype(np.float32)  if os.path.exists(gnn_path)  else None
    reply_proj  = np.load(proj_path).astype(np.float32) if os.path.exists(proj_path) else None
    return df, email_sbert, reply_sbert, email_gnn, reply_proj


@st.cache_resource(show_spinner=False)
def load_sbert_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2", device=str(DEVICE))


@st.cache_resource(show_spinner=False)
def load_mmr_models():
    with open(f"{PRES_DIR}/models/kmeans.pkl",       "rb") as f: kmeans = pickle.load(f)
    with open(f"{PRES_DIR}/models/cluster_bank.pkl", "rb") as f: bank   = pickle.load(f)
    for cid in bank:
        if isinstance(bank[cid]['reply_embs'], list):
            bank[cid]['reply_embs'] = np.array(bank[cid]['reply_embs'], dtype=np.float32)
        if isinstance(bank[cid]['centroid'], list):
            bank[cid]['centroid']   = np.array(bank[cid]['centroid'],   dtype=np.float32)
    return kmeans, bank


@st.cache_resource(show_spinner=False)
def load_phi_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    stale = os.path.expanduser(
        r"~\.cache\huggingface\modules\transformers_modules\microsoft"
    )
    if os.path.exists(stale):
        shutil.rmtree(stale)

    bnb = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = bnb,
        device_map          = "auto",
        trust_remote_code   = False,
        dtype               = torch.bfloat16,
        attn_implementation = "eager",
        low_cpu_mem_usage   = True,
    )
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    tok   = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    tok.pad_token = tok.eos_token
    model.eval()
    return model, tok

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def sbert_retrieve(query_emb, email_embs, reply_texts, top_k=3):
    sims    = email_embs @ query_emb
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{"rank": i+1, "reply": reply_texts[idx], "similarity": float(sims[idx])}
            for i, idx in enumerate(top_idx)]


def gnn_retrieve(query_emb, email_gnn, reply_texts, top_k=3):
    # project query to GNN output dim
    out_dim = email_gnn.shape[1]
    in_dim  = query_emb.shape[0]
    if out_dim != in_dim:
        q = query_emb[:out_dim] if out_dim < in_dim else np.pad(query_emb, (0, out_dim - in_dim))
    else:
        q = query_emb.copy()
    q = q / (np.linalg.norm(q) + 1e-9)

    sims    = email_gnn @ q
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{"rank": i+1, "reply": reply_texts[idx], "similarity": float(sims[idx])}
            for i, idx in enumerate(top_idx)]


def mmr_retrieve(query_emb, kmeans, bank, top_k=5, alpha=0.4):
    cid = int(kmeans.predict(query_emb.reshape(1, -1))[0])
    if cid not in bank:
        dists = np.array([np.linalg.norm(bank[c]['centroid'] - query_emb) for c in bank])
        cid   = list(bank.keys())[int(np.argmin(dists))]

    b         = bank[cid]
    r_embs    = b['reply_embs']
    replies   = b['replies']
    relevance = r_embs @ query_emb

    selected, sel_embs, remaining = [], [], list(range(len(replies)))
    for _ in range(min(top_k, len(remaining))):
        if not sel_embs:
            best_i = remaining[int(np.argmax(relevance[remaining]))]
        else:
            sm     = np.array(sel_embs)
            scores = [alpha * relevance[i] - (1-alpha) * (r_embs[i] @ sm.T).max()
                      for i in remaining]
            best_i = remaining[int(np.argmax(scores))]
        selected.append({"rank": len(selected)+1, "reply": replies[best_i],
                          "relevance": float(relevance[best_i])})
        sel_embs.append(r_embs[best_i])
        remaining.remove(best_i)

    return selected, int(cid), b['size']


def phi_generate(email_text, model, tok, max_new_tokens=200):
    prompt = (
        f"<|user|>\nYou are an email assistant. Write a professional reply "
        f"to the following email.\n\nEmail:\n{email_text.strip()[:600]}\n"
        f"<|end|>\n<|assistant|>\n"
    )
    inputs = tok(prompt, return_tensors="pt",
                 truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens  = max_new_tokens,
            do_sample       = False,
            pad_token_id    = tok.eos_token_id,
            eos_token_id    = tok.eos_token_id,
            use_cache       = True,
            past_key_values = None,
        )
    gen = out[0][inputs['input_ids'].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def render_candidate(item, box_class, score_label):
    score = item.get("similarity", item.get("relevance", 0))
    pct   = int(max(0, min(100, score * 100)))
    text  = str(item["reply"])[:600]
    st.markdown(
        f'<div class="model-box {box_class}">'
        f'<div class="rank-label">Candidate {item["rank"]} — {score_label}: {pct}%</div>'
        f'<div class="reply-text">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✉️ AutoMail")
    st.markdown("**All 4 Models Demo**")
    st.markdown("---")

    try:
        with open(f"{LOG_DIR}/part6_results.json") as f:
            r6 = json.load(f)
        cmp = r6.get("comparison", {})
        sb  = cmp.get("sbert_baseline",  {})
        gn  = cmp.get("gnn_graphsage",   {})
        mm  = cmp.get("mmr_retrieval",   {})
        ph  = cmp.get("phi35_finetuned", {})

        st.markdown("### 📊 Eval Scores")
        st.markdown("🟢 **SBERT Baseline (Part 4)**")
        st.markdown(f"Recall@1: `{sb.get('recall@1',0):.4f}` · MRR: `{sb.get('mrr',0):.4f}`")
        st.markdown("🟣 **GNN GraphSAGE (Part 4)**")
        st.markdown(f"Recall@1: `{gn.get('recall@1',0):.4f}` · MRR: `{gn.get('mrr',0):.4f}`")
        st.caption("Over-smoothing — worse than SBERT")
        st.markdown("🔵 **MMR Retrieval (Part 5)**")
        st.markdown(f"R1:`{mm.get('rouge1',0):.3f}` R2:`{mm.get('rouge2',0):.3f}` RL:`{mm.get('rougeL',0):.3f}`")
        st.markdown("🔴 **Phi-3.5 Fine-tuned (Part 6)**")
        st.markdown(f"R1:`{ph.get('rouge1',0):.3f}` R2:`{ph.get('rouge2',0):.3f}` RL:`{ph.get('rougeL',0):.3f}`")
    except FileNotFoundError:
        st.warning("Run parts 4–6 first to see scores here.")

    st.markdown("---")
    gpu  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0
    st.markdown(f"**GPU**: {gpu}")
    st.markdown(f"**VRAM**: {vram:.1f} GB")
    st.caption("Enron Email Dataset · AutoMail Project")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="header-grad">✉️ AutoMail — All Models Demo</div>',
            unsafe_allow_html=True)
st.markdown("Compare **SBERT · GNN · MMR · Phi-3.5** on any email")
st.markdown("---")

# toggles
c1, c2, c3, c4, c5 = st.columns(5)
with c1: use_sbert = st.checkbox("🟢 SBERT Baseline", value=True)
with c2: use_gnn   = st.checkbox("🟣 GNN GraphSAGE",  value=True)
with c3: use_mmr   = st.checkbox("🔵 MMR Retrieval",  value=True)
with c4: use_phi   = st.checkbox("🔴 Phi-3.5 LLM",    value=True)
with c5: top_k     = st.slider("Top-K replies", 1, 5, 3)

st.markdown("---")

EXAMPLES = {
    "Trading — Price Query": (
        "Hi,\n\nI wanted to follow up on the gas price contract we discussed "
        "last week. Could you confirm the delivery point and whether the price "
        "is fixed at $3.25 or indexed to the daily market? We need this "
        "finalized before end of day Friday.\n\nThanks,\nMark"
    ),
    "Meeting Request": (
        "John,\n\nCan we schedule a call this week to go over the quarterly "
        "numbers? I have reviewed the report and have a few questions about "
        "the Houston figures. Tuesday or Wednesday afternoon works for me.\n\n"
        "Let me know,\nSarah"
    ),
    "Document Approval": (
        "Team,\n\nPlease review the attached contract and send me your comments "
        "by Thursday. Legal needs the signed copy no later than next Monday. "
        "If you have any issues with the terms, flag them immediately.\n\n"
        "Regards,\nDavid"
    ),
    "Quick Yes/No": (
        "Hi,\n\nAre you available to cover the desk on Friday? Mike is out "
        "and we need someone to handle the afternoon trades. Let me know as "
        "soon as possible.\n\nThanks"
    ),
    "System Issue": (
        "Hello,\n\nI am getting an error when trying to access the risk "
        "management system this morning. It says my credentials have expired "
        "but IT has not responded yet. Can you help me get access restored "
        "before the 9am meeting?\n\nThanks,\nKaren"
    ),
    "✏️ Custom (type below)": "",
}

choice      = st.selectbox("📧 Pick an example or write your own:", list(EXAMPLES.keys()))
email_input = st.text_area("Email to reply to:", value=EXAMPLES[choice],
                            height=160, max_chars=1200,
                            placeholder="Paste any email here...")
run_btn     = st.button("🚀 Generate Replies from All Models",
                         type="primary", disabled=not email_input.strip())

# ── Run ─────────────────────────────────────────────────────────────────────
if run_btn and email_input.strip():

    st.markdown("---")

    with st.spinner("Loading dataset + embeddings..."):
        df, email_sbert, reply_sbert, email_gnn, reply_proj = \
            load_dataset_and_embeddings()
        reply_texts = df['reply_body'].tolist()

    with st.spinner("Loading SBERT encoder..."):
        sbert_enc = load_sbert_encoder()

    query_emb = sbert_enc.encode(
        [email_input], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False
    )[0].astype(np.float32)

    results = {}

    if use_sbert:
        with st.spinner("🟢 SBERT retrieving..."):
            t0 = time.time()
            results["sbert"] = {"items": sbert_retrieve(query_emb, email_sbert,
                                                         reply_texts, top_k),
                                  "time": time.time()-t0}

    if use_gnn:
        if email_gnn is not None:
            with st.spinner("🟣 GNN retrieving..."):
                t0 = time.time()
                results["gnn"] = {"items": gnn_retrieve(query_emb, email_gnn,
                                                          reply_texts, top_k),
                                   "time": time.time()-t0}
        else:
            st.warning("🟣 GNN embeddings not found (`embeddings/email_gnn.npy`). Skipping.")

    if use_mmr:
        with st.spinner("🔵 MMR running..."):
            try:
                kmeans, bank = load_mmr_models()
                t0 = time.time()
                items, cid, csz = mmr_retrieve(query_emb, kmeans, bank, top_k)
                results["mmr"] = {"items": items, "time": time.time()-t0,
                                   "cluster_id": cid, "cluster_size": csz}
            except Exception as e:
                st.error(f"MMR error: {e}")

    if use_phi:
        with st.spinner("🔴 Phi-3.5 generating (15–30s)..."):
            try:
                phi_model, phi_tok = load_phi_model()
                t0 = time.time()
                phi_text = phi_generate(email_input, phi_model, phi_tok)
                results["phi"] = {"text": phi_text, "time": time.time()-t0}
            except Exception as e:
                st.error(f"Phi-3.5 error: {e}")

    # display
    active = [k for k in ["sbert","gnn","mmr","phi"] if k in results]
    if not active:
        st.error("No models ran."); st.stop()

    cols = st.columns(len(active))

    for col, key in zip(cols, active):
        with col:
            res = results[key]

            if key == "sbert":
                st.markdown(
                    f'<div class="model-box box-sbert">'
                    f'<div class="model-label" style="color:#06D6A0">🟢 SBERT Baseline — Part 4</div>'
                    f'<span class="pill">Nearest Neighbour</span>'
                    f'<span class="pill">Recall@1: 0.115</span>'
                    f'<span class="pill">⏱ {res["time"]:.2f}s</span>'
                    f'</div>', unsafe_allow_html=True)
                st.markdown(f"**Top {len(res['items'])} similar replies:**")
                for item in res["items"]:
                    render_candidate(item, "box-sbert", "Similarity")

            elif key == "gnn":
                st.markdown(
                    f'<div class="model-box box-gnn">'
                    f'<div class="model-label" style="color:#8338EC">🟣 GNN GraphSAGE — Part 4</div>'
                    f'<span class="pill">Graph Refined</span>'
                    f'<span class="pill">Recall@1: 0.003 ⚠️</span>'
                    f'<span class="pill">⏱ {res["time"]:.2f}s</span>'
                    f'</div>', unsafe_allow_html=True)
                st.caption("⚠️ Over-smoothed embeddings — included for comparison")
                st.markdown(f"**Top {len(res['items'])} results:**")
                for item in res["items"]:
                    render_candidate(item, "box-gnn", "Similarity")

            elif key == "mmr":
                st.markdown(
                    f'<div class="model-box box-mmr">'
                    f'<div class="model-label" style="color:#3A86FF">🔵 MMR Retrieval — Part 5</div>'
                    f'<span class="pill">Cluster {res["cluster_id"]} ({res["cluster_size"]} emails)</span>'
                    f'<span class="pill">ROUGE-1: 0.41</span>'
                    f'<span class="pill">⏱ {res["time"]:.2f}s</span>'
                    f'</div>', unsafe_allow_html=True)
                st.markdown(f"**{len(res['items'])} diverse candidates (MMR):**")
                for item in res["items"]:
                    render_candidate(item, "box-mmr", "Relevance")

            elif key == "phi":
                st.markdown(
                    f'<div class="model-box box-phi">'
                    f'<div class="model-label" style="color:#FF6B6B">🔴 Phi-3.5 Fine-tuned — Part 6</div>'
                    f'<span class="pill">QLoRA · 3.8B params</span>'
                    f'<span class="pill">Generative LLM</span>'
                    f'<span class="pill">⏱ {res["time"]:.1f}s</span>'
                    f'</div>', unsafe_allow_html=True)
                st.markdown("**Generated reply:**")
                st.markdown(
                    f'<div class="model-box box-phi" style="border-width:2px">'
                    f'<div class="reply-text">{res["text"]}</div>'
                    f'</div>', unsafe_allow_html=True)
                st.code(res["text"], language=None)

    # summary bar
    st.markdown("---")
    st.markdown("### 📊 Model Comparison Summary")
    s1, s2, s3, s4 = st.columns(4)
    for col, (name, task, metric, score, color) in zip(
        [s1,s2,s3,s4],
        [("🟢 SBERT",  "Retrieval",  "Recall@1", "0.115",    "#06D6A0"),
         ("🟣 GNN",    "Retrieval",  "Recall@1", "0.003 ⚠️", "#8338EC"),
         ("🔵 MMR",    "Retrieval+", "ROUGE-1",  "0.412",    "#3A86FF"),
         ("🔴 Phi-3.5","Generative", "ROUGE-1",  "see eval", "#FF6B6B")],
    ):
        with col:
            st.markdown(
                f'<div class="model-box" style="border-color:{color};text-align:center">'
                f'<div style="font-size:1rem;font-weight:800;color:{color}">{name}</div>'
                f'<div style="color:#888;font-size:0.78rem">{task}</div>'
                f'<div style="font-size:1.5rem;font-weight:800;margin:6px 0">{score}</div>'
                f'<div style="color:#666;font-size:0.73rem">{metric}</div>'
                f'</div>', unsafe_allow_html=True)

elif not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:60px 0;color:#444">
        <div style="font-size:3.5rem">✉️</div>
        <div style="font-size:1.1rem;margin-top:12px;color:#666">
            Select an example email or type your own,<br>
            then click <strong style="color:#aaa">Generate Replies from All Models</strong>
        </div>
        <div style="margin-top:16px;color:#555;font-size:0.85rem">
            🟢 SBERT &nbsp;·&nbsp; 🟣 GNN &nbsp;·&nbsp; 🔵 MMR &nbsp;·&nbsp; 🔴 Phi-3.5
        </div>
    </div>
    """, unsafe_allow_html=True)