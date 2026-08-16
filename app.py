import json
import os
import re
import time
from pathlib import Path

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="ChiaroSubito Benchmark", page_icon="🧪", layout="wide")

API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

MODELS = [
    ("GPT-5.6 Sol", "gpt-5.6-sol"),
    ("GPT-5.6 Terra", "gpt-5.6-terra"),
    ("GPT-5.6 Luna", "gpt-5.6-luna"),
]

PROMPT = r"""
Sei ChiaroSubito, un assistente specializzato nella comprensione di documenti per persone non esperte.

Analizza esclusivamente il documento fornito. Non inventare informazioni.
Individua:
1. tipo e scopo del documento;
2. fatti importanti;
3. importi;
4. scadenze;
5. obblighi e diritti;
6. anomalie o elementi da verificare;
7. azioni pratiche consigliate;
8. ciò che il documento non permette di determinare.

Distingui sempre:
FACT = informazione esplicitamente presente;
CALCULATION = calcolo derivato dai dati;
INTERPRETATION = interpretazione basata sul testo;
UNKNOWN = non determinabile.

Per ogni informazione importante indica la pagina quando possibile.
Scrivi in italiano semplice e diretto.

Restituisci esclusivamente JSON.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "document_type": {"type": "string"},
        "document_purpose": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "deadlines": {"type": "array", "items": {"type": "string"}},
        "checks": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "unknowns": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "document_type", "document_purpose", "confidence", "summary",
        "facts", "deadlines", "checks", "actions", "unknowns"
    ],
}

def extract_pdf(uploaded):
    reader = PdfReader(uploaded)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        pages.append(f"[PAGINA {i}]\n{text}")
    return "\n\n".join(pages), len(pages)

def estimate_cost(usage, model):
    if not usage:
        return None
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cached = getattr(usage, "input_tokens_details", None)
    cached_tokens = 0
    if cached:
        cached_tokens = getattr(cached, "cached_tokens", 0) or 0
    prices = {
        "gpt-5.6-sol": (5.0, 30.0, 0.5),
        "gpt-5.6-terra": (2.0, 12.0, 0.2),
        "gpt-5.6-luna": (0.2, 1.2, 0.02),
    }
    if model not in prices:
        return None
    pin, pout, pcached = prices[model]
    uncached = max(inp - cached_tokens, 0)
    cost = (uncached / 1_000_000) * pin + (cached_tokens / 1_000_000) * pcached + (out / 1_000_000) * pout
    return {"input_tokens": inp, "output_tokens": out, "cached_tokens": cached_tokens, "usd": cost}

def run_model(client, model, document, effort):
    start = time.perf_counter()
    response = client.responses.create(
        model=model,
        instructions=PROMPT,
        input="DOCUMENTO:\n\n" + document,
        reasoning={"effort": effort},
        text={
            "format": {
                "type": "json_schema",
                "name": "chiarosubito_benchmark",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    )
    elapsed = time.perf_counter() - start
    data = json.loads(response.output_text)
    return elapsed, data, estimate_cost(getattr(response, "usage", None), model)

st.title("🧪 ChiaroSubito — Benchmark v0.2")
st.caption("Laboratorio esterno: stesso PDF + stesso prompt + stesso schema, cambia solo il modello.")

if not API_KEY:
    st.error("Manca OPENAI_API_KEY nei Secrets di Streamlit.")
    st.stop()

uploaded = st.file_uploader("Carica il PDF da confrontare", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    effort = st.selectbox(
        "Reasoning effort",
        ["none", "low", "medium"],
        index=1,
        help="Per un confronto orientato alla velocità, parti da 'low'."
    )
with col2:
    runs = st.number_input("Ripetizioni per modello", min_value=1, max_value=3, value=1)

st.info("Per il primo test ti consiglio: stessa bolletta, effort = low, 1 ripetizione.")

if not uploaded:
    st.stop()

try:
    document, pages = extract_pdf(uploaded)
except Exception as e:
    st.error(f"Impossibile leggere il PDF: {e}")
    st.stop()

st.write(f"**Documento:** {uploaded.name} · **Pagine:** {pages} · **Caratteri:** {len(document):,}")

if st.button("🚀 Avvia benchmark", type="primary"):
    client = OpenAI(api_key=API_KEY)
    all_results = []

    for label, model in MODELS:
        for run_no in range(1, int(runs) + 1):
            with st.spinner(f"{label} — test {run_no}/{runs}..."):
                try:
                    seconds, data, cost = run_model(client, model, document, effort)
                    all_results.append({
                        "label": label,
                        "model": model,
                        "run": run_no,
                        "seconds": seconds,
                        "data": data,
                        "cost": cost,
                        "error": None,
                    })
                except Exception as e:
                    all_results.append({
                        "label": label,
                        "model": model,
                        "run": run_no,
                        "seconds": None,
                        "data": None,
                        "cost": None,
                        "error": str(e),
                    })

    st.session_state["benchmark_results"] = all_results

results = st.session_state.get("benchmark_results")
if not results:
    st.stop()

st.divider()
st.header("📊 Risultati")

rows = []
for label, model in MODELS:
    rr = [x for x in results if x["model"] == model and x["seconds"] is not None]
    if not rr:
        rows.append({"Modello": label, "Tempo medio": "ERRORE", "Confidenza": "—", "Costo stimato": "—"})
        continue
    avg = sum(x["seconds"] for x in rr) / len(rr)
    conf = sum(x["data"]["confidence"] for x in rr) / len(rr)
    costs = [x["cost"]["usd"] for x in rr if x["cost"]]
    rows.append({
        "Modello": label,
        "Tempo medio": f"{avg:.2f} s",
        "Confidenza": f"{conf*100:.0f}%",
        "Costo stimato": f"${sum(costs)/len(costs):.5f}" if costs else "n/d",
    })

st.table(rows)

for label, model in MODELS:
    st.subheader(label)
    rr = [x for x in results if x["model"] == model]
    for x in rr:
        if x["error"]:
            st.error(f"Test {x['run']}: {x['error']}")
            continue
        d = x["data"]
        c = x["cost"]
        a,b,c1 = st.columns(3)
        a.metric("Tempo", f"{x['seconds']:.2f} s")
        b.metric("Confidenza", f"{d['confidence']*100:.0f}%")
        c1.metric("Costo stimato", f"${c['usd']:.5f}" if c else "n/d")
        st.markdown(f"**Tipo:** {d['document_type']}")
        st.markdown(f"**Sintesi:** {d['summary']}")
        if c:
            st.caption(f"Token input: {c['input_tokens']:,} · output: {c['output_tokens']:,} · cached: {c['cached_tokens']:,}")
        with st.expander("Dettaglio analisi"):
            st.json(d)

st.divider()
st.header("🧭 Come decidiamo")
st.markdown("""
Non scegliamo il modello solo perché è più veloce.

Valutiamo insieme:
- **tempo**;
- **costo**;
- **accuratezza dei fatti**;
- **scadenze individuate**;
- **anomalie individuate**;
- **qualità delle azioni suggerite**.

Per il primo giro, confrontiamo la stessa bolletta con `low`.
Poi, se serve, ripetiamo il modello vincente con `medium`.
""")

st.caption("I costi sono stime basate sui token restituiti dall'API e sui prezzi correnti dei modelli; il costo effettivo può variare.")
