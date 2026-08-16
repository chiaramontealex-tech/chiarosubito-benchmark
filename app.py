import json
import os
import re
import time

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="ChiaroSubito Benchmark", page_icon="🧪", layout="wide")

API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
BENCHMARK_PASSWORD = st.secrets.get("BENCHMARK_PASSWORD", os.getenv("BENCHMARK_PASSWORD"))

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

def check_password():
    if "benchmark_authenticated" not in st.session_state:
        st.session_state.benchmark_authenticated = False

    if st.session_state.benchmark_authenticated:
        return True

    st.title("🔐 ChiaroSubito Benchmark")
    st.write("Laboratorio privato per confrontare i modelli AI.")
    password = st.text_input("Password del laboratorio", type="password")

    if st.button("Accedi", type="primary"):
        if BENCHMARK_PASSWORD and password == BENCHMARK_PASSWORD:
            st.session_state.benchmark_authenticated = True
            st.rerun()
        else:
            st.error("Password non corretta.")

    return False

if not BENCHMARK_PASSWORD:
    st.error("Manca BENCHMARK_PASSWORD nei Secrets di Streamlit.")
    st.stop()

if not check_password():
    st.stop()

if not API_KEY:
    st.error("Manca OPENAI_API_KEY nei Secrets di Streamlit.")
    st.stop()

st.title("🧪 ChiaroSubito — Benchmark v0.3")
st.caption("Laboratorio privato: stesso PDF + stesso prompt + stesso schema, cambia solo il modello.")

uploaded = st.file_uploader("Carica il PDF da confrontare", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    effort = st.selectbox(
        "Reasoning effort",
        ["none", "low", "medium"],
        index=1,
        help="Per il primo test consigliamo low."
    )
with col2:
    runs = st.number_input("Ripetizioni per modello", min_value=1, max_value=3, value=1)

st.info("Primo test consigliato: stessa bolletta, reasoning = low, 1 ripetizione.")

if not uploaded:
    st.stop()

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
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
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    prices = {
        "gpt-5.6-sol": (5.0, 30.0, 0.5),
        "gpt-5.6-terra": (2.0, 12.0, 0.2),
        "gpt-5.6-luna": (0.2, 1.2, 0.02),
    }
    if model not in prices:
        return None
    pin, pout, pcached = prices[model]
    uncached = max(inp - cached, 0)
    usd = (uncached / 1_000_000) * pin + (cached / 1_000_000) * pcached + (out / 1_000_000) * pout
    return {"input_tokens": inp, "output_tokens": out, "cached_tokens": cached, "usd": usd}

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

try:
    document, pages = extract_pdf(uploaded)
except Exception as exc:
    st.error(f"Impossibile leggere il PDF: {exc}")
    st.stop()

st.write(f"**Documento:** {uploaded.name} · **Pagine:** {pages} · **Caratteri:** {len(document):,}")

if st.button("🚀 Avvia benchmark", type="primary"):
    client = OpenAI(api_key=API_KEY)
    results = []

    for label, model in MODELS:
        for run_no in range(int(runs)):
            with st.spinner(f"{label} — test {run_no + 1}/{runs}..."):
                try:
                    seconds, data, cost = run_model(client, model, document, effort)
                    results.append({
                        "label": label, "model": model, "run": run_no + 1,
                        "seconds": seconds, "data": data, "cost": cost, "error": None
                    })
                except Exception as exc:
                    results.append({
                        "label": label, "model": model, "run": run_no + 1,
                        "seconds": None, "data": None, "cost": None, "error": str(exc)
                    })

    st.session_state["results"] = results

results = st.session_state.get("results")
if not results:
    st.stop()

st.divider()
st.header("📊 Risultati")

table = []
for label, model in MODELS:
    rr = [x for x in results if x["model"] == model and x["seconds"] is not None]
    if not rr:
        table.append({"Modello": label, "Tempo medio": "ERRORE", "Confidenza": "—", "Costo": "—"})
        continue
    avg = sum(x["seconds"] for x in rr) / len(rr)
    conf = sum(x["data"]["confidence"] for x in rr) / len(rr)
    costs = [x["cost"]["usd"] for x in rr if x["cost"]]
    table.append({
        "Modello": label,
        "Tempo medio": f"{avg:.2f} s",
        "Confidenza": f"{conf * 100:.0f}%",
        "Costo": f"${sum(costs)/len(costs):.5f}" if costs else "n/d"
    })

st.table(table)

for label, model in MODELS:
    st.subheader(label)
    for x in [r for r in results if r["model"] == model]:
        if x["error"]:
            st.error(f"Test {x['run']}: {x['error']}")
            continue
        d = x["data"]
        cost = x["cost"]
        a, b, c = st.columns(3)
        a.metric("Tempo", f"{x['seconds']:.2f} s")
        b.metric("Confidenza", f"{d['confidence'] * 100:.0f}%")
        c.metric("Costo stimato", f"${cost['usd']:.5f}" if cost else "n/d")
        st.markdown(f"**Tipo:** {d['document_type']}")
        st.markdown(f"**Sintesi:** {d['summary']}")
        if cost:
            st.caption(
                f"Input: {cost['input_tokens']:,} token · "
                f"Output: {cost['output_tokens']:,} · "
                f"Cached: {cost['cached_tokens']:,}"
            )
        with st.expander("Dettaglio analisi"):
            st.json(d)

st.divider()
st.header("🧭 Come scegliamo")
st.markdown("""
Non scegliamo il modello solo perché è più veloce.

Valutiamo insieme:
- tempo;
- costo;
- accuratezza dei fatti;
- scadenze;
- anomalie;
- qualità delle azioni suggerite.

Per il primo giro: stessa bolletta, reasoning `low`, una ripetizione.
""")
