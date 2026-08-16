import json, os, re, time
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="ChiaroSubito Compression Benchmark", page_icon="🧪", layout="wide")

MODEL = "gpt-5.6-luna"
API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
BENCHMARK_PASSWORD = st.secrets.get("BENCHMARK_PASSWORD", os.getenv("BENCHMARK_PASSWORD"))

SYSTEM_PROMPT = r"""
Sei ChiaroSubito, un assistente specializzato nella comprensione di documenti per persone non esperte.
Obiettivo: CAPISCI → CONTROLLA → VERIFICA → AGISCI.
Usa esclusivamente informazioni presenti nel documento. Non inventare dati, scadenze, obblighi,
diritti, importi o conseguenze. Ogni informazione importante deve avere pagina ed evidenza.
Distingui FACT, CALCULATION, INTERPRETATION, UNKNOWN. Non presentare interpretazioni come fatti.
Se una risposta non è determinabile dal documento, dichiaralo. Non dare consulenza legale, fiscale
o finanziaria come certezza professionale. Non mostrare nella sintesi dati personali non necessari.
Cerca scadenze, importi, obblighi, diritti, clausole importanti, documenti richiamati, contraddizioni
e anomalie. Proponi solo azioni supportate dal documento. Scrivi in italiano semplice.
Usa i marcatori [PAGINA N] per le fonti. Restituisci esclusivamente JSON conforme allo schema.
"""

SCHEMA = {
 "type":"object","additionalProperties":False,
 "properties":{
  "document_type":{"type":"string"},"document_purpose":{"type":"string"},
  "confidence":{"type":"number","minimum":0,"maximum":1},"summary":{"type":"string"},
  "facts":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
   "label":{"type":"string"},"value":{"type":"string"},"page":{"type":"integer"},"evidence":{"type":"string"},
   "certainty":{"type":"string","enum":["FACT","CALCULATION","INTERPRETATION","UNKNOWN"]},
   "importance":{"type":"string","enum":["high","medium","low"]}},
   "required":["label","value","page","evidence","certainty","importance"]}},
  "deadlines":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
   "what":{"type":"string"},"date_or_term":{"type":"string"},"consequence":{"type":"string"},
   "page":{"type":"integer"},"evidence":{"type":"string"}},
   "required":["what","date_or_term","consequence","page","evidence"]}},
  "checks":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
   "title":{"type":"string"},"why":{"type":"string"},"page":{"type":"integer"},"evidence":{"type":"string"},
   "certainty":{"type":"string","enum":["FACT","CALCULATION","INTERPRETATION","UNKNOWN"]},
   "priority":{"type":"string","enum":["high","medium","low"]}},
   "required":["title","why","page","evidence","certainty","priority"]}},
  "calculations":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
   "description":{"type":"string"},"result":{"type":"string"},"page":{"type":"integer"},"evidence":{"type":"string"}},
   "required":["description","result","page","evidence"]}},
  "actions":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
   "action":{"type":"string"},"reason":{"type":"string"},"priority":{"type":"string","enum":["high","medium","low"]},
   "page":{"type":"integer"}},
   "required":["action","reason","priority","page"]}},
  "unknowns":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
   "question":{"type":"string"},"reason":{"type":"string"},"page":{"type":"integer"}},
   "required":["question","reason","page"]}}
 },
 "required":["document_type","document_purpose","confidence","summary","facts","deadlines","checks","calculations","actions","unknowns"]
}

def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()

def extract_pages(f):
    r = PdfReader(f)
    return [{"page": i, "text": p.extract_text() or ""} for i, p in enumerate(r.pages, 1)]

def full_text(pages):
    return "\n\n".join(f"[PAGINA {p['page']}]\n{clean(p['text'])}" for p in pages)

# EXACT compact_index() taken from the uploaded ChiaroSubito v0.6.
def compact_index(pages):
    rx = re.compile(
        r"\b(scadenza|entro|termine|recesso|pagamento|importo|totale|sanzion|interess|"
        r"obblig|diritto|contest|reclamo|modifica|condizion|canone|imposta|iva|contratto|"
        r"accertamento|adesione|notifica|risoluzione)\b", re.I
    )
    rows = []
    for p in pages:
        t = clean(p["text"])
        snippets = []
        for m in rx.finditer(t):
            snippets.append(t[max(0, m.start()-100):m.end()+180])
            if len(snippets) >= 3:
                break
        rows.append(f"[PAGINA {p['page']}] " + (" | ".join(snippets) if snippets else t[:350]))
    return "\n".join(rows)

def call_ai(text):
    client = OpenAI(api_key=API_KEY)
    t0 = time.perf_counter()
    r = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input="Analizza il seguente documento:\n\n" + text,
        reasoning={"effort": "low"},
        text={"format": {"type": "json_schema", "name": "chiarosubito_analysis",
                          "strict": True, "schema": SCHEMA}}
    )
    elapsed = time.perf_counter() - t0
    return elapsed, json.loads(r.output_text), getattr(r, "usage", None)

def usage_values(usage):
    if not usage:
        return None
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    return inp, out, cached

if not BENCHMARK_PASSWORD:
    st.error("Manca BENCHMARK_PASSWORD nei Secrets di Streamlit.")
    st.stop()
if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    st.title("🔐 ChiaroSubito Compression Benchmark")
    pw = st.text_input("Password del laboratorio", type="password")
    if st.button("Accedi", type="primary"):
        if pw == BENCHMARK_PASSWORD:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Password non corretta.")
    st.stop()

if not API_KEY:
    st.error("Manca OPENAI_API_KEY nei Secrets di Streamlit.")
    st.stop()

st.title("🧪 ChiaroSubito — Compression Benchmark v0.4")
st.caption("Confronto rigoroso: stesso modello, stesso prompt, stesso schema, cambia solo il testo inviato.")

uploaded = st.file_uploader("Carica il PDF da confrontare", type=["pdf"])
if uploaded is not None and st.session_state.get("last_uploaded_name") != uploaded.name:
    st.session_state.pop("results", None)
    st.session_state["last_uploaded_name"] = uploaded.name
st.info("Test consigliato: contratto Fineco · 1 prova · GPT-5.6 Luna · reasoning LOW.")

if not uploaded:
    st.stop()

try:
    pages = extract_pages(uploaded)
    text = full_text(pages)
    compact = compact_index(pages)
except Exception as e:
    st.error(f"Impossibile leggere il PDF: {e}")
    st.stop()

MAX_CHARS = 110000
st.write(f"**Documento:** {uploaded.name} · **Pagine:** {len(pages)}")
st.write(f"**Testo completo:** {len(text):,} caratteri")
st.write(f"**Scansione compatta:** {len(compact):,} caratteri")
if len(text) <= MAX_CHARS:
    st.warning("Questo PDF non supera la soglia v0.6 di 110.000 caratteri: la differenza potrebbe essere poco significativa.")

if st.button("🚀 Avvia confronto completo vs compatto", type="primary"):
    results = []
    client = OpenAI(api_key=API_KEY)

    for label, payload in [("DOCUMENTO COMPLETO", text), ("SCANSIONE COMPATTA", compact)]:
        with st.spinner(f"Analisi {label.lower()}..."):
            try:
                t0 = time.perf_counter()
                r = client.responses.create(
                    model=MODEL,
                    instructions=SYSTEM_PROMPT,
                    input="Analizza il seguente documento:\n\n" + payload,
                    reasoning={"effort": "low"},
                    text={"format": {"type": "json_schema", "name": "chiarosubito_analysis",
                                      "strict": True, "schema": SCHEMA}}
                )
                elapsed = time.perf_counter() - t0
                data = json.loads(r.output_text)
                u = usage_values(getattr(r, "usage", None))
                results.append({"label": label, "chars": len(payload), "seconds": elapsed,
                                 "data": data, "usage": u, "error": None})
            except Exception as e:
                results.append({"label": label, "chars": len(payload), "seconds": None,
                                 "data": None, "usage": None, "error": str(e)})

    st.session_state.results = results

results = st.session_state.get("results")
if not results:
    st.stop()

st.divider()
st.header("📊 Confronto")

for r in results:
    if r["error"]:
        st.error(f"{r['label']}: {r['error']}")
        continue
    u = r.get("usage")
    a, b, c = st.columns(3)
    a.metric("Tempo AI", f"{r['seconds']:.2f} s")
    b.metric("Caratteri inviati", f"{r['chars']:,}")
    c.metric("Confidenza", f"{r['data']['confidence']*100:.0f}%")
    st.subheader(r["label"])
    if u:
        st.caption(f"Input token: {u[0]:,} · Output token: {u[1]:,} · Cached: {u[2]:,}")
    st.markdown(f"**Tipo:** {r['data']['document_type']}")
    st.markdown(f"**Sintesi:** {r['data']['summary']}")
    with st.expander("Dettaglio analisi JSON"):
        st.json(r["data"])

st.divider()
st.header("🔍 Cosa dobbiamo verificare")
st.markdown("""
**1. Velocità:** il compatto deve essere sensibilmente più rapido.

**2. Completezza:** deve mantenere fatti, scadenze, controlli e azioni importanti.

**3. Affidabilità:** confrontiamo soprattutto il contratto Fineco, dove la v0.6 compatta aveva mostrato una perdita di qualità.

**4. Decisione:** se il compatto perde informazioni importanti, lo eliminiamo; se mantiene la qualità con un grande guadagno di tempo/costo, lo teniamo.
""")
