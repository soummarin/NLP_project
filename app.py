# ============================================================
# NLP Summarization App — Fine-tuned BART Model
# ============================================================
# HOW TO RUN:
#   streamlit run app.py
#
# REQUIRED PACKAGES:
#   pip install streamlit transformers torch pdfplumber fpdf2 PyPDF2
#
# Model files must be in the SAME folder as this script.
# ============================================================

import os
import re
import io
import json
import datetime
import traceback
import unicodedata

import torch
import streamlit as st

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="BART Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Global background ── */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 45%, #0d2137 100%);
    min-height: 100vh;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Hero ── */
  .hero {
    text-align: center;
    padding: 2.8rem 1rem 2rem;
    animation: fadeDown 0.8s ease;
  }
  .hero h1 {
    font-size: 2.9rem;
    font-weight: 700;
    color: #7b2ff7;
    margin-bottom: 0.5rem;
  }
  .hero p {
    color: #a0aec0;
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0 auto;
  }

  /* ── Section label ── */
  .section-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #7b2ff7;
    margin-bottom: 0.6rem;
  }

  /* ── Summary output box ── */
  .summary-box {
    background: rgba(0,210,255,0.07);
    border: 1px solid rgba(0,210,255,0.25);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.8;
    white-space: pre-wrap;
    animation: fadeIn 0.9s ease;
  }

  /* ── Placeholder text (waiting state) ── */
  .placeholder-text {
    color: #4a5568;
    font-style: italic;
    text-align: center;
    padding: 3rem 1rem;
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 14px;
  }

  /* ── Primary button ── */
  .stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.4rem;
    font-size: 1rem;
    font-weight: 600;
    transition: transform 0.18s, box-shadow 0.18s, opacity 0.18s;
    box-shadow: 0 4px 20px rgba(123,47,247,0.45);
    width: 100%;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(123,47,247,0.62);
    opacity: 0.92;
  }
  .stButton > button:active { transform: translateY(0); }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #11998e, #38ef7d) !important;
    color: #0f0c29 !important;
    border: none !important;
    border-radius: 10px;
    padding: 0.7rem 1.4rem;
    font-size: 1rem;
    font-weight: 700;
    width: 100%;
    transition: transform 0.18s, box-shadow 0.18s;
    box-shadow: 0 4px 20px rgba(56,239,125,0.38);
  }
  .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(56,239,125,0.55) !important;
  }

  /* ── Labels ── */
  label { color: #cbd5e0 !important; font-weight: 500 !important; }

  /* ── Text area — bright/light background ── */
  .stTextArea textarea {
    background: rgba(255,255,255,0.88) !important;
    border: 1px solid rgba(123,47,247,0.4) !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
  }
  .stTextArea textarea::placeholder { color: #6b7280 !important; }
  .stTextArea textarea:focus {
    border-color: rgba(0,210,255,0.6) !important;
    box-shadow: 0 0 0 2px rgba(0,210,255,0.15) !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.06) !important;
    border: 2px dashed rgba(123,47,247,0.5) !important;
    border-radius: 12px !important;
  }

  /* ── Radio ── */
  .stRadio label { color: #a0aec0 !important; }

  /* ── Divider ── */
  hr { border-color: rgba(255,255,255,0.09) !important; margin: 1.5rem 0 !important; }

  /* ── Keyframes ── */
  @keyframes fadeIn   { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
  @keyframes fadeDown { from { opacity:0; transform:translateY(-16px);} to { opacity:1; transform:translateY(0); } }
  @keyframes shine    { to   { background-position: 200% center; } }
</style>
""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════
# Model loading (cached)
# ══════════════════════════════════════════════════════════════════════
MODEL_REPO = "soumia11111/summarizer"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as fd:
            return json.load(fd)
    except Exception:
        return []


def save_history(history):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as fd:
            json.dump(history, fd, indent=2, ensure_ascii=False)
    except Exception:
        pass


def record_summary_history(entry):
    history = load_history()
    history.insert(0, entry)
    save_history(history)
    return history


def render_home(tokenizer, model, device):
    # ── Hero ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>✦ BART Summarizer</h1>
      <p>Upload a PDF or paste your text and get an AI-powered summary
         generated by a fine-tuned BART model.</p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # INPUT SECTION  — two side-by-side columns
    # ══════════════════════════════════════════════════════════════
    up_col, txt_col = st.columns([1, 1], gap="large")

    with up_col:
        st.markdown('<p class="section-label">📄 Upload a PDF</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="drop pdf here",
            type=["pdf"],
            label_visibility="collapsed",
        )

    with txt_col:
        st.markdown('<p class="section-label">✏️ Or paste / type text</p>', unsafe_allow_html=True)
        user_text = st.text_area(
            label="paste text",
            height=220,
            placeholder="Paste any article, report, or document text here …",
            label_visibility="collapsed",
        )

    input_source = "pdf"
    if uploaded_file and user_text.strip():
        input_source = st.radio(
            "Both inputs detected — which should be used?",
            options=["pdf", "text"],
            format_func=lambda x: "📄 Uploaded PDF" if x == "pdf" else "📝 Pasted Text",
            horizontal=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_btn = st.button("✨  Generate Summary", use_container_width=True)

    st.markdown("---")

    st.markdown('<p class="section-label">💡 Summary Output</p>', unsafe_allow_html=True)
    output_area = st.empty()
    download_area = st.empty()
    output_area.markdown(
        '<div class="placeholder-text">Your summary will appear here after you click Generate.</div>',
        unsafe_allow_html=True,
    )

    if run_btn:
        if not model_loaded:
            st.error("Model is not loaded. Fix the error above and restart the app.")
        else:
            raw_text = ""
            if uploaded_file and (input_source == "pdf" or not user_text.strip()):
                with st.spinner("Extracting text from PDF …"):
                    try:
                        raw_text = extract_pdf_text(uploaded_file.read())
                        if not raw_text.strip():
                            st.error("The PDF appears empty or is image-based (no selectable text).")
                    except RuntimeError as e:
                        st.error(str(e))
            elif user_text.strip():
                raw_text = user_text.strip()
            else:
                st.warning("Please upload a PDF or paste some text before clicking Generate.")

            if raw_text.strip():
                with st.spinner("Generating summary — this may take a moment …"):
                    try:
                        summary = summarize_text(raw_text, tokenizer, model, device)
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        filename = "summary_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf"
                        output_area.markdown(
                            f'<div class="summary-box">{summary}</div>'
                            f'<p style="color:#4a5568;font-size:0.78rem;text-align:right;margin-top:0.6rem;">'
                            f'Generated at {timestamp}</p>',
                            unsafe_allow_html=True,
                        )
                        record_summary_history({
                            "timestamp": timestamp,
                            "source": "pdf" if uploaded_file and (input_source == "pdf" or not user_text.strip()) else "text",
                            "file_name": uploaded_file.name if uploaded_file and (input_source == "pdf" or not user_text.strip()) else "-",
                            "input_length": len(raw_text),
                            "summary_length": len(summary),
                            "summary_preview": summary[:250] + ("..." if len(summary) > 250 else ""),
                        })
                        try:
                            pdf_bytes = build_pdf_report(raw_text, summary, timestamp)
                            _, dl_col, _ = st.columns([1, 2, 1])
                            with dl_col:
                                st.download_button(
                                    label="⬇️  Download Summary as PDF",
                                    data=pdf_bytes,
                                    file_name=filename,
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                        except Exception as pdf_err:
                            st.warning(f"Could not build PDF report: {pdf_err}\n"
                                       "Make sure fpdf2 is installed: pip install fpdf2")
                    except RuntimeError as oom_err:
                        if "out of memory" in str(oom_err).lower():
                            st.error("GPU out of memory. Try shorter text or unload other GPU processes.")
                        else:
                            st.error(f"Summarisation failed:\n\n{traceback.format_exc()}")
                    except Exception:
                        st.error(f"Summarisation failed:\n\n{traceback.format_exc()}")


@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import BartTokenizer, BartForConditionalGeneration
    tokenizer = BartTokenizer.from_pretrained(MODEL_REPO)
    model     = BartForConditionalGeneration.from_pretrained(MODEL_REPO)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


# ══════════════════════════════════════════════════════════════
# PDF text extraction
# ══════════════════════════════════════════════════════════════
def extract_pdf_text(pdf_bytes: bytes) -> str:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text
    except Exception:
        pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {e}")


# ══════════════════════════════════════════════════════════════
# Summarisation
# ══════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def summarize_text(text: str, tokenizer, model, device) -> str:
    MAX_CHUNK = 900
    OVERLAP   = 50
    MIN_LEN   = 56
    MAX_LEN   = 142

    text = clean_text(text)
    ids  = tokenizer.encode(text, add_special_tokens=False)
    step = MAX_CHUNK - OVERLAP

    chunks = []
    for start in range(0, len(ids), step):
        chunk_ids = ids[start: start + MAX_CHUNK]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if start + MAX_CHUNK >= len(ids):
            break

    summaries = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="longest",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=MAX_LEN,
                min_length=MIN_LEN,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return " ".join(summaries)


# ══════════════════════════════════════════════════════════════
# PDF report builder
# ══════════════════════════════════════════════════════════════
def to_latin(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if ord(c) < 256)


def build_pdf_report(original_text: str, summary: str, timestamp: str) -> bytes:
    from fpdf import FPDF

    PURPLE = (123,  47, 247)
    TEAL   = (  0, 180, 210)
    DARK   = ( 15,  12,  41)
    GRAY   = (100, 110, 130)
    WHITE  = (255, 255, 255)
    LIGHT  = (235, 237, 248)

    summary_safe  = to_latin(summary)
    original_safe = to_latin(original_text)
    MAX_ORIG = 2000
    excerpt = original_safe[:MAX_ORIG]
    if len(original_safe) > MAX_ORIG:
        excerpt += "\n\n[... text truncated for brevity ...]"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Header band
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 0, 210, 40, style="F")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(14, 8)
    pdf.cell(0, 12, "Document Summary")
    pdf.ln(13)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*TEAL)
    pdf.set_xy(14, 26)
    pdf.cell(0, 6, f"Generated by Fine-tuned BART  |  {timestamp}")

    # Divider
    pdf.set_draw_color(*PURPLE)
    pdf.set_line_width(0.8)
    pdf.line(14, 42, 196, 42)
    pdf.set_xy(14, 48)

    def section_title(label):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*PURPLE)
        pdf.set_fill_color(*LIGHT)
        pdf.set_x(14)
        pdf.cell(182, 8, f"  {label}", fill=True)
        pdf.ln(11)

    def body(content, color=DARK):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*color)
        pdf.set_x(14)
        pdf.multi_cell(182, 6, content)
        pdf.ln(4)

    section_title("Summary")
    body(summary_safe)

    pdf.ln(3)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.4)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(6)

    section_title("Original Text (excerpt)")
    body(excerpt, color=GRAY)

    pdf.set_y(-14)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 6, f"BART Summarizer  |  {timestamp}  |  Page {pdf.page_no()}", align="C")

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════
# Load model once
# ══════════════════════════════════════════════════════════════
with st.spinner("Loading fine-tuned BART model — please wait …"):
    try:
        tokenizer, model, device = load_model()
        model_loaded = True
    except Exception:
        st.error(f"Failed to load model:\n\n{traceback.format_exc()}")
        model_loaded = False




def render_history():
    st.markdown('<p class="section-label">🗂️ Summary History</p>', unsafe_allow_html=True)
    history = load_history()

    if not history:
        st.warning("No summaries have been generated yet. Once you summarize a PDF or paste text, the history will appear here.")
        return

    pdf_history = [entry for entry in history if entry.get("source") == "pdf"]
    if pdf_history:
        st.markdown("<h3 style='color:#e2e8f0;'>Previously Summarized PDFs</h3>", unsafe_allow_html=True)
        table_data = [
            {
                "Timestamp": entry["timestamp"],
                "PDF Name": entry.get("file_name", "-") or "-",
                "Input chars": entry["input_length"],
                "Summary chars": entry["summary_length"],
            }
            for entry in pdf_history
        ]
        st.table(table_data)

        for entry in pdf_history:
            label = entry.get("file_name", "Unknown PDF") or "Unknown PDF"
            with st.expander(f"{label} • {entry['timestamp']}"):
                st.markdown("**Summary preview:** " + entry.get("summary_preview", "-"))
                st.markdown("**Input length:** " + str(entry.get("input_length", 0)) + " characters")
                st.markdown("**Summary length:** " + str(entry.get("summary_length", 0)) + " characters")
    else:
        st.info("No PDF summaries in history yet. Generate a summary from a PDF to see it listed here.")

    text_history = [entry for entry in history if entry.get("source") == "text"]
    if text_history:
        st.markdown("<h3 style='color:#e2e8f0;margin-top:1.4rem;'>Other Past Summaries</h3>", unsafe_allow_html=True)
        for entry in text_history[:8]:
            with st.expander(f"Text summary • {entry['timestamp']}"):
                st.markdown("**Summary preview:** " + entry.get("summary_preview", "-"))
                st.markdown("**Input length:** " + str(entry.get("input_length", 0)) + " characters")
                st.markdown("**Summary length:** " + str(entry.get("summary_length", 0)) + " characters")


def render_statistics():
    st.markdown('<p class="section-label">📊 Summary Statistics</p>', unsafe_allow_html=True)
    history = load_history()

    if not history:
        st.warning("No summary data is available yet. Generate one or more summaries to populate statistics.")
        return

    total = len(history)
    pdf_count = sum(1 for entry in history if entry.get("source") == "pdf")
    text_count = sum(1 for entry in history if entry.get("source") == "text")
    avg_input = int(sum(entry["input_length"] for entry in history) / total)
    avg_summary = int(sum(entry["summary_length"] for entry in history) / total)
    avg_ratio = round(sum(entry["summary_length"] / max(entry["input_length"], 1) for entry in history) / total, 3)

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Total summaries", total)
    metrics_col2.metric("PDF summaries", pdf_count)
    metrics_col3.metric("Text summaries", text_count)

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Avg input chars", avg_input)
    stats_col2.metric("Avg summary chars", avg_summary)
    stats_col3.metric("Avg summary ratio", f"{avg_ratio}")

    st.markdown("---")
    st.markdown("**Summary source distribution**")
    st.bar_chart({"PDF": pdf_count, "Text": text_count})

    st.markdown("---")
    st.markdown("**Recent summaries**")
    for entry in history[:5]:
        label = entry.get("file_name") if entry.get("file_name") and entry.get("file_name") != "-" else f"Text summary"
        with st.expander(f'{label} • {entry["timestamp"]}'):
            st.write(f"Input length: {entry['input_length']} chars")
            st.write(f"Summary length: {entry['summary_length']} chars")
            st.write(f"Source: {entry['source'].upper()}")
            st.write(entry.get("summary_preview", "-"))


home_tab, history_tab, stats_tab = st.tabs(["Home", "History", "Statistics"])
with home_tab:
    render_home(tokenizer, model, device)
with history_tab:
    render_history()
with stats_tab:
    render_statistics()

st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#4a5568; font-size:0.8rem; padding-bottom:1.5rem;">
  Built with ❤️ using <strong style="color:#7b2ff7;">Streamlit</strong>
  &nbsp;·&nbsp; Fine-tuned <strong style="color:#00d2ff;">BART</strong>
  &nbsp;·&nbsp; HuggingFace Transformers
</p>
""", unsafe_allow_html=True)
