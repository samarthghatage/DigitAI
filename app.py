"""DigitAI: a two-workspace MNIST recognition lab."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import ARTIFACTS_DIR, MODEL_NAMES
from src.data import load_data
from src.history import add, clear, recent
from src.inference import predict
from src.validation import validate_upload


st.set_page_config(page_title="DigitAI — Handwritten Digit Recognition", page_icon="✦", layout="wide")
st.markdown("""
<style>
:root { --bg:#F5F7FB; --card:#FFFFFF; --ink:#172033; --muted:#5B667A; --line:#D9E1EC; --indigo:#4F46E5; --green:#15803D; --slate:#64748B; }
.stApp { background:var(--bg); color:var(--ink); }
.block-container { max-width:1220px; padding:2.2rem 2.4rem 4rem; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding:1.4rem 1rem; } [data-testid="stSidebar"] * { color:var(--ink); }
.brand { display:flex; align-items:center; gap:.7rem; margin:.2rem .4rem 2rem; }
.brand-mark { width:35px; height:35px; border-radius:11px; display:grid; place-items:center; background:#EEF0FF; color:var(--indigo); font-size:1.2rem; font-weight:800; }
.brand-name { font-size:1.15rem; font-weight:800; letter-spacing:-.02em; } .brand-subtitle { color:var(--muted); font-size:.72rem; margin-top:.1rem; }
.status-card { border:1px solid var(--line); border-radius:12px; padding:.9rem; margin:.8rem .25rem; background:#F8FAFD; }
.status-title { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; } .status-value { color:var(--ink); font-weight:750; margin-top:.25rem; } .status-detail { color:var(--muted); font-size:.78rem; margin-top:.15rem; }
.hero, .panel { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.35rem 1.45rem; box-shadow:0 8px 24px rgba(23,32,51,.045); }
.hero { border-top:4px solid var(--indigo); margin-bottom:1.1rem; } .hero h1 { color:var(--ink); font-size:2.15rem; letter-spacing:-.04em; margin:.1rem 0 .4rem; } .hero p { color:var(--muted); font-size:1rem; margin:0; }
.panel h2, .panel h3 { color:var(--ink); } .muted { color:var(--muted); } .metric-value { color:var(--ink); font-size:1.75rem; font-weight:800; } .metric-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
.digit-value { color:var(--indigo); font-size:5rem; font-weight:850; line-height:1; margin:.35rem 0; } .draw-card { border:1px solid var(--line); border-radius:16px; padding:1rem; background:var(--card); }
.st-key-canvas_frame { width:382px !important; max-width:100%; box-sizing:border-box; padding:8px !important; border:3px solid #334155 !important; border-radius:8px !important; background:#FFFFFF !important; box-shadow:0 3px 10px rgba(51,65,85,.16) !important; overflow:visible !important; }
div[data-testid="stSegmentedControl"] { margin:.5rem 0 1rem; } div[data-testid="stSegmentedControl"] button { min-height:2.7rem; font-weight:650; }
.stButton > button { border-radius:9px; min-height:2.65rem; font-weight:700; } .stButton > button[kind="primary"] { background:var(--green); border-color:var(--green); color:#fff; } .stButton > button:not([kind="primary"]) { background:var(--slate); border-color:var(--slate); color:#fff; }
div[data-testid="stSegmentedControl"] button { color:var(--ink) !important; background:#F8FAFD; border-color:var(--line); } div[data-testid="stSegmentedControl"] button[aria-pressed="true"] { color:#fff !important; background:var(--indigo) !important; }
[data-testid="stSidebar"] div[role="radiogroup"] label { border-left:3px solid transparent; border-radius:0 8px 8px 0; padding:.35rem .5rem; margin:.12rem 0; } [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { border-left-color:var(--indigo); background:#EEF0FF; font-weight:750; }
button[data-baseweb="tab"] { color:var(--ink) !important; font-weight:650; } button[data-baseweb="tab"][aria-selected="true"] { color:var(--indigo) !important; } div[data-baseweb="tab-highlight"] { background:var(--indigo) !important; } div[data-testid="stMetricValue"] { color:var(--ink); }
@media (max-width: 700px) { .block-container { padding:1.25rem 1rem 3rem; } .hero h1 { font-size:1.7rem; } .digit-value { font-size:4rem; } }
</style>
""", unsafe_allow_html=True)

def metric_rows():
    rows = {}
    for key in MODEL_NAMES:
        path = ARTIFACTS_DIR / f"{key}_evaluation.json"
        if path.exists(): rows[key] = json.loads(path.read_text(encoding="utf-8"))
    return rows

def history_table():
    st.markdown('<div class="panel">', unsafe_allow_html=True); st.subheader("Recent Predictions")
    rows = recent(20)
    if rows:
        frame = pd.DataFrame(rows, columns=["id", "timestamp", "model", "predicted_digit", "confidence", "inference_ms", "source"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S")
        frame["confidence"] = frame["confidence"].map(lambda value: f"{value:.1%}"); frame["inference_ms"] = frame["inference_ms"].map(lambda value: f"{value:.1f} ms")
        st.dataframe(frame[["timestamp", "predicted_digit", "model", "confidence", "source", "inference_ms"]].rename(columns={"timestamp":"Time", "predicted_digit":"Digit", "model":"Model", "confidence":"Confidence", "source":"Source", "inference_ms":"Inference time"}), use_container_width=True, hide_index=True)
    else: st.info("No predictions yet. Draw, upload, or select a sample and click Predict Digit.")
    if st.button("Clear Prediction History", key="clear_history"): st.session_state["confirm_clear"] = True
    if st.session_state.get("confirm_clear"):
        st.warning("This removes saved prediction history only."); yes, no = st.columns(2)
        if yes.button("Confirm clear", type="primary"): clear(); st.session_state.pop("confirm_clear", None); st.rerun()
        if no.button("Cancel"): st.session_state.pop("confirm_clear", None); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def mnist_samples():
    _, _, (x, y) = load_data(); ids = [int(np.flatnonzero(y == digit)[0]) for digit in range(10)]
    return [(x[i], Image.fromarray((x[i, ..., 0] * 255).astype("uint8"))) for i in ids]

def result_cards(result, selected):
    left, right = st.columns([.85, 1.65], gap="large")
    with left: st.markdown(f'<div class="panel"><div class="metric-label">Predicted digit</div><div class="digit-value">{result["digit"]}</div><div class="metric-value">{result["confidence"]:.2%}</div><div class="muted">Confidence</div><hr><div class="muted"><b>Model:</b> {selected}<br><b>Inference duration:</b> {result["inference_ms"]:.1f} ms</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><h3>Probability for Each Digit</h3>', unsafe_allow_html=True)
        digits = list(range(10)); colors = ["#4F46E5" if digit == result["digit"] else "#94A3B8" for digit in digits]
        fig = go.Figure(go.Bar(x=[value * 100 for value in result["probabilities"]], y=[str(digit) for digit in digits], orientation="h", marker_color=colors, hovertemplate="Digit %{y}: %{x:.2f}%<extra></extra>"))
        fig.update_layout(height=360, title=dict(text="Model confidence by digit", font=dict(color="#172033", size=16)), margin=dict(l=42, r=24, t=48, b=58), xaxis=dict(title="Predicted probability (%)", range=[0, 100], ticks="outside", ticksuffix="%", tickfont=dict(color="#334155"), title_font=dict(color="#172033"), showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#334155"), yaxis=dict(title="Digit", autorange="reversed", ticks="outside", tickfont=dict(color="#334155"), title_font=dict(color="#172033"), showgrid=False, linecolor="#334155"), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", showlegend=False, font=dict(color="#172033", family="Arial"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}); st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><h3>Processed Model Input</h3><p class="muted">This is the exact normalized 28 × 28 grayscale tensor sent to the selected model.</p>', unsafe_allow_html=True)
    st.image((result["tensor"][0, ..., 0] * 255).astype("uint8"), width=180, caption="Normalized model input"); st.markdown('</div>', unsafe_allow_html=True)

def recognize():
    st.markdown('<div class="hero"><h1>Handwritten Digit Recognition</h1><p>Draw, upload, or select a sample to recognize a digit from 0 to 9.</p></div>', unsafe_allow_html=True)
    metrics = metric_rows(); options = list(MODEL_NAMES.items()); selected = st.selectbox("Choose prediction model", [label for _, label in options]); key = next(model_key for model_key, label in options if label == selected)
    if key in metrics: st.caption(f"{selected} · {metrics[key]['accuracy']:.2%} actual test accuracy. This model converts your input into a normalized 28 × 28 tensor before prediction.")
    else: st.caption(f"{selected} · Test accuracy is not available yet. This model converts your input into a normalized 28 × 28 tensor before prediction.")
    input_method = st.segmented_control("Input method", ["Draw a Digit", "Upload an Image", "Try a Sample"], default="Draw a Digit", key="input_method"); image, tensor, source = None, None, None
    if input_method == "Draw a Digit":
        st.markdown('<div class="draw-card"><h3>Drawing Pad</h3><p class="muted">Use dark ink on the light canvas. Draw one digit from 0 to 9, then predict.</p>', unsafe_allow_html=True)
        try:
            from streamlit_drawable_canvas import st_canvas
            with st.container(border=True, width=382, key="canvas_frame"):
                canvas = st_canvas(fill_color="rgba(241,245,249,0)", stroke_color="#334155", background_color="#F1F5F9", stroke_width=14, height=360, width=360, drawing_mode="freedraw", return_image_data=True, key=f"canvas_{st.session_state.get('canvas_version', 0)}")
            if canvas is not None and canvas.image_data is not None: image, source = canvas.image_data, "canvas"
        except Exception as exc: st.warning(f"Canvas unavailable: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)
    elif input_method == "Upload an Image":
        st.markdown('<div class="panel"><h3>Upload an Image</h3><p class="muted">Choose a PNG, JPG, or JPEG containing one handwritten digit.</p>', unsafe_allow_html=True); uploaded = st.file_uploader("Image file", type=["png", "jpg", "jpeg"], key=f"upload_{st.session_state.get('upload_version', 0)}")
        if uploaded:
            validation = validate_upload(uploaded)
            if validation.valid:
                uploaded.seek(0); image = Image.open(uploaded).convert("L"); source = "upload"; st.image(image, width=180, caption="Uploaded input")
            else:
                st.session_state.pop("result", None)
                st.error(validation.message)
                if st.button("Remove invalid upload", key="remove_invalid_upload"):
                    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1
                    st.session_state.pop("result", None)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel"><h3>Try a Sample</h3><p class="muted">Use a genuine MNIST test sample to explore the model.</p>', unsafe_allow_html=True); samples = mnist_samples(); selected_digit = st.selectbox("Sample class for the demo", list(range(10))); tensor, image = samples[selected_digit]; source = "sample"; st.image(image, width=150, caption="MNIST sample"); st.markdown('</div>', unsafe_allow_html=True)
    predict_col, clear_col = st.columns(2)
    with predict_col:
        if st.button("Predict Digit", type="primary", use_container_width=True, disabled=image is None):
            try:
                result = predict(selected, tensor=tensor) if source == "sample" else predict(selected, image=image); add(selected, result["digit"], result["confidence"], result["inference_ms"], source); st.session_state["result"] = result
            except (ValueError, FileNotFoundError) as exc: st.error(str(exc))
    with clear_col:
        if st.button("Clear Canvas", use_container_width=True): st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1; st.session_state.pop("result", None); st.rerun()
    if st.session_state.get("result"): result_cards(st.session_state["result"], selected)
    history_table()

def lab():
    st.markdown('<div class="hero"><h1>Model Lab</h1><p>Compare the performance of the two trained convolutional models.</p></div>', unsafe_allow_html=True); rows = metric_rows()
    if not rows: st.warning("Train and evaluate the models first: python -m src.train --model all --epochs 5"); return
    cards = st.columns(2, gap="large")
    for col, key in zip(cards, rows):
        data = rows[key]; macro = data["classification_report"]["macro avg"]
        with col:
            st.markdown(f'<div class="panel"><div class="metric-label">{data["model"]}</div><div class="metric-value">{data["accuracy"]:.2%}</div><div class="muted">Actual test accuracy</div></div>', unsafe_allow_html=True); m1, m2, m3 = st.columns(3); m1.metric("Precision", f'{macro["precision"]:.2%}'); m2.metric("Recall", f'{macro["recall"]:.2%}'); m3.metric("F1-score", f'{macro["f1-score"]:.2%}')
    chart_rows = []
    for data in rows.values():
        macro = data["classification_report"]["macro avg"]; chart_rows.extend([{"Model": data["model"], "Metric": metric, "Score": value} for metric, value in [("Accuracy", data["accuracy"]), ("Precision", macro["precision"]), ("Recall", macro["recall"]), ("F1-score", macro["f1-score"])]] )
    fig = go.Figure(); palette = ["#4F46E5", "#15803D"]
    for index, model in enumerate([data["model"] for data in rows.values()]):
        values = [item["Score"] for item in chart_rows if item["Model"] == model]; fig.add_trace(go.Bar(name=model, x=["Accuracy", "Precision", "Recall", "F1-score"], y=values, marker_color=palette[index % len(palette)], text=[f"{value:.1%}" for value in values], textposition="auto"))
    fig.update_layout(barmode="group", height=360, title=dict(text="Model quality comparison", font=dict(color="#172033", size=16)), yaxis=dict(title="Measured score", range=[.9, 1], tickformat=".0%", ticks="outside", tickfont=dict(color="#334155"), title_font=dict(color="#172033"), showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#334155"), xaxis=dict(title="Evaluation metric", ticks="outside", tickfont=dict(color="#334155"), title_font=dict(color="#172033"), showgrid=False, linecolor="#334155"), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=58, r=24, t=52, b=62), legend=dict(title=dict(text="Model"), font=dict(color="#172033"), bgcolor="#FFFFFF", bordercolor="#CBD5E1", borderwidth=1), font=dict(color="#172033", family="Arial")); st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}); st.caption("Accessible metric summary: accuracy is overall correctness; precision, recall, and F1-score are macro averages across all ten digit classes.")
    st.subheader("Confusion Matrices"); matrix_cols = st.columns(len(rows), gap="large")
    for col, (key, data) in zip(matrix_cols, rows.items()):
        matrix = np.asarray(data["confusion_matrix"]); heat = go.Figure(go.Heatmap(z=matrix, x=list(range(10)), y=list(range(10)), colorscale="Blues", text=matrix, texttemplate="%{text}", hovertemplate="Actual %{y} → Predicted %{x}: %{z}<extra></extra>", colorbar=dict(title="Count", tickfont=dict(color="#334155"), title_font=dict(color="#172033")))); heat.update_layout(height=500, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", xaxis=dict(title="Predicted digit", ticks="outside", tickfont=dict(color="#334155"), title_font=dict(color="#172033"), showgrid=False, linecolor="#334155"), yaxis=dict(title="Actual digit", ticks="outside", tickfont=dict(color="#334155"), title_font=dict(color="#172033"), showgrid=False, linecolor="#334155"), margin=dict(l=58, r=24, t=58, b=58), title=dict(text=data["model"], font=dict(color="#172033", size=16)), font=dict(color="#172033", family="Arial"))
        with col: st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})
    history_table()

lab_metrics = metric_rows(); status = "Models ready"; status_detail = ""
if lab_metrics:
    best_key = max(lab_metrics, key=lambda item: lab_metrics[item]["accuracy"]); status_detail = f"{MODEL_NAMES[best_key]} · {lab_metrics[best_key]['accuracy']:.2%} test accuracy"
else: status = "Awaiting evaluation"; status_detail = "Run evaluation to populate model status"
st.sidebar.markdown('<div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">DigitAI</div><div class="brand-subtitle">Handwritten Digit Recognition</div></div></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Workspace", ["Recognize Digit", "Model Lab"], label_visibility="collapsed")
st.sidebar.markdown(f'<div class="status-card"><div class="status-title">Model status</div><div class="status-value">{status}</div><div class="status-detail">{status_detail}</div></div>', unsafe_allow_html=True); st.sidebar.caption("Fresh artifacts · seed 42 · CPU-ready")
recognize() if page == "Recognize Digit" else lab()
