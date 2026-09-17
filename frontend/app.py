from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Neuromorphic-Ops", page_icon="N", layout="wide")
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")
st.markdown("""
<style>
:root { --ink:#102a43; --accent:#f97316; --paper:#f7f3ed; }
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at top right, #ffe3c2, var(--paper) 42%); color:var(--ink); }
[data-testid="stMetric"] { background:#fffaf3; border:1px solid #ead8c4; padding:1rem; border-radius:8px; }
</style>
""", unsafe_allow_html=True)
st.title("Neuromorphic-Ops")
st.caption("Autonomous edge model optimization control room")

uploaded = st.file_uploader("Upload a PyTorch model", type=["pt"])
strategy = st.selectbox("Optimization strategy", ["Automatic", "pruning", "quantization", "hybrid"])
if uploaded and st.button("Optimize model", type="primary"):
    import requests
    response = requests.post(f"{API_URL}/upload", files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    strategy_value = None if strategy == "Automatic" else strategy
    result = requests.post(f"{API_URL}/optimize", json={"model_path": payload["model_path"], "strategy": strategy_value}, timeout=600)
    result.raise_for_status()
    st.session_state["run"] = result.json()

run = st.session_state.get("run")
if run:
    state = run["state"]
    st.subheader("Optimization telemetry")
    cols = st.columns(5)
    for col, label, value in zip(cols, ["Original MB", "Optimized MB", "Accuracy", "Latency ms", "Memory MB"], [state.get("original_size_mb", 0), state.get("optimized_size_mb", 0), state.get("optimized_accuracy", 0), state.get("latency_ms", 0), state.get("memory_usage_mb", 0)]):
        col.metric(label, f"{value:.3f}" if isinstance(value, float) else value)
    st.write(f"Strategy: **{state.get('optimization_strategy', 'unknown')}** · Status: **{state.get('deployment_status', 'unknown')}**")
    if state.get("deployment_status") == "awaiting_approval":
        left, right = st.columns(2)
        if left.button("Approve deployment"):
            import requests
            approved = requests.post(f"{API_URL}/approve/{run['thread_id']}", json={"approved": True}, timeout=120)
            st.session_state["run"] = approved.json()
            st.rerun()
        if right.button("Reject"):
            import requests
            requests.post(f"{API_URL}/approve/{run['thread_id']}", json={"approved": False}, timeout=120)
            st.error("Deployment rejected")
    st.download_button("Download report JSON", json.dumps(state, indent=2), file_name="optimization_state.json")
