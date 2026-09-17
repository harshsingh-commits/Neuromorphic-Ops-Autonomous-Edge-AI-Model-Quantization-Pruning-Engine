
from __future__ import annotations

import html
import os
from typing import Any

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

MAX_FRONTEND_UPLOAD_SIZE = 100 * 1024 * 1024


st.set_page_config(
    page_title="Neuromorphic-Ops",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 90% 0%,
                rgba(255, 95, 55, 0.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 5% 30%,
                rgba(255, 150, 70, 0.06),
                transparent 25%
            ),
            #07090d;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1rem;
        padding-bottom: 4rem;
    }

    /* ======================================================
       STREAMLIT TEXT
       ====================================================== */

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div {
        line-height: 1.5;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        min-height: 44px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.08),
                rgba(255,255,255,0.035)
            ) !important;
        color: #f4f6f9 !important;
        font-weight: 750 !important;
        transition: 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(255,112,70,0.5) !important;
        box-shadow:
            0 12px 32px rgba(255,83,55,0.14);
    }

    .stButton > button[kind="primary"] {
        border: none !important;
        background:
            linear-gradient(
                135deg,
                #ff7043,
                #ff4f58
            ) !important;
        box-shadow:
            0 12px 35px rgba(255,83,55,0.18);
    }

    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] section {
        border-radius: 18px !important;
        border: 1px dashed rgba(255,255,255,0.15) !important;
        background:
            rgba(255,255,255,0.025) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #a3abba !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: 20px !important;
    }

    /* ======================================================
       SELECTBOX
       ====================================================== */

    div[data-baseweb="select"] > div {
        background:
            rgba(255,255,255,0.035) !important;
        border-color:
            rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }

    /* ======================================================
       SLIDER
       ====================================================== */

    [data-testid="stSlider"] {
        padding-top: 4px;
    }

    /* ======================================================
       EXPANDER
       ====================================================== */

    [data-testid="stExpander"] {
        border-radius: 17px !important;
        border: 1px solid rgba(255,255,255,0.065) !important;
        background:
            rgba(255,255,255,0.022) !important;
    }

    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(
    content: str,
) -> None:
    """
    Render custom HTML safely through Streamlit's HTML renderer.

    This avoids showing raw HTML markup as visible text.
    """

    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(
            content,
            unsafe_allow_html=True,
        )


# ============================================================
# API
# ============================================================

def api_request(
    method: str,
    endpoint: str,
    **kwargs: Any,
) -> requests.Response:
    """Call the FastAPI backend."""

    return requests.request(
        method,
        f"{API_URL}{endpoint}",
        timeout=120,
        **kwargs,
    )


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value: Any,
) -> float | None:
    """Convert numeric values to float."""

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    return None


def escaped(
    value: Any,
) -> str:
    """Escape dynamic values before inserting into HTML."""

    return html.escape(
        str(value)
    )


def status_value(
    value: Any,
) -> str:
    """Format workflow state for display."""

    if value is None:
        return "Unknown"

    return str(value).replace(
        "_",
        " ",
    ).title()


# ============================================================
# SECTION HEADER
# ============================================================

def section_header(
    kicker: str,
    title: str,
    subtitle: str = "",
) -> None:

    render_html(
        f"""
        <div style="
            margin-top:30px;
            margin-bottom:14px;
        ">

            <div style="
                color:#ff8d5c;
                font-size:10px;
                font-weight:800;
                letter-spacing:.18em;
                text-transform:uppercase;
                margin-bottom:5px;
            ">
                {escaped(kicker)}
            </div>

            <div style="
                color:#f6f7fa;
                font-size:28px;
                font-weight:850;
                letter-spacing:-.045em;
                margin:0;
            ">
                {escaped(title)}
            </div>

            {
                f'''
                <div style="
                    color:#777f90;
                    font-size:12px;
                    margin-top:5px;
                ">
                    {escaped(subtitle)}
                </div>
                '''
                if subtitle
                else ""
            }

        </div>
        """
    )


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(
    label: str,
    value: str,
    meta: str = "",
    meta_class: str = "",
) -> None:

    meta_color = (
        "#71e6a0"
        if meta_class == "positive"
        else (
            "#ffb66c"
            if meta_class == "warning"
            else "#8891a0"
        )
    )

    render_html(
        f"""
        <div style="
            position:relative;
            overflow:hidden;
            min-height:142px;
            padding:21px;
            border-radius:20px;
            border:1px solid rgba(255,255,255,.065);
            background:
                linear-gradient(
                    145deg,
                    rgba(20,23,30,.98),
                    rgba(11,13,18,.98)
                );
        ">

            <div style="
                position:absolute;
                right:-45px;
                top:-45px;
                width:130px;
                height:130px;
                border-radius:50%;
                background:
                    radial-gradient(
                        circle,
                        rgba(255,122,72,.14),
                        transparent 68%
                    );
            "></div>

            <div style="
                color:#7f8797;
                font-size:10px;
                font-weight:800;
                letter-spacing:.09em;
                text-transform:uppercase;
            ">
                {escaped(label)}
            </div>

            <div style="
                color:#ffffff;
                font-size:29px;
                font-weight:850;
                letter-spacing:-.04em;
                margin-top:10px;
            ">
                {escaped(value)}
            </div>

            <div style="
                color:{meta_color};
                font-size:11px;
                margin-top:9px;
            ">
                {escaped(meta)}
            </div>

        </div>
        """
    )


# ============================================================
# METRICS
# ============================================================

def show_metrics(
    state: dict[str, Any],
) -> None:

    original_size = safe_float(
        state.get("original_size_mb")
    )

    optimized_size = safe_float(
        state.get("optimized_size_mb")
    )

    original_accuracy = safe_float(
        state.get("original_accuracy")
    )

    optimized_accuracy = safe_float(
        state.get("optimized_accuracy")
    )

    baseline_latency = safe_float(
        state.get("baseline_latency_ms")
    )

    latency = safe_float(
        state.get("latency_ms")
    )

    baseline_memory = safe_float(
        state.get("baseline_memory_usage_mb")
    )

    memory = safe_float(
        state.get("memory_usage_mb")
    )

    accuracy_drop = safe_float(
        state.get("accuracy_drop")
    )

    compression = safe_float(
        state.get("compression_ratio")
    )

    latency_improvement = safe_float(
        state.get(
            "latency_improvement_percent"
        )
    )

    memory_reduction = safe_float(
        state.get(
            "memory_reduction_percent"
        )
    )

    cols = st.columns(
        4,
        gap="medium",
    )

    with cols[0]:

        metric_card(
            "Original Size",
            (
                f"{original_size:.2f} MB"
                if original_size is not None
                else "—"
            ),
            "Baseline model footprint",
        )

    with cols[1]:

        smaller = (
            (
                (original_size - optimized_size)
                / original_size
            )
            * 100
            if (
                original_size is not None
                and optimized_size is not None
                and original_size > 0
            )
            else None
        )

        metric_card(
            "Optimized Size",
            (
                f"{optimized_size:.2f} MB"
                if optimized_size is not None
                else "—"
            ),
            (
                f"{smaller:.1f}% smaller"
                if smaller is not None
                else "Optimized artifact"
            ),
            "positive",
        )

    with cols[2]:

        latency_meta = (
            (
                f"{latency_improvement:.2f}% improvement"
                if latency_improvement is not None
                else (
                    f"Baseline {baseline_latency:.2f} ms"
                    if baseline_latency is not None
                    else "CPU inference"
                )
            )
            if latency is not None
            else "CPU inference"
        )

        metric_card(
            "Latency",
            (
                f"{latency:.2f} ms"
                if latency is not None
                else "—"
            ),
            latency_meta,
            (
                "positive"
                if (
                    latency_improvement is not None
                    and latency_improvement > 0
                )
                else ""
            ),
        )

    with cols[3]:

        memory_meta = (
            (
                f"{memory_reduction:.2f}% reduction"
                if memory_reduction is not None
                else (
                    f"Baseline {baseline_memory:.2f} MB"
                    if baseline_memory is not None
                    else "Runtime footprint"
                )
            )
            if memory is not None
            else "Runtime footprint"
        )

        metric_card(
            "Memory",
            (
                f"{memory:.2f} MB"
                if memory is not None
                else "—"
            ),
            memory_meta,
            (
                "positive"
                if (
                    memory_reduction is not None
                    and memory_reduction > 0
                )
                else ""
            ),
        )

    st.markdown(
        "<div style='height:12px'></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(
        4,
        gap="medium",
    )

    with cols[0]:

        metric_card(
            "Baseline Accuracy",
            (
                f"{original_accuracy:.2f}%"
                if original_accuracy is not None
                else "—"
            ),
            "Original model",
        )

    with cols[1]:

        metric_card(
            "Optimized Accuracy",
            (
                f"{optimized_accuracy:.2f}%"
                if optimized_accuracy is not None
                else "—"
            ),
            "Post-optimization",
        )

    with cols[2]:

        threshold = safe_float(
            state.get(
                "accuracy_threshold"
            )
        )

        within_threshold = (
            accuracy_drop is not None
            and threshold is not None
            and accuracy_drop <= threshold
        )

        metric_card(
            "Accuracy Drop",
            (
                f"{accuracy_drop:.2f} pp"
                if accuracy_drop is not None
                else "—"
            ),
            (
                "Within selected threshold"
                if within_threshold
                else "Evaluation result"
            ),
            (
                "positive"
                if within_threshold
                else (
                    "warning"
                    if accuracy_drop is not None
                    else ""
                )
            ),
        )

    with cols[3]:

        metric_card(
            "Compression Ratio",
            (
                f"{compression:.2f}×"
                if compression is not None
                else "—"
            ),
            "Original / optimized size",
            "positive",
        )


# ============================================================
# WORKFLOW RAIL
# ============================================================

def show_workflow_rail(
    workflow_status: str,
) -> None:

    normalized = workflow_status.lower()

    stages = [
        (
            "01",
            "Upload",
            normalized in {
                "uploaded",
                "analysis_complete",
                "strategy_selected",
                "optimizing",
                "report_generated",
                "waiting_for_approval",
                "deployment_ready",
                "completed",
                "workflow_completed",
            },
        ),
        (
            "02",
            "Analyze",
            normalized in {
                "analysis_complete",
                "strategy_selected",
                "optimizing",
                "report_generated",
                "waiting_for_approval",
                "deployment_ready",
                "completed",
                "workflow_completed",
            },
        ),
        (
            "03",
            "Optimize",
            normalized in {
                "optimizing",
                "report_generated",
                "waiting_for_approval",
                "deployment_ready",
                "completed",
                "workflow_completed",
            },
        ),
        (
            "04",
            "Evaluate",
            normalized in {
                "report_generated",
                "waiting_for_approval",
                "deployment_ready",
                "completed",
                "workflow_completed",
            },
        ),
        (
            "05",
            "Approval",
            normalized in {
                "waiting_for_approval",
                "deployment_ready",
                "completed",
                "workflow_completed",
            },
        ),
        (
            "06",
            "Deploy",
            normalized in {
                "deployment_ready",
                "completed",
                "workflow_completed",
            },
        ),
    ]

    blocks: list[str] = []

    for index, name, completed in stages:

        active = (
            (
                name == "Approval"
                and normalized
                == "waiting_for_approval"
            )
            or (
                name == "Optimize"
                and normalized
                == "optimizing"
            )
            or (
                name == "Deploy"
                and normalized
                == "deployment_ready"
            )
        )

        background = (
            "linear-gradient("
            "135deg,"
            "rgba(255,114,68,.12),"
            "rgba(255,255,255,.025)"
            ")"
            if active
            else (
                "rgba(85,213,136,.045)"
                if completed
                else "rgba(255,255,255,.025)"
            )
        )

        border = (
            "rgba(255,126,77,.30)"
            if active
            else (
                "rgba(85,213,136,.15)"
                if completed
                else "rgba(255,255,255,.06)"
            )
        )

        blocks.append(
            f"""
            <div style="
                flex:1;
                min-height:72px;
                padding:13px;
                border-radius:15px;
                border:1px solid {border};
                background:{background};
            ">

                <div style="
                    color:#6d7585;
                    font-size:9px;
                    font-weight:800;
                    letter-spacing:.1em;
                ">
                    {index}
                </div>

                <div style="
                    color:#f4f5f8;
                    font-size:11px;
                    font-weight:800;
                    margin-top:6px;
                ">
                    {name}
                </div>

            </div>
            """
        )

    render_html(
        f"""
        <div style="
            display:flex;
            gap:8px;
            width:100%;
            margin:8px 0 8px 0;
        ">
            {''.join(blocks)}
        </div>
        """
    )


# ============================================================
# CHARTS
# ============================================================

def show_optimization_chart(
    state: dict[str, Any],
) -> None:

    import plotly.graph_objects as go

    original_size = safe_float(
        state.get("original_size_mb")
    )

    optimized_size = safe_float(
        state.get("optimized_size_mb")
    )

    baseline_latency = safe_float(
        state.get("baseline_latency_ms")
    )

    latency = safe_float(
        state.get("latency_ms")
    )

    chart_cols = st.columns(
        2,
        gap="medium",
    )

    with chart_cols[0]:

        if (
            original_size is not None
            and optimized_size is not None
        ):

            figure = go.Figure()

            figure.add_bar(
                x=[
                    "Original",
                    "Optimized",
                ],
                y=[
                    original_size,
                    optimized_size,
                ],
                marker=dict(
                    color=[
                        "#687080",
                        "#ff7043",
                    ],
                    line=dict(
                        width=0
                    ),
                ),
                text=[
                    f"{original_size:.3f} MB",
                    f"{optimized_size:.3f} MB",
                ],
                textposition="auto",
            )

            figure.update_layout(
                title=dict(
                    text="Model Footprint",
                    font=dict(
                        color="#f4f6f9",
                        size=17,
                    ),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color="#8d95a5"
                ),
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
                xaxis=dict(
                    showgrid=False,
                ),
                yaxis=dict(
                    title="MB",
                    gridcolor="rgba(255,255,255,.06)",
                    zeroline=False,
                ),
                showlegend=False,
                bargap=0.35,
            )

            st.plotly_chart(
                figure,
                width="stretch",
                config={
                    "displayModeBar": False,
                },
            )

    with chart_cols[1]:

        if (
            baseline_latency is not None
            and latency is not None
        ):

            figure = go.Figure()

            figure.add_bar(
                x=[
                    "Baseline",
                    "Optimized",
                ],
                y=[
                    baseline_latency,
                    latency,
                ],
                marker=dict(
                    color=[
                        "#687080",
                        "#ff7043",
                    ],
                    line=dict(
                        width=0
                    ),
                ),
                text=[
                    f"{baseline_latency:.3f} ms",
                    f"{latency:.3f} ms",
                ],
                textposition="auto",
            )

            figure.update_layout(
                title=dict(
                    text="Inference Latency",
                    font=dict(
                        color="#f4f6f9",
                        size=17,
                    ),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color="#8d95a5"
                ),
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
                xaxis=dict(
                    showgrid=False,
                ),
                yaxis=dict(
                    title="Milliseconds",
                    gridcolor="rgba(255,255,255,.06)",
                    zeroline=False,
                ),
                showlegend=False,
                bargap=0.35,
            )

            st.plotly_chart(
                figure,
                width="stretch",
                config={
                    "displayModeBar": False,
                },
            )


# ============================================================
# TOP NAV
# ============================================================

render_html(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding:13px 4px 19px 4px;
    ">

        <div style="
            display:flex;
            align-items:center;
            gap:12px;
        ">

            <div style="
                width:42px;
                height:42px;
                border-radius:13px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:23px;
                background:
                    linear-gradient(
                        135deg,
                        #ff8b46,
                        #ff4f5e
                    );
                box-shadow:
                    0 10px 35px rgba(255,92,59,.25);
            ">
                ⚡
            </div>

            <div>

                <div style="
                    color:#f7f8fb;
                    font-size:18px;
                    font-weight:800;
                    letter-spacing:-.03em;
                ">
                    Neuromorphic-Ops
                </div>

                <div style="
                    color:#7f8797;
                    font-size:10px;
                    margin-top:2px;
                    letter-spacing:.08em;
                ">
                    AUTONOMOUS EDGE-AI OPTIMIZATION
                </div>

            </div>

        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            padding:8px 13px;
            border-radius:999px;
            font-size:11px;
            font-weight:700;
            color:#a6f7c5;
            border:1px solid rgba(83,210,133,.18);
            background:rgba(83,210,133,.08);
        ">

            <span style="
                width:7px;
                height:7px;
                border-radius:50%;
                background:#61e28d;
                box-shadow:
                    0 0 12px rgba(97,226,141,.9);
            "></span>

            SYSTEM ONLINE

        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div style="
        position:relative;
        overflow:hidden;
        min-height:275px;
        padding:40px 42px;
        border-radius:26px;
        border:1px solid rgba(255,255,255,.075);
        background:
            linear-gradient(
                115deg,
                rgba(20,22,29,.98),
                rgba(11,13,18,.96)
            );
        box-shadow:
            0 30px 100px rgba(0,0,0,.28);
    ">

        <div style="
            position:absolute;
            width:420px;
            height:420px;
            right:-110px;
            top:-170px;
            border-radius:50%;
            background:
                radial-gradient(
                    circle,
                    rgba(255,101,56,.30),
                    rgba(255,101,56,.03) 62%,
                    transparent 70%
                );
        "></div>

        <div style="
            color:#ff9a64;
            font-size:11px;
            font-weight:800;
            letter-spacing:.16em;
            text-transform:uppercase;
            margin-bottom:12px;
        ">
            Autonomous Model Intelligence
        </div>

        <div style="
            max-width:850px;
            color:#ffffff;
            font-size:clamp(38px,5vw,66px);
            line-height:.98;
            font-weight:900;
            letter-spacing:-.055em;
        ">
            Make your AI model
            <span style="
                background:
                    linear-gradient(
                        90deg,
                        #ffffff 10%,
                        #ff9a64 55%,
                        #ff6258 100%
                    );
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
            ">
                smaller, faster & edge-ready.
            </span>
        </div>

        <div style="
            max-width:760px;
            margin-top:18px;
            color:#9ca3b1;
            font-size:14px;
            line-height:1.7;
        ">
            Analyze, optimize, evaluate and deploy your PyTorch
            models through an autonomous LangGraph workflow with
            pruning, quantization and ONNX Runtime support.
        </div>

        <div style="
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top:22px;
        ">

            <span style="
                padding:7px 11px;
                color:#cfd4dd;
                background:rgba(255,255,255,.045);
                border:1px solid rgba(255,255,255,.065);
                border-radius:999px;
                font-size:10px;
                font-weight:700;
            ">PyTorch</span>

            <span style="
                padding:7px 11px;
                color:#cfd4dd;
                background:rgba(255,255,255,.045);
                border:1px solid rgba(255,255,255,.065);
                border-radius:999px;
                font-size:10px;
                font-weight:700;
            ">LangGraph</span>

            <span style="
                padding:7px 11px;
                color:#cfd4dd;
                background:rgba(255,255,255,.045);
                border:1px solid rgba(255,255,255,.065);
                border-radius:999px;
                font-size:10px;
                font-weight:700;
            ">Pruning</span>

            <span style="
                padding:7px 11px;
                color:#cfd4dd;
                background:rgba(255,255,255,.045);
                border:1px solid rgba(255,255,255,.065);
                border-radius:999px;
                font-size:10px;
                font-weight:700;
            ">Quantization</span>

            <span style="
                padding:7px 11px;
                color:#cfd4dd;
                background:rgba(255,255,255,.045);
                border:1px solid rgba(255,255,255,.065);
                border-radius:999px;
                font-size:10px;
                font-weight:700;
            ">ONNX Runtime</span>

            <span style="
                padding:7px 11px;
                color:#cfd4dd;
                background:rgba(255,255,255,.045);
                border:1px solid rgba(255,255,255,.065);
                border-radius:999px;
                font-size:10px;
                font-weight:700;
            ">Human Approval</span>

        </div>

    </div>
    """
)


# ============================================================
# BACKEND HEALTH
# ============================================================

try:

    health_response = api_request(
        "GET",
        "/health",
    )

    if not health_response.ok:

        st.error(
            "FastAPI backend is unavailable."
        )

        st.stop()

except requests.RequestException as exc:

    st.error(
        f"Cannot connect to FastAPI backend: {exc}"
    )

    st.code(
        "docker compose up -d",
        language="powershell",
    )

    st.stop()


# ============================================================
# MODEL UPLOAD
# ============================================================

section_header(
    "01 · Model Input",
    "Upload your model",
    "Start with a trusted PyTorch .pt or .pth artifact.",
)

render_html(
    """
    <div style="
        padding:24px;
        border-radius:22px;
        border:1px solid rgba(255,255,255,.07);
        background:
            linear-gradient(
                135deg,
                rgba(22,25,33,.98),
                rgba(13,15,20,.98)
            );
    ">

        <div style="
            display:flex;
            align-items:center;
            justify-content:space-between;
            margin-bottom:17px;
        ">

            <div>

                <div style="
                    color:#f2f4f8;
                    font-size:16px;
                    font-weight:800;
                ">
                    Model Workspace
                </div>

                <div style="
                    color:#727a8b;
                    font-size:11px;
                    margin-top:3px;
                ">
                    Upload a PyTorch artifact for autonomous optimization.
                </div>

            </div>

            <div style="
                padding:7px 11px;
                border-radius:999px;
                background:rgba(255,142,84,.08);
                border:1px solid rgba(255,142,84,.14);
                color:#ffab80;
                font-size:10px;
                font-weight:800;
                letter-spacing:.09em;
                text-transform:uppercase;
            ">
                PyTorch
            </div>

        </div>

    </div>
    """
)

uploaded_file = st.file_uploader(
    "Upload a PyTorch model",
    type=[
        "pt",
        "pth",
    ],
    label_visibility="collapsed",
    key="model_uploader",
)

if uploaded_file is not None:

    file_size = uploaded_file.size

    render_html(
        f"""
        <div style="
            margin-top:10px;
            padding:12px 14px;
            border-radius:12px;
            background:rgba(255,255,255,.025);
            border:1px solid rgba(255,255,255,.05);
            color:#9ea6b5;
            font-size:11px;
        ">
            Selected:
            <strong style="color:#f0f2f5;">
                {escaped(uploaded_file.name)}
            </strong>
            ·
            {file_size / 1024:.1f} KB
        </div>
        """
    )

    if (
        file_size is not None
        and file_size > MAX_FRONTEND_UPLOAD_SIZE
    ):

        st.error(
            "Model file exceeds the 100 MB upload limit."
        )

    else:

        st.markdown(
            "<div style='height:8px'></div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "⚡  Upload Model",
            type="primary",
            width="stretch",
            key="upload_model_btn",
        ):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/octet-stream",
                )
            }

            with st.spinner(
                "Uploading model artifact..."
            ):

                try:

                    response = api_request(
                        "POST",
                        "/upload",
                        files=files,
                    )

                except requests.RequestException as exc:

                    st.error(
                        f"Upload request failed: {exc}"
                    )

                else:

                    if response.ok:

                        try:

                            upload_data = (
                                response.json()
                            )

                            model_path = (
                                upload_data.get(
                                    "model_path"
                                )
                            )

                        except (
                            ValueError,
                            AttributeError,
                        ):

                            st.error(
                                "Upload succeeded but the backend returned invalid JSON."
                            )

                        else:

                            if not model_path:

                                st.error(
                                    "Backend did not return a model path."
                                )

                            else:

                                st.session_state[
                                    "model_path"
                                ] = model_path

                                st.session_state[
                                    "model_name"
                                ] = uploaded_file.name

                                st.success(
                                    "Model uploaded successfully."
                                )

                                st.rerun()

                    else:

                        try:

                            error_data = (
                                response.json()
                            )

                            error_message = (
                                error_data.get(
                                    "detail",
                                    response.text,
                                )
                            )

                        except (
                            ValueError,
                            AttributeError,
                        ):

                            error_message = (
                                response.text
                                or "Unknown upload error."
                            )

                        st.error(
                            f"Upload failed: {error_message}"
                        )


# ============================================================
# OPTIMIZATION CONTROLS
# ============================================================

if "model_path" in st.session_state:

    section_header(
        "02 · Optimization Engine",
        "Configure optimization",
        "Choose how aggressively the engine should transform the model.",
    )

    control_cols = st.columns(
        [1.35, 1, 1],
        gap="medium",
    )

    with control_cols[0]:

        model_name = st.session_state.get(
            "model_name",
            "Unknown model",
        )

        model_path = st.session_state.get(
            "model_path",
            "",
        )

        render_html(
            f"""
            <div style="
                min-height:120px;
                padding:20px;
                border-radius:20px;
                border:1px solid rgba(255,255,255,.07);
                background:
                    linear-gradient(
                        145deg,
                        rgba(22,25,33,.94),
                        rgba(12,14,19,.94)
                    );
            ">

                <div style="
                    color:#7f8797;
                    font-size:10px;
                    font-weight:800;
                    letter-spacing:.12em;
                    text-transform:uppercase;
                ">
                    Active Model
                </div>

                <div style="
                    margin-top:10px;
                    color:#ffffff;
                    font-size:16px;
                    font-weight:800;
                    word-break:break-word;
                ">
                    {escaped(model_name)}
                </div>

                <div style="
                    margin-top:7px;
                    color:#687183;
                    font-size:9px;
                    word-break:break-word;
                ">
                    {escaped(model_path)}
                </div>

            </div>
            """
        )

    with control_cols[1]:

        strategy = st.selectbox(
            "Optimization Strategy",
            options=[
                "Automatic",
                "pruning",
                "quantization",
                "hybrid",
            ],
            key="optimization_strategy_select",
        )

    with control_cols[2]:

        threshold = st.slider(
            "Maximum Accuracy Drop",
            min_value=0.1,
            max_value=5.0,
            value=1.5,
            step=0.1,
            format="%.1f pp",
            key="accuracy_threshold_slider",
        )

    if st.button(
        "🚀  Start Autonomous Optimization",
        type="primary",
        width="stretch",
        key="start_optimization_btn",
    ):

        selected_strategy = (
            None
            if strategy == "Automatic"
            else strategy
        )

        payload = {
            "model_path": st.session_state[
                "model_path"
            ],
            "strategy": selected_strategy,
            "accuracy_threshold": threshold,
        }

        with st.spinner(
            "Running Neuromorphic-Ops optimization graph..."
        ):

            try:

                response = api_request(
                    "POST",
                    "/optimize",
                    json=payload,
                )

            except requests.RequestException as exc:

                st.error(
                    f"Optimization request failed: {exc}"
                )

            else:

                if response.ok:

                    try:

                        data = response.json()

                    except ValueError:

                        st.error(
                            "Backend returned invalid optimization JSON."
                        )

                    else:

                        thread_id = data.get(
                            "thread_id"
                        )

                        if not thread_id:

                            st.error(
                                "Backend did not return a workflow thread ID."
                            )

                        else:

                            st.session_state[
                                "thread_id"
                            ] = thread_id

                            st.session_state[
                                "state"
                            ] = data.get(
                                "state",
                                {},
                            )

                            st.success(
                                f"Optimization started · {thread_id}"
                            )

                else:

                    try:

                        error_data = response.json()

                        error_message = (
                            error_data.get(
                                "detail",
                                response.text,
                            )
                        )

                    except ValueError:

                        error_message = response.text

                    st.error(
                        f"Optimization failed: {error_message}"
                    )


# ============================================================
# WORKFLOW DASHBOARD
# ============================================================

if "thread_id" in st.session_state:

    state = st.session_state.get(
        "state",
        {},
    )

    thread_id = st.session_state[
        "thread_id"
    ]

    workflow_status = str(
        state.get(
            "workflow_status",
            "unknown",
        )
    )

    deployment_status = str(
        state.get(
            "deployment_status",
            "unknown",
        )
    )

    approval_status = str(
        state.get(
            "approval_status",
            "unknown",
        )
    )

    # --------------------------------------------------------
    # Workflow
    # --------------------------------------------------------

    section_header(
        "03 · Autonomous Pipeline",
        "Workflow command center",
        "Track optimization, evaluation, approval and deployment.",
    )

    show_workflow_rail(
        workflow_status
    )

    status_cols = st.columns(
        4,
        gap="medium",
    )

    status_items = [
        (
            status_cols[0],
            "Workflow",
            workflow_status,
        ),
        (
            status_cols[1],
            "Approval",
            approval_status,
        ),
        (
            status_cols[2],
            "Deployment",
            deployment_status,
        ),
        (
            status_cols[3],
            "Run ID",
            thread_id,
        ),
    ]

    for column, label, value in status_items:

        with column:

            render_html(
                f"""
                <div style="
                    min-height:85px;
                    padding:17px 19px;
                    border-radius:17px;
                    border:1px solid rgba(255,255,255,.065);
                    background:rgba(255,255,255,.027);
                ">

                    <div style="
                        color:#70798b;
                        font-size:9px;
                        font-weight:800;
                        letter-spacing:.11em;
                        text-transform:uppercase;
                    ">
                        {escaped(label)}
                    </div>

                    <div style="
                        margin-top:7px;
                        color:#f3f5f8;
                        font-size:13px;
                        font-weight:800;
                        word-break:break-word;
                    ">
                        {escaped(
                            status_value(value)
                            if label != "Run ID"
                            else value
                        )}
                    </div>

                </div>
                """
            )

    if st.button(
        "↻  Refresh Workflow Status",
        key="refresh_status_btn",
    ):

        try:

            response = api_request(
                "GET",
                f"/status/{thread_id}",
            )

            if response.ok:

                status_data = response.json()

                st.session_state[
                    "state"
                ] = status_data.get(
                    "state",
                    {},
                )

                st.rerun()

            else:

                st.error(
                    f"Status request failed: {response.text}"
                )

        except requests.RequestException as exc:

            st.error(
                f"Backend request failed: {exc}"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    section_header(
        "04 · Intelligence Report",
        "Performance overview",
        "Measured results from the optimized model.",
    )

    show_metrics(
        state
    )

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    section_header(
        "05 · Before vs After",
        "Optimization comparison",
        "Visualize the model footprint and inference behavior.",
    )

    show_optimization_chart(
        state
    )

    # --------------------------------------------------------
    # Details
    # --------------------------------------------------------

    section_header(
        "06 · Strategy Intelligence",
        "Optimization details",
        "Configuration selected by the optimization workflow.",
    )

    detail_cols = st.columns(
        3,
        gap="medium",
    )

    details = [
        (
            detail_cols[0],
            "Strategy",
            status_value(
                state.get(
                    "optimization_strategy",
                    "Not available",
                )
            ),
        ),
        (
            detail_cols[1],
            "Pruning Ratio",
            None,
        ),
        (
            detail_cols[2],
            "Quantization",
            status_value(
                state.get(
                    "quantization_type",
                    "Not available",
                )
            ),
        ),
    ]

    pruning_ratio = state.get(
        "pruning_ratio",
        "Not available",
    )

    if isinstance(
        pruning_ratio,
        (int, float),
    ):

        pruning_text = (
            f"{float(pruning_ratio) * 100:.1f}%"
            if float(pruning_ratio) <= 1
            else f"{float(pruning_ratio):.1f}%"
        )

    else:

        pruning_text = str(
            pruning_ratio
        )

    details[1] = (
        detail_cols[1],
        "Pruning Ratio",
        pruning_text,
    )

    for column, label, value in details:

        with column:

            render_html(
                f"""
                <div style="
                    min-height:110px;
                    padding:20px;
                    border-radius:20px;
                    border:1px solid rgba(255,255,255,.07);
                    background:
                        linear-gradient(
                            145deg,
                            rgba(20,23,30,.94),
                            rgba(11,13,18,.94)
                        );
                ">

                    <div style="
                        color:#7f8797;
                        font-size:10px;
                        font-weight:800;
                        letter-spacing:.09em;
                        text-transform:uppercase;
                    ">
                        {escaped(label)}
                    </div>

                    <div style="
                        margin-top:10px;
                        color:#ffffff;
                        font-size:19px;
                        font-weight:850;
                    ">
                        {escaped(value)}
                    </div>

                </div>
                """
            )

    # --------------------------------------------------------
    # Refinement history
    # --------------------------------------------------------

    history = state.get(
        "refinement_history",
        [],
    )

    if history:

        section_header(
            "07 · Adaptive Loop",
            "Refinement history",
            "Optimization iterations recorded by the workflow.",
        )

        st.dataframe(
            history,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # Human approval
    # --------------------------------------------------------

    if workflow_status == "waiting_for_approval":

        section_header(
            "08 · Human-in-the-Loop",
            "Deployment approval",
            "The optimized artifact is waiting for engineer review.",
        )

        render_html(
            """
            <div style="
                padding:24px;
                border-radius:21px;
                border:1px solid rgba(255,166,86,.16);
                background:
                    linear-gradient(
                        145deg,
                        rgba(65,40,22,.30),
                        rgba(22,18,15,.30)
                    );
            ">

                <div style="
                    color:#fff3e6;
                    font-size:18px;
                    font-weight:850;
                ">
                    Optimization ready for review
                </div>

                <div style="
                    color:#ae9f91;
                    font-size:12px;
                    line-height:1.7;
                    margin-top:6px;
                ">
                    Review the measured metrics before deployment.
                    Approval resumes the workflow and prepares the
                    optimized artifact for deployment.
                </div>

            </div>
            """
        )

        approval_cols = st.columns(
            2,
            gap="medium",
        )

        with approval_cols[0]:

            if st.button(
                "✓  Approve Deployment",
                type="primary",
                width="stretch",
                key="approve_deployment_btn",
            ):

                try:

                    response = api_request(
                        "POST",
                        f"/approve/{thread_id}",
                        json={
                            "approved": True
                        },
                    )

                    if response.ok:

                        result = response.json()

                        st.session_state[
                            "state"
                        ] = result.get(
                            "state",
                            {},
                        )

                        st.success(
                            "Deployment approved."
                        )

                        st.rerun()

                    else:

                        try:

                            error_data = response.json()

                            message = error_data.get(
                                "detail",
                                response.text,
                            )

                        except ValueError:

                            message = response.text

                        st.error(
                            f"Approval failed: {message}"
                        )

                except requests.RequestException as exc:

                    st.error(
                        f"Approval request failed: {exc}"
                    )

        with approval_cols[1]:

            if st.button(
                "✕  Reject Deployment",
                width="stretch",
                key="reject_deployment_btn",
            ):

                try:

                    response = api_request(
                        "POST",
                        f"/approve/{thread_id}",
                        json={
                            "approved": False
                        },
                    )

                    if response.ok:

                        result = response.json()

                        st.session_state[
                            "state"
                        ] = result.get(
                            "state",
                            {},
                        )

                        st.warning(
                            "Deployment rejected."
                        )

                        st.rerun()

                    else:

                        try:

                            error_data = response.json()

                            message = error_data.get(
                                "detail",
                                response.text,
                            )

                        except ValueError:

                            message = response.text

                        st.error(
                            f"Rejection failed: {message}"
                        )

                except requests.RequestException as exc:

                    st.error(
                        f"Rejection request failed: {exc}"
                    )

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    section_header(
        "09 · Deployment Assets",
        "Artifacts",
        "Download the optimized runtime model and generated report.",
    )

    artifact_cols = st.columns(
        2,
        gap="medium",
    )

    onnx_path = state.get(
        "onnx_path"
    )

    report_path = state.get(
        "report_path"
    )

    with artifact_cols[0]:

        render_html(
            f"""
            <div style="
                min-height:145px;
                padding:20px;
                border-radius:18px;
                border:1px solid rgba(255,255,255,.07);
                background:
                    linear-gradient(
                        145deg,
                        rgba(18,21,28,.98),
                        rgba(11,13,18,.98)
                    );
            ">

                <div style="
                    width:40px;
                    height:40px;
                    border-radius:12px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background:rgba(255,122,72,.10);
                    border:1px solid rgba(255,122,72,.14);
                    font-size:18px;
                    margin-bottom:12px;
                ">
                    ⚙
                </div>

                <div style="
                    color:#f1f3f7;
                    font-size:14px;
                    font-weight:800;
                ">
                    Optimized ONNX Model
                </div>

                <div style="
                    color:#727b8c;
                    font-size:10px;
                    margin-top:5px;
                    line-height:1.6;
                    word-break:break-word;
                ">
                    {
                        escaped(onnx_path)
                        if onnx_path
                        else "Generated after successful ONNX export."
                    }
                </div>

            </div>
            """
        )

        if onnx_path:

            try:

                response = api_request(
                    "GET",
                    "/download/onnx",
                )

                if response.ok:

                    st.download_button(
                        label="Download ONNX",
                        data=response.content,
                        file_name="optimized_model.onnx",
                        mime="application/octet-stream",
                        width="stretch",
                        key="download_onnx_btn",
                    )

            except requests.RequestException:

                st.info(
                    "ONNX artifact is currently unavailable."
                )

    with artifact_cols[1]:

        render_html(
            f"""
            <div style="
                min-height:145px;
                padding:20px;
                border-radius:18px;
                border:1px solid rgba(255,255,255,.07);
                background:
                    linear-gradient(
                        145deg,
                        rgba(18,21,28,.98),
                        rgba(11,13,18,.98)
                    );
            ">

                <div style="
                    width:40px;
                    height:40px;
                    border-radius:12px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background:rgba(255,122,72,.10);
                    border:1px solid rgba(255,122,72,.14);
                    font-size:18px;
                    margin-bottom:12px;
                ">
                    ◫
                </div>

                <div style="
                    color:#f1f3f7;
                    font-size:14px;
                    font-weight:800;
                ">
                    Optimization Report
                </div>

                <div style="
                    color:#727b8c;
                    font-size:10px;
                    margin-top:5px;
                    line-height:1.6;
                    word-break:break-word;
                ">
                    {
                        escaped(report_path)
                        if report_path
                        else "Generated after workflow evaluation."
                    }
                </div>

            </div>
            """
        )

        if report_path:

            try:

                response = api_request(
                    "GET",
                    "/report",
                )

                if response.ok:

                    st.download_button(
                        label="Download JSON Report",
                        data=response.content,
                        file_name="optimization_report.json",
                        mime="application/json",
                        width="stretch",
                        key="download_report_btn",
                    )

            except requests.RequestException:

                st.info(
                    "Report download is currently unavailable."
                )

    # --------------------------------------------------------
    # Developer state
    # --------------------------------------------------------

    with st.expander(
        "Developer view · Raw workflow state"
    ):

        st.json(
            state
        )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div style="
        margin-top:48px;
        padding-top:20px;
        border-top:1px solid rgba(255,255,255,.05);
        color:#555d6c;
        font-size:10px;
        text-align:center;
        letter-spacing:.08em;
        text-transform:uppercase;
    ">
        Neuromorphic-Ops · Autonomous Edge-AI Model Quantization
        & Pruning Engine · FastAPI · LangGraph · ONNX Runtime
    </div>
    """
)

