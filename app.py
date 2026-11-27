import os
import re
import sys
import time
from pathlib import Path
from typing import Any
import streamlit as st
from dotenv import load_dotenv

# Ensure local src directory is on sys.path for cloud deployments
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

load_dotenv()

# Mirror Streamlit Cloud secrets into os.environ for LangChain and Tavily
try:
    for secret_key, secret_val in st.secrets.items():
        if isinstance(secret_val, str) and secret_key not in os.environ:
            os.environ[secret_key] = secret_val
except Exception:
    pass

from multi_agent_system.agents import DEFAULT_MODELS
from multi_agent_system.pipeline import run_research_pipeline_stream

st.set_page_config(
    page_title="Research",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Minimal, aesthetic styling
st.markdown(
    """
    <style>
    /* Hide sidebar and decoration */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    #MainMenu,
    footer,
    header {
        display: none !important;
    }

    /* Base Typography & Container */
    .block-container {
        max-width: 820px;
        padding-top: 3.5rem;
        padding-bottom: 5rem;
    }

    h1, h2, h3, p, span, div {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Minimal Header */
    .title-area {
        margin-bottom: 2rem;
    }
    .title-text {
        font-size: 1.85rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }
    .subtitle-text {
        color: #888888;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* Suggestions */
    .chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
        margin-bottom: 2rem;
    }

    /* Metric & Meta row */
    .meta-bar {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 12px 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.9rem;
        color: #888888;
    }
    .meta-item strong {
        color: #ffffff;
        font-weight: 500;
    }

    /* Source item */
    .source-row {
        padding: 12px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .source-row:last-child {
        border-bottom: none;
    }
    .source-title {
        color: #ffffff;
        font-weight: 500;
        font-size: 0.95rem;
        text-decoration: none;
    }
    .source-title:hover {
        text-decoration: underline;
    }
    .source-url {
        font-size: 0.78rem;
        color: #666666;
        margin-top: 2px;
    }
    .source-snippet {
        font-size: 0.85rem;
        color: #999999;
        margin-top: 5px;
        line-height: 1.45;
    }

    /* Review sections */
    .verdict-box {
        padding: 14px 18px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        font-size: 0.92rem;
        color: #dddddd;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .critique-col h4 {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #777777;
        margin-bottom: 0.75rem;
    }
    .critique-item {
        font-size: 0.9rem;
        color: #cccccc;
        margin-bottom: 8px;
        line-height: 1.4;
    }

    /* Clean inputs & buttons */
    div[data-testid="stTextInput"] input {
        border-radius: 10px;
    }
    button[kind="primary"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_critic(text: str) -> dict[str, Any]:
    data = {"score": None, "strengths": [], "improvements": [], "verdict": None}
    if not text:
        return data

    m_score = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
    if m_score:
        data["score"] = m_score.group(1)

    m_verdict = re.search(r"One\s*line\s*verdict:\s*\n*(.+)", text, re.IGNORECASE)
    if m_verdict:
        data["verdict"] = m_verdict.group(1).strip()

    m_str = re.search(
        r"Strengths:\s*\n(.*?)(?=\n\s*(?:Areas to Improve|Improvements|One line verdict|\Z))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m_str:
        data["strengths"] = [
            b.strip("-* ").strip()
            for b in m_str.group(1).split("\n")
            if b.strip() and not b.strip().lower().startswith("strengths")
        ]

    m_imp = re.search(
        r"(?:Areas to Improve|Improvements):\s*\n(.*?)(?=\n\s*(?:One line verdict|\Z))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m_imp:
        data["improvements"] = [
            b.strip("-* ").strip()
            for b in m_imp.group(1).split("\n")
            if b.strip() and not b.strip().lower().startswith("areas to improve")
        ]

    return data


def parse_sources(text: str) -> list[dict[str, str]]:
    sources = []
    if not text:
        return sources
    for block in text.split("----"):
        block = block.strip()
        if not block:
            continue
        title = ""
        url = ""
        snippet = ""
        m_t = re.search(r"Title:\s*(.+)", block)
        m_u = re.search(r"URL:\s*(https?://[^\s\n]+)", block)
        m_s = re.search(r"Snippet:\s*(.+)", block, re.DOTALL)
        if m_t:
            title = m_t.group(1).strip()
        if m_u:
            url = m_u.group(1).strip()
        if m_s:
            snippet = m_s.group(1).strip()
        if url or title:
            sources.append({"title": title or url, "url": url, "snippet": snippet})
    return sources


if "topic" not in st.session_state:
    st.session_state["topic"] = ""
if "research" not in st.session_state:
    st.session_state["research"] = None

# Minimal Header
st.markdown(
    """
    <div class="title-area">
        <div class="title-text">Research</div>
        <div class="subtitle-text">Autonomous web search, deep read, synthesis, and review.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search Input
with st.form("search_form", clear_on_submit=False):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Topic",
            value=st.session_state.get("topic", ""),
            placeholder="Enter any topic or question...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

# Subtle suggestion chips
sample_topics = [
    "Solid-state battery commercialization",
    "Agentic AI frameworks",
    "CRISPR clinical breakthroughs",
]
chip_cols = st.columns(len(sample_topics) + 1)
for i, t in enumerate(sample_topics):
    if chip_cols[i].button(t, key=f"chip_{i}"):
        st.session_state["topic"] = t
        st.rerun()

# Execution
if submitted and query.strip():
    cleaned_topic = query.strip()
    st.session_state["topic"] = cleaned_topic

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("Missing GOOGLE_API_KEY or TAVILY_API_KEY in environment.")
    else:
        state_data: dict[str, Any] = {}
        with st.status("Investigating topic...", expanded=True) as status_box:
            try:
                for event in run_research_pipeline_stream(topic=cleaned_topic):
                    step = event.get("step")
                    status = event.get("status")
                    msg = event.get("message", "")
                    data = event.get("data")

                    if status == "running":
                        status_box.update(label=msg)
                    elif status == "complete":
                        if step == "search":
                            state_data["search"] = data
                        elif step == "reader":
                            state_data["scraped"] = data
                        elif step == "writer":
                            state_data["report"] = data
                        elif step == "critic":
                            state_data["critic"] = data
                        elif step == "done":
                            status_box.update(label="Complete", state="complete", expanded=False)

                st.session_state["research"] = {
                    "topic": cleaned_topic,
                    "search": state_data.get("search", ""),
                    "scraped": state_data.get("scraped", ""),
                    "report": state_data.get("report", ""),
                    "critic": state_data.get("critic", ""),
                    "critic_parsed": parse_critic(state_data.get("critic", "")),
                }
            except Exception as err:
                status_box.update(label="Failed", state="error")
                st.error(str(err))

# Results Display
res = st.session_state.get("research")
if res:
    report = res.get("report", "")
    critic_raw = res.get("critic", "")
    critic_data = res.get("critic_parsed") or parse_critic(critic_raw)
    sources = parse_sources(res.get("search", ""))

    score = critic_data.get("score")
    words = len(report.split()) if report else 0

    score_display = f"{score}/10" if score else "—"
    st.markdown(
        f"""
        <div class="meta-bar">
            <div class="meta-item">Score: <strong>{score_display}</strong></div>
            <div class="meta-item">Length: <strong>{words} words</strong></div>
            <div class="meta-item">Sources: <strong>{len(sources)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_rep, tab_crit, tab_src, tab_scrape = st.tabs(["Report", "Critique", "Sources", "Scraped"])

    with tab_rep:
        col_t, col_d = st.columns([4, 1])
        with col_d:
            st.download_button(
                "Download .md",
                data=report,
                file_name="report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        st.markdown(report)

    with tab_crit:
        verdict = critic_data.get("verdict")
        if verdict:
            st.markdown(f'<div class="verdict-box">{verdict}</div>', unsafe_allow_html=True)

        c_str, c_imp = st.columns(2)
        with c_str:
            st.markdown(
                '<div class="critique-col"><h4>Strengths</h4>'
                + "".join(f'<div class="critique-item">• {s}</div>' for s in critic_data.get("strengths", []))
                + "</div>",
                unsafe_allow_html=True,
            )
        with c_imp:
            st.markdown(
                '<div class="critique-col"><h4>Areas to Improve</h4>'
                + "".join(f'<div class="critique-item">• {i}</div>' for i in critic_data.get("improvements", []))
                + "</div>",
                unsafe_allow_html=True,
            )

    with tab_src:
        if sources:
            for s in sources:
                st.markdown(
                    f"""
                    <div class="source-row">
                        <a href="{s['url']}" target="_blank" class="source-title">{s['title']}</a>
                        <div class="source-url">{s['url']}</div>
                        <div class="source-snippet">{s['snippet']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write(res.get("search", ""))

    with tab_scrape:
        st.text_area("Scraped Content", value=res.get("scraped", ""), height=350, disabled=True, label_visibility="collapsed")
