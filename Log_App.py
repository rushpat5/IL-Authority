import logging
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from urllib.parse import urlparse, urlunparse

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Internal Authority Flow Analyzer",
    page_icon="🌿",
    layout="wide",
)

# =========================================================
# MINIMAL GREEN UI THEME (VERY SUBTLE)
# =========================================================
st.markdown("""
<style>
    :root {
        --accent: #16a34a;        /* subtle green */
        --accent-light: #dcfce7;  /* extremely soft green */
        --bg: #ffffff;
        --bg-soft: #f6f7f9;
        --border: #e5e7eb;
        --text: #1f2937;
        --muted: #6b7280;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-soft);
        border-right: 1px solid var(--border);
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--accent) !important;
        color: white !important;
        padding: 0.45rem 1.2rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        margin-bottom: 12px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 4px;
    }

    .metric-value {
        color: var(--accent);
        font-weight: 700;
        font-size: 1.4rem;
    }

    /* Info box */
    .info-box {
        background: var(--accent-light);
        border-left: 4px solid var(--accent);
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 18px;
        font-size: 0.95rem;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def canonicalize_url(url: str, strip_query=True, strip_fragment=True):
    """Normalizes URLs for consistent comparisons."""
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    query = "" if strip_query else parsed.query
    fragment = "" if strip_fragment else parsed.fragment
    return urlunparse((scheme, netloc, path, "", query, fragment))


def build_graph(df: pd.DataFrame):
    """Creates adjacency structure for PageRank calculation."""
    adjacency = {}
    nodes = set()

    for _, row in df.iterrows():
        src = row["source_url_norm"]
        dst = row["target_url_norm"]
        if not src or not dst or src == dst:
            continue
        nodes.add(src)
        nodes.add(dst)
        adjacency.setdefault(src, []).append(dst)

    return adjacency, sorted(nodes)


def compute_pagerank(adjacency, nodes, damping=0.85, max_iter=50, tol=1e-6):
    """Simple PageRank simulation (internal authority flow)."""
    N = len(nodes)
    if N == 0:
        return {}

    pr = {node: 1.0 / N for node in nodes}
    outdeg = {node: len(adjacency.get(node, [])) for node in nodes}

    for _ in range(max_iter):
        new_pr = {node: (1 - damping) / N for node in nodes}
        dangling_mass = sum(pr[n] for n in nodes if outdeg[n] == 0)

        for src in nodes:
            targets = adjacency.get(src, [])
            if not targets:
                continue
            share = pr[src] * damping / outdeg[src]
            for dst in targets:
                new_pr[dst] += share

        if dangling_mass > 0:
            spread = damping * dangling_mass / N
            for node in nodes:
                new_pr[node] += spread

        if sum(abs(new_pr[n] - pr[n]) for n in nodes) < tol:
            break

        pr = new_pr

    total = sum(pr.values())
    return {k: v / total for k, v in pr.items()}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Upload Your CSV")
    st.write("Your file must include two columns:")
    st.write("• **source_url** → where the link is from\n• **target_url** → where the link points")

    uploaded = st.file_uploader("Upload internal links CSV", type=["csv"])

    st.markdown("---")

    st.subheader("URL Cleaning")
    strip_query = st.checkbox("Remove tracking parameters (?utm etc)", True)
    strip_fragment = st.checkbox("Remove page sections (#...)", True)

    st.markdown("---")

    st.subheader("Filter Navigation/Footer Links")
    boilerplate_threshold = st.slider(
        "If a page is linked from more than this % of pages, hide it",
        0, 100, 80, 5,
    )

    st.markdown("---")

    st.subheader("Authority Flow Settings")
    damping = st.slider(
        "Authority kept per hop (lower = more loss)",
        0.5, 0.95, 0.85, 0.05,
    )


# =========================================================
# MAIN TITLE
# =========================================================
st.title("🌿 Internal Authority Flow Analyzer")

st.markdown("""
<div class="info-box">
<strong>What this tool does:</strong><br>
It shows which pages on your website receive the most internal authority based on how your pages link to each other.<br><br>
We do <em>not</em> visit your pages or read content — we only use your link structure, because internal links decide how authority flows.
</div>
""", unsafe_allow_html=True)

# =========================================================
# UPLOAD CHECK
# =========================================================
if not uploaded:
    st.info("Upload a CSV file in the sidebar to begin.")
    st.stop()

# =========================================================
# LOAD & PREPARE DATA
# =========================================================
df = pd.read_csv(uploaded)
df.columns = [c.lower() for c in df.columns]

if not {"source_url", "target_url"}.issubset(df.columns):
    st.error("Your CSV must contain 'source_url' and 'target_url' columns.")
    st.stop()

df["source_url_norm"] = df["source_url"].apply(lambda x: canonicalize_url(x, strip_query, strip_fragment))
df["target_url_norm"] = df["target_url"].apply(lambda x: canonicalize_url(x, strip_query, strip_fragment))

df = df[
    (df["source_url_norm"] != "") &
    (df["target_url_norm"] != "") &
    (df["source_url_norm"] != df["target_url_norm"])
]

# =========================================================
# REMOVE NAV/FOOTER LINKS
# =========================================================
page_count = df["source_url_norm"].nunique()
target_sources = df.groupby("target_url_norm")["source_url_norm"].nunique().reset_index(name="num_sources")
target_sources["source_fraction"] = target_sources["num_sources"] / page_count

boilerplate_targets = set(
    target_sources.loc[target_sources["source_fraction"] >= boilerplate_threshold / 100, "target_url_norm"]
)

df_filtered = df[~df["target_url_norm"].isin(boilerplate_targets)]

# =========================================================
# BUILD GRAPH + CALCULATE INTERNAL AUTHORITY
# =========================================================
adj, nodes = build_graph(df_filtered)
scores = compute_pagerank(adj, nodes, damping=damping)

df_scores = pd.DataFrame([{"url": u, "authority": s} for u, s in scores.items()])
df_scores = df_scores.sort_values("authority", ascending=False)
max_score = df_scores["authority"].max()
df_scores["index"] = (df_scores["authority"] / max_score) * 100

# =========================================================
# METRIC CARDS
# =========================================================
c1, c2, c3 = st.columns(3)

c1.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Pages Analyzed</div>
    <div class="metric-value">{len(df_scores)}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Links Used</div>
    <div class="metric-value">{len(df_filtered)}</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Navigation Links Hidden</div>
    <div class="metric-value">{len(boilerplate_targets)}</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# BAR CHART
# =========================================================
st.markdown("### 🔝 Top Pages by Internal Authority")

top_n = st.slider("Show top pages:", 10, 200, 40)

fig = px.bar(
    df_scores.head(top_n),
    x="index",
    y="url",
    orientation="h",
    color="index",
    color_continuous_scale=["#dcfce7", "#16a34a"],
    labels={"index": "Authority Index (0–100)", "url": ""},
)

fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FULL TABLE
# =========================================================
st.markdown("### 📄 All Pages (Ranked)")
st.dataframe(df_scores, hide_index=True, use_container_width=True)

st.markdown("""
**How to read this:**  
- A higher score → the page receives more internal authority  
- Pages near index 100 → strongest hubs  
- Pages under 20 → weak or isolated  
""")
