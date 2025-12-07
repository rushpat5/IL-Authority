import logging
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from urllib.parse import urlparse, urlunparse

# =========================================================
# PAGE CONFIG & STYLE
# =========================================================
st.set_page_config(
    page_title="Internal Authority Flow Analyzer",
    page_icon="🌿",
    layout="wide",
)

# Custom green UI theme
st.markdown(
    """
    <style>
        :root {
            --green-main: #10b981;
            --green-soft: #ecfdf5;
            --bg: #ffffff;
            --text: #064e3b;
            --border: #d1fae5;
            --muted: #6b7280;
        }

        .stApp {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }

        section[data-testid="stSidebar"] {
            background-color: var(--green-soft);
            border-right: 1px solid var(--border);
        }

        .metric-card {
            background: #ffffff;
            border-radius: 10px;
            border: 1px solid var(--border);
            padding: 14px 18px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.04);
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--muted);
        }

        .metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--green-main);
        }

        .context-box {
            background: var(--green-soft);
            border-left: 4px solid var(--green-main);
            padding: 14px;
            margin-bottom: 16px;
            border-radius: 6px;
            color: var(--text);
        }

        .stButton > button {
            background-color: var(--green-main) !important;
            color: white !important;
            padding: 0.5rem 1.3rem !important;
            border-radius: 8px !important;
            border: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def canonicalize_url(url: str, strip_query=True, strip_fragment=True):
    """Clean and simplify URLs so similar pages are treated as one."""
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
    """Turn your list of links into a map of how your website connects."""
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
    """Simple internal authority flow model (PageRank style)."""
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
            if not targets: continue
            share = pr[src] * damping / outdeg[src]
            for dst in targets:
                new_pr[dst] += share

        # spread dangling
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
# SIDEBAR (USER INPUT)
# =========================================================
with st.sidebar:
    st.header("Upload Your Data")
    uploaded = st.file_uploader(
        "Upload your internal links CSV",
        type=["csv"],
        help="Your CSV must contain: source_url, target_url",
    )

    st.markdown("### URL Cleaning Options")
    strip_query = st.checkbox("Remove tracking parameters", True)
    strip_fragment = st.checkbox("Remove #sections", True)

    st.markdown("### Remove Navigation/Footer Links")
    boilerplate_threshold = st.slider(
        "Hide pages linked from more than this % of your site",
        0, 100, 80, 5,
    )

    st.markdown("### Authority Flow Settings")
    damping = st.slider(
        "Authority retained per hop (lower = more loss)",
        0.5, 0.95, 0.85, 0.05,
    )

# =========================================================
# MAIN UI
# =========================================================
st.title("🌿 Internal Authority Flow Analyzer")

st.markdown(
    """
<div class="context-box">
<strong>What this tool shows:</strong><br>
• Which pages on your site receive the most internal authority  
• Which pages are weak or isolated  
• How your internal links help (or hurt) your important pages  
<br><br>
<strong>How it works:</strong>  
We don't visit your pages or read content.  
We only look at your internal link structure, because it controls how authority flows.
</div>
""",
    unsafe_allow_html=True,
)

if not uploaded:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# =========================================================
# LOAD & CLEAN DATA
# =========================================================
df = pd.read_csv(uploaded)
df.columns = [c.lower() for c in df.columns]

if not {"source_url", "target_url"}.issubset(df.columns):
    st.error("Your CSV is missing required columns.")
    st.stop()

df["source_url_norm"] = df["source_url"].apply(
    lambda x: canonicalize_url(x, strip_query, strip_fragment)
)
df["target_url_norm"] = df["target_url"].apply(
    lambda x: canonicalize_url(x, strip_query, strip_fragment)
)

df = df[
    (df["source_url_norm"] != "") &
    (df["target_url_norm"] != "") &
    (df["source_url_norm"] != df["target_url_norm"])
]

# Remove boilerplate (nav/footer)
page_count = df["source_url_norm"].nunique()
target_sources = (
    df.groupby("target_url_norm")["source_url_norm"]
    .nunique()
    .reset_index(name="num_sources")
)
target_sources["source_fraction"] = target_sources["num_sources"] / page_count
boilerplate_targets = set(
    target_sources.loc[target_sources["source_fraction"] >= boilerplate_threshold / 100, "target_url_norm"]
)

df_filtered = df[~df["target_url_norm"].isin(boilerplate_targets)]

# =========================================================
# GRAPH + PAGERANK
# =========================================================
adj, nodes = build_graph(df_filtered)
scores = compute_pagerank(adj, nodes, damping=damping)

df_scores = pd.DataFrame(
    [{"url": u, "authority": s} for u, s in scores.items()]
)
df_scores = df_scores.sort_values("authority", ascending=False)
max_score = df_scores["authority"].max()
df_scores["index"] = (df_scores["authority"] / max_score) * 100

# =========================================================
# OUTPUT
# =========================================================
c1, c2, c3 = st.columns(3)

c1.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Pages Analyzed</div>
        <div class="metric-value">{len(df_scores)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c2.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Internal Links Used</div>
        <div class="metric-value">{len(df_filtered)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c3.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Navigation Links Removed</div>
        <div class="metric-value">{len(boilerplate_targets)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 🔝 Your Strongest Pages (Top Authority)")
top_n = st.slider("How many pages to display?", 10, 200, 40)

fig = px.bar(
    df_scores.head(top_n),
    x="index",
    y="url",
    orientation="h",
    color="index",
    color_continuous_scale=["#d1fae5", "#10b981"],
    labels={"index": "Authority Index (0–100)", "url": ""},
)
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📄 All Pages")
st.dataframe(df_scores, hide_index=True, use_container_width=True)

st.markdown(
    """
**How to read this:**  
- A higher score means the page receives more internal authority  
- Pages near index 100 are your strongest hubs  
- Pages under 20 are weak, isolated, or poorly linked  
"""
)
