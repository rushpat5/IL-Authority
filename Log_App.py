import logging
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from urllib.parse import urlparse, urlunparse

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# STREAMLIT CONFIG & CUSTOM STYLES
# ---------------------------------------------------------
st.set_page_config(
    page_title="Internal Authority Flow Analyzer",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --brand: #2563eb;
            --brand-soft: #dbeafe;
            --bg: #ffffff;
            --text: #0f172a;
            --border: #e2e8f0;
            --muted: #64748b;
        }

        .stApp {
            background-color: var(--bg);
            color: var(--text);
            font-family: system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid var(--border);
        }

        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 14px 18px;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.03);
        }

        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            color: var(--muted);
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text);
        }

        .context-box {
            background: #ecfdf5;
            border-left: 4px solid #16a34a;
            padding: 14px 16px;
            border-radius: 8px;
            margin-bottom: 14px;
            color: #14532d;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .stButton > button {
            background-color: var(--brand) !important;
            color: #ffffff !important;
            border-radius: 999px !important;
            padding: 0.5rem 1.4rem !important;
            border: none;
            font-weight: 600;
            font-size: 0.95rem !important;
        }

        .pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            background: var(--brand-soft);
            color: var(--brand);
            font-weight: 500;
            margin-right: 6px;
            margin-bottom: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def canonicalize_url(
    url: str,
    strip_query: bool = True,
    strip_fragment: bool = True,
) -> str:
    """
    Normalize URLs so that small differences don't fragment the graph.
    """
    if not isinstance(url, str):
        return ""

    url = url.strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url)
    except Exception:
        return url

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()

    path = parsed.path or "/"
    # normalize trailing slash
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    query = "" if strip_query else (parsed.query or "")
    fragment = "" if strip_fragment else (parsed.fragment or "")

    return urlunparse((scheme, netloc, path, "", query, fragment))


def build_graph(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Build adjacency list from filtered internal links.
    Returns adjacency dict and sorted list of nodes.
    """
    adjacency: Dict[str, List[str]] = {}
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


def compute_pagerank(
    adjacency: Dict[str, List[str]],
    nodes: List[str],
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Simple PageRank implementation:
    - authority flows through links with damping factor (e.g. 0.85).
    - (1 - damping) is the "loss" or teleport factor per hop.
    """
    N = len(nodes)
    if N == 0:
        return {}

    # initial uniform distribution
    pr = {node: 1.0 / N for node in nodes}
    outdeg = {node: len(adjacency.get(node, [])) for node in nodes}

    for _ in range(max_iter):
        new_pr = {node: (1.0 - damping) / N for node in nodes}
        dangling_mass = sum(pr[node] for node in nodes if outdeg[node] == 0)

        # distribute from nodes with outgoing links
        for src in nodes:
            targets = adjacency.get(src, [])
            if not targets:
                continue
            share = pr[src] * damping / outdeg[src]
            for dst in targets:
                new_pr[dst] += share

        # distribute dangling mass
        if dangling_mass > 0:
            dangling_share = damping * dangling_mass / N
            for node in nodes:
                new_pr[node] += dangling_share

        diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
        pr = new_pr
        if diff < tol:
            break

    # normalize to sum to 1
    total = sum(pr.values())
    if total > 0:
        pr = {k: v / total for k, v in pr.items()}

    return pr


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded = st.file_uploader(
        "Internal links CSV",
        type=["csv"],
        help="Required columns (case-insensitive): source_url, target_url",
    )

    key_pages_file = st.file_uploader(
        "Key / money pages CSV (optional)",
        type=["csv"],
        help="At minimum, a column named 'url'. These are pages you care about most.",
    )

    st.markdown("**URL Normalization**")
    strip_query = st.checkbox("Strip query parameters (?utm=…)", value=True)
    strip_fragment = st.checkbox("Strip fragments (#section)", value=True)

    st.markdown("---")
    st.markdown("**Navigation / Boilerplate Filtering**")

    boilerplate_threshold = st.slider(
        "Hide targets linked from more than X% of pages",
        min_value=0,
        max_value=100,
        value=80,
        step=5,
        help=(
            "Targets linked from more than this percentage of unique pages "
            "are treated as nav/footer and removed from the graph."
        ),
    )

    st.markdown("---")
    st.markdown("**Asset Filtering (Recommended)**")

    filter_assets = st.checkbox(
        "Filter out images, scripts, PDFs, fonts, etc.",
        value=True,
        help="Removes non-HTML URLs such as .png, .svg, .css, .js, .pdf from the internal-link graph.",
    )

    with st.expander("Advanced asset filter settings"):
        custom_ext_input = st.text_area(
            "Blocked extensions (comma-separated)",
            ".png, .jpg, .jpeg, .gif, .svg, .webp, .ico, .css, .js, .woff, .woff2, .ttf, .eot, .pdf, .zip",
            help="Edit only if your site uses unusual file types.",
        )
        blocked_extensions = tuple(
            ext.strip().lower()
            for ext in custom_ext_input.split(",")
            if ext.strip()
        )

    st.markdown("---")
    st.markdown("**Authority Flow Model**")

    damping = st.slider(
        "Damping factor (authority retained per hop)",
        min_value=0.5,
        max_value=0.95,
        value=0.85,
        step=0.05,
        help="0.85 ≈ ~15% authority loss per hop.",
    )

    max_iter = st.slider(
        "Max PageRank iterations",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
    )

# ---------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------
st.title("🕸️ Internal Authority Flow Analyzer")
st.markdown(
    "Understand how link authority moves through your site based on internal links."
)

st.markdown(
    """
<div class="context-box">
<strong>Concept:</strong> External backlinks give your site authority, but internal links decide where that authority actually goes.  
This tool builds a map of your internal links and runs a PageRank-style model to show which pages accumulate authority and which ones are structurally weak.  
Upload a list of your <em>key / money pages</em> to see how much of that authority actually reaches them.
</div>
""",
    unsafe_allow_html=True,
)

if uploaded is None:
    st.info(
        "Upload a CSV of internal links in the sidebar to get started.\n\n"
        "Minimum columns: `source_url`, `target_url`."
    )
    st.stop()

# ---------------------------------------------------------
# LOAD & CLEAN INTERNAL LINKS
# ---------------------------------------------------------
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read internal links CSV: {e}")
    st.stop()

# normalize column names to lowercase
df.columns = [c.lower() for c in df.columns]
required_cols = {"source_url", "target_url"}

if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

df = df[["source_url", "target_url"]].dropna()

# canonicalize URLs
df["source_url_norm"] = df["source_url"].apply(
    lambda x: canonicalize_url(
        x, strip_query=strip_query, strip_fragment=strip_fragment
    )
)
df["target_url_norm"] = df["target_url"].apply(
    lambda x: canonicalize_url(
        x, strip_query=strip_query, strip_fragment=strip_fragment
    )
)

df = df[
    (df["source_url_norm"] != "") &
    (df["target_url_norm"] != "") &
    (df["source_url_norm"] != df["target_url_norm"])
].copy()

# ---------------------------------------------------------
# FILTER OUT NON-HTML ASSETS (images, scripts, fonts, PDFs)
# ---------------------------------------------------------
def is_asset(url: str) -> bool:
    if not isinstance(url, str):
        return False
    url_lower = url.lower()
    return any(url_lower.endswith(ext) for ext in blocked_extensions)

if filter_assets:
    df_before = df.shape[0]
    df = df[
        (~df["source_url_norm"].apply(is_asset)) &
        (~df["target_url_norm"].apply(is_asset))
    ].copy()
    df_after = df.shape[0]

    if df_before > 0:
        removed_pct = (df_before - df_after) / df_before * 100
        if removed_pct >= 80:
            st.warning(
                f"⚠️ Asset filtering removed {removed_pct:.1f}% of internal links. "
                f"This may indicate your crawl export included mostly non-HTML resources."
            )

if df.empty:
    st.error("All links were filtered out (possibly due to asset filtering or URL normalization).")
    st.stop()

# ---------------------------------------------------------
# BOILERPLATE / NAV FILTERING
# ---------------------------------------------------------
page_count = df["source_url_norm"].nunique()
target_sources = (
    df.groupby("target_url_norm")["source_url_norm"]
    .nunique()
    .reset_index(name="num_sources")
)

target_sources["source_fraction"] = target_sources["num_sources"] / max(page_count, 1)
boilerplate_cutoff = boilerplate_threshold / 100.0
boilerplate_targets = set(
    target_sources.loc[
        target_sources["source_fraction"] >= boilerplate_cutoff, "target_url_norm"
    ]
)

df_filtered = df[~df["target_url_norm"].isin(boilerplate_targets)].copy()

if df_filtered.empty:
    st.warning(
        "After removing boilerplate/nav targets, no internal editorial links remain. "
        "Try lowering the boilerplate threshold."
    )
    st.stop()

# ---------------------------------------------------------
# BUILD GRAPH & RUN PAGERANK
# ---------------------------------------------------------
adjacency, nodes = build_graph(df_filtered)
pagerank_scores = compute_pagerank(
    adjacency,
    nodes,
    damping=damping,
    max_iter=max_iter,
)

if not pagerank_scores:
    st.warning("Could not compute authority scores. Check your input data.")
    st.stop()

df_scores = pd.DataFrame(
    [{"url": u, "authority_score": s} for u, s in pagerank_scores.items()]
).sort_values("authority_score", ascending=False)

# normalize to 0–100 index
max_score = df_scores["authority_score"].max()
if max_score > 0:
    df_scores["authority_index"] = (df_scores["authority_score"] / max_score) * 100
else:
    df_scores["authority_index"] = 0.0

# ---------------------------------------------------------
# OPTIONAL: LOAD KEY / MONEY PAGES
# ---------------------------------------------------------
has_key_pages = False
authority_share_money = None
matched_key_pages = 0
total_key_pages = 0

if key_pages_file is not None:
    try:
        df_keys = pd.read_csv(key_pages_file)
        df_keys.columns = [c.lower() for c in df_keys.columns]
        if "url" in df_keys.columns:
            df_keys = df_keys[["url"]].dropna()
            df_keys["url_norm"] = df_keys["url"].apply(
                lambda x: canonicalize_url(
                    x, strip_query=strip_query, strip_fragment=strip_fragment
                )
            )
            df_keys = df_keys[df_keys["url_norm"] != ""].drop_duplicates(
                subset=["url_norm"]
            )
            total_key_pages = df_keys.shape[0]

            df_scores["is_key_page"] = df_scores["url"].isin(df_keys["url_norm"])
            has_key_pages = True

            authority_share_money = df_scores.loc[
                df_scores["is_key_page"], "authority_score"
            ].sum()
            matched_key_pages = int(df_scores["is_key_page"].sum())
        else:
            st.warning(
                "Key pages CSV must have a 'url' column. Ignoring key pages file."
            )
    except Exception as e:
        st.warning(f"Could not read key pages CSV: {e}")

if not has_key_pages:
    df_scores["is_key_page"] = False

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Pages in Graph</div>
            <div class="metric-value">{df_scores.shape[0]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Internal Links (filtered)</div>
            <div class="metric-value">{df_filtered.shape[0]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Boilerplate Targets Removed</div>
            <div class="metric-value">{len(boilerplate_targets)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Damping Factor</div>
            <div class="metric-value">{damping:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    if has_key_pages and authority_share_money is not None:
        share_pct = authority_share_money * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Authority on Key Pages</div>
                <div class="metric-value">{share_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Authority on Key Pages</div>
                <div class="metric-value">–</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# CHART & TABLE
# ---------------------------------------------------------
st.markdown("### 🔎 Top Pages by Internal Authority")

view_only_keys = False
if has_key_pages:
    view_only_keys = st.checkbox(
        "Show only key / money pages in chart and table",
        value=False,
    )

if view_only_keys:
    df_view = df_scores[df_scores["is_key_page"]].copy()
else:
    df_view = df_scores.copy()

if df_view.empty:
    st.warning("No pages to display with current filters.")
    st.stop()

top_n = st.slider(
    "Show top N pages",
    min_value=10,
    max_value=min(200, df_view.shape[0]),
    value=min(50, df_view.shape[0]),
    step=10,
)

fig = px.bar(
    df_view.head(top_n),
    x="authority_index",
    y="url",
    orientation="h",
    labels={
        "authority_index": "Authority Index (0–100)",
        "url": "URL",
    },
    height=600,
    color="is_key_page",
    color_discrete_map={False: "#93c5fd", True: "#2563eb"},
)
fig.update_layout(
    yaxis=dict(autorange="reversed"),
    margin=dict(l=0, r=10, t=40, b=10),
    legend_title_text="Key Page",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📄 Full Authority Table")

df_table = df_view[["url", "authority_score", "authority_index", "is_key_page"]].copy()
df_table.rename(columns={"is_key_page": "key_page"}, inplace=True)

st.dataframe(
    df_table,
    use_container_width=True,
    hide_index=True,
)

if has_key_pages:
    st.markdown(
        f"""
**Key pages overlay:**

- Total key pages in your CSV: **{total_key_pages}**  
- Key pages present in the internal link graph: **{matched_key_pages}**  
- Share of internal authority on those matched key pages: **{authority_share_money * 100:.1f}%** (via `authority_score`)
        """
    )
else:
    st.markdown(
        """
Upload a **Key / money pages CSV** in the sidebar (with a `url` column)  
to see how much internal authority your most important pages receive.
        """
    )

st.markdown(
    """
**How to interpret the scores:**

- `authority_score` is the raw PageRank value (all pages sum to 1.0).
- `authority_index` rescales this so the strongest page becomes 100 and others scale relative to it.
- If key / money pages have **low authority_index** or capture a **small share of authority**, your current internal linking is likely under-supporting them.
"""
)
