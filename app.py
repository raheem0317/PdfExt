import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from services.ai_processor import process_chunks_parallel

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Department Analyzer – Dual PDF",
    layout="wide",
    page_icon="📄",
)

# ─────────────────────────────────────────────
# Custom CSS for a premium look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hero header ── */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .hero p  { font-size: 1rem; opacity: .75; margin: .4rem 0 0; }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        text-align: center;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; }
    .metric-card .lbl { font-size: .8rem; opacity: .85; }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #667eea;
        padding-left: .7rem;
        margin: 1.5rem 0 .8rem;
    }

    /* ── PDF badge colours ── */
    .badge-pdf1 {
        background:#667eea22; border-radius:8px; padding:.2rem .7rem;
        color:#667eea; font-weight:600;
    }
    .badge-pdf2 {
        background:#f64f5922; border-radius:8px; padding:.2rem .7rem;
        color:#f64f59; font-weight:600;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #e8e8e8; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📄 AI Department &amp; Application Analyzer</h1>
  <p>Upload two PDFs — AI extracts departments, applications &amp; relationships and renders a live comparison dashboard.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached processing helpers
# ─────────────────────────────────────────────
import os as _os
import pathlib

PDF_FOLDER = pathlib.Path(__file__).parent / "pdf"


@st.cache_data(show_spinner=False)
def process_pdf(pdf_bytes: bytes, filename: str):
    """Extract + AI-process a PDF. Cached by file bytes."""
    import io
    text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    if not text.strip():
        return []
    chunks = chunk_text(text)
    return process_chunks_parallel(chunks)


@st.cache_data(show_spinner=False)
def process_pdf_from_path(file_path: str):
    """Extract + AI-process a PDF from a file path. Cached by path + mtime."""
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        return []
    chunks = chunk_text(text)
    return process_chunks_parallel(chunks)


def build_dataframe(raw: list, source_label: str) -> pd.DataFrame:
    """Turn raw AI output list into a clean DataFrame."""
    cols = ["department", "application", "relationship", "business_context"]
    if not raw:
        return pd.DataFrame(columns=cols + ["source"])
    
    df = pd.DataFrame(raw)
    for col in cols:
        if col not in df.columns:
            df[col] = "Unknown"
    
    # 1. Clean and Normalize strings
    for col in ["department", "application"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    
    # 2. Filter out noise (records with no actual application found)
    noise_values = ["None Mentioned", "None", "Unknown", "N/A", "", "None."]
    df = df[~df["application"].isin(noise_values)]
    
    # 3. Handle duplicates: Group by Dept/App and join contexts if they differ
    if not df.empty:
        df = df.groupby(["department", "application"]).agg({
            "relationship": "first",
            "business_context": lambda x: " | ".join(set(str(i) for i in x if str(i).lower() != "unknown"))
        }).reset_index()
    
    # Fill remaining blanks
    df["business_context"] = df["business_context"].replace("", "Details in report")
    df["source"] = source_label
    return df


# ─────────────────────────────────────────────
# Auto-detect PDFs from the pdf/ folder
# ─────────────────────────────────────────────
local_pdfs = sorted(PDF_FOLDER.glob("*.pdf")) if PDF_FOLDER.exists() else []

# Sidebar: show detected files
with st.sidebar:
    st.markdown("### 📂 PDF Source")
    if local_pdfs:
        st.success(f"Found **{len(local_pdfs)}** PDF(s) in `pdf/` folder:")
        for p in local_pdfs:
            st.markdown(f"- `{p.name}`")
    else:
        st.warning("No PDFs found in the `pdf/` folder.")

# ─────────────────────────────────────────────
# Guard: need at least 2 PDFs in the folder
# ─────────────────────────────────────────────
if len(local_pdfs) < 2:
    st.info(
        f"📁 Need at least **2 PDFs** in the `pdf/` folder. "
        f"Currently found **{len(local_pdfs)}**. Please add PDF files and refresh."
    )
    st.stop()

pdf1_name = local_pdfs[0].name
pdf2_name = local_pdfs[1].name

# Show active PDFs in the main area
col_b1, col_b2 = st.columns(2)
col_b1.markdown(f'<p class="badge-pdf1">📘 PDF 1: {pdf1_name}</p>', unsafe_allow_html=True)
col_b2.markdown(f'<p class="badge-pdf2">📕 PDF 2: {pdf2_name}</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Process with progress feedback
# ─────────────────────────────────────────────
with st.spinner(f"🤖 Analysing **{pdf1_name}** …"):
    raw1 = process_pdf_from_path(str(PDF_FOLDER / pdf1_name))

with st.spinner(f"🤖 Analysing **{pdf2_name}** …"):
    raw2 = process_pdf_from_path(str(PDF_FOLDER / pdf2_name))

df1 = build_dataframe(raw1, pdf1_name)
df2 = build_dataframe(raw2, pdf2_name)

if df1.empty and df2.empty:
    st.error("❌ No data could be extracted from either PDF.")
    st.stop()

combined = pd.concat([df1, df2], ignore_index=True)

# ─────────────────────────────────────────────
# ── KPI row ──
# ─────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, val, label):
    col.markdown(f"""
    <div class="metric-card">
      <div class="val">{val}</div>
      <div class="lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, len(combined),                          "Total Records")
kpi(k2, combined["department"].nunique(),        "Unique Departments")
kpi(k3, combined["application"].nunique(),       "Unique Applications")
kpi(k4, len(df1),                               f"Records – PDF 1")
kpi(k5, len(df2),                               f"Records – PDF 2")

st.markdown('<hr>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ── Tabs ──
# ─────────────────────────────────────────────
tab_pdf1, tab_pdf2, tab_combined, tab_compare, tab_graph = st.tabs([
    f"📘 {pdf1_name[:25]}",
    f"📕 {pdf2_name[:25]}",
    "🔗 Combined View",
    "📊 Comparison",
    "🕸️ Network Graph",
])

# ── Tab 1: PDF 1 ─────────────────────────────
with tab_pdf1:
    if df1.empty:
        st.warning("No data extracted from PDF 1.")
    else:
        st.markdown('<div class="section-title">Raw Data – PDF 1</div>', unsafe_allow_html=True)
        st.dataframe(df1.drop(columns=["source"]), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Departments (PDF 1)</div>', unsafe_allow_html=True)
            st.bar_chart(df1["department"].value_counts())
        with c2:
            st.markdown('<div class="section-title">Relationships (PDF 1)</div>', unsafe_allow_html=True)
            st.bar_chart(df1["relationship"].value_counts())

        st.markdown('<div class="section-title">Filter by Department – PDF 1</div>', unsafe_allow_html=True)
        sel1 = st.selectbox("Select department", df1["department"].unique(), key="dep1")
        st.dataframe(df1[df1["department"] == sel1].drop(columns=["source"]), width="stretch")

# ── Tab 2: PDF 2 ─────────────────────────────
with tab_pdf2:
    if df2.empty:
        st.warning("No data extracted from PDF 2.")
    else:
        st.markdown('<div class="section-title">Raw Data – PDF 2</div>', unsafe_allow_html=True)
        st.dataframe(df2.drop(columns=["source"]), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Departments (PDF 2)</div>', unsafe_allow_html=True)
            st.bar_chart(df2["department"].value_counts())
        with c2:
            st.markdown('<div class="section-title">Relationships (PDF 2)</div>', unsafe_allow_html=True)
            st.bar_chart(df2["relationship"].value_counts())

        st.markdown('<div class="section-title">Filter by Department – PDF 2</div>', unsafe_allow_html=True)
        sel2 = st.selectbox("Select department", df2["department"].unique(), key="dep2")
        st.dataframe(df2[df2["department"] == sel2].drop(columns=["source"]), width="stretch")

# ── Tab 3: Combined ───────────────────────────
with tab_combined:
    st.markdown('<div class="section-title">All Records (Both PDFs)</div>', unsafe_allow_html=True)

    # Source colour badge in table
    st.dataframe(combined, width="stretch")

    st.markdown('<div class="section-title">Filter by Department (Combined)</div>', unsafe_allow_html=True)
    sel_c = st.selectbox("Select department", combined["department"].unique(), key="depc")
    st.dataframe(combined[combined["department"] == sel_c], width="stretch")

# ── Tab 4: Comparison ────────────────────────
with tab_compare:
    st.markdown('<div class="section-title">Department Count Comparison</div>', unsafe_allow_html=True)

    depts_1 = df1["department"].value_counts().rename("PDF 1")
    depts_2 = df2["department"].value_counts().rename("PDF 2")
    dept_compare = pd.concat([depts_1, depts_2], axis=1).fillna(0).astype(int)
    st.bar_chart(dept_compare)

    st.markdown('<div class="section-title">Relationship Type Comparison</div>', unsafe_allow_html=True)
    rel_1 = df1["relationship"].value_counts().rename("PDF 1")
    rel_2 = df2["relationship"].value_counts().rename("PDF 2")
    rel_compare = pd.concat([rel_1, rel_2], axis=1).fillna(0).astype(int)
    st.bar_chart(rel_compare)

    st.markdown('<div class="section-title">Departments Shared vs Unique</div>', unsafe_allow_html=True)
    set1 = set(df1["department"].unique())
    set2 = set(df2["department"].unique())
    shared   = set1 & set2
    only_1   = set1 - set2
    only_2   = set2 - set1

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("🔗 Shared Departments",   len(shared))
    sc2.metric(f"Only in PDF 1",          len(only_1))
    sc3.metric(f"Only in PDF 2",          len(only_2))

    if shared:
        st.info("**Shared departments:** " + ", ".join(sorted(shared)))
    if only_1:
        st.markdown(f"**Only in PDF 1:** " + ", ".join(sorted(only_1)))
    if only_2:
        st.markdown(f"**Only in PDF 2:** " + ", ".join(sorted(only_2)))

# ── Tab 5: Network Graph ──────────────────────
with tab_graph:
    st.markdown('<div class="section-title">Department–Application Network (Both PDFs)</div>', unsafe_allow_html=True)

    # Limit to avoid rendering overload
    plot_df = combined.head(50)

    G = nx.Graph()
    for _, row in plot_df.iterrows():
        dept = row["department"]
        app  = row["application"]
        src  = row["source"]
        G.add_node(dept, node_type="dept", source=src)
        G.add_node(app,  node_type="app",  source=src)
        G.add_edge(dept, app, source=src)

    pos = nx.spring_layout(G, k=1.2, seed=42)

    dept_nodes_1 = [n for n in G.nodes if G.nodes[n].get("node_type") == "dept" and G.nodes[n].get("source") == pdf1_name]
    dept_nodes_2 = [n for n in G.nodes if G.nodes[n].get("node_type") == "dept" and G.nodes[n].get("source") == pdf2_name]
    app_nodes_1  = [n for n in G.nodes if G.nodes[n].get("node_type") == "app"  and G.nodes[n].get("source") == pdf1_name]
    app_nodes_2  = [n for n in G.nodes if G.nodes[n].get("node_type") == "app"  and G.nodes[n].get("source") == pdf2_name]

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    nx.draw_networkx_nodes(G, pos, nodelist=dept_nodes_1, node_color="#667eea", node_shape="s", node_size=500, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=dept_nodes_2, node_color="#f64f59", node_shape="s", node_size=500, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=app_nodes_1,  node_color="#43e97b", node_shape="o", node_size=350, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=app_nodes_2,  node_color="#fa8231", node_shape="o", node_size=350, ax=ax)

    nx.draw_networkx_edges(G, pos, edge_color="#ffffff30", width=1, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", ax=ax)

    legend_handles = [
        mpatches.Patch(color="#667eea", label=f"Dept – PDF 1"),
        mpatches.Patch(color="#f64f59", label=f"Dept – PDF 2"),
        mpatches.Patch(color="#43e97b", label=f"App  – PDF 1"),
        mpatches.Patch(color="#fa8231", label=f"App  – PDF 2"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              facecolor="#1a1a2e", edgecolor="none", labelcolor="white", fontsize=8)

    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)
st.caption("AI Department Analyzer · Powered by Groq · Built with Streamlit")
