import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from services.ai_processor import process_chunks_parallel

load_dotenv()


@st.cache_data
def process_pdf(pdf_file):
    try:
        st.info("Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            st.error("No text could be extracted from the PDF. The PDF might be scanned images or empty.")
            return []
            
        st.info(f"Extracted {len(text)} characters of text")
        st.info("Processing text with AI...")
        chunks = chunk_text(text)
        st.info(f"Split into {len(chunks)} chunks for processing")
        
        result = process_chunks_parallel(chunks)
        st.info(f"AI processing complete. Found {len(result)} items")
        return result
        
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        return []
    except Exception as e:
        st.error(f"Processing error: {e}")
        return []


st.set_page_config(page_title="AI Department Analyzer", layout="wide")

st.title("AI-Powered Department & Application Analyzer")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    data = process_pdf(uploaded_file)

    if data:
        df = pd.DataFrame(data)

        cols = ["department", "application", "relationship", "business_context"]
        for c in cols:
            if c not in df.columns:
                df[c] = "Unknown"

        df = df.fillna("Unknown")

        # Step 1: Clean your data
        df["department"] = df["department"].str.strip().str[:30]
        df["application"] = df["application"].str.strip().str[:30]
        
        # Step 2: Remove duplicates
        df = df.drop_duplicates(subset=["department", "application"])
        
        # Step 3: Limit nodes (VERY IMPORTANT)
        df = df.head(30)

        st.dataframe(df, use_container_width=True)

        dept = st.selectbox("Department", df["department"].unique())
        st.dataframe(df[df["department"] == dept])

        st.bar_chart(df["department"].value_counts())
        st.bar_chart(df["relationship"].value_counts())

        # Step 4: Improve Graph Design
        G = nx.Graph()

        for _, r in df.iterrows():
            dept = r["department"]
            app = r["application"]
            G.add_node(dept, type="dept")
            G.add_node(app, type="app")
            G.add_edge(dept, app)

        pos = nx.spring_layout(G, k=0.8)

        dept_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "dept"]
        app_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "app"]

        plt.figure(figsize=(12, 8))

        nx.draw_networkx_nodes(G, pos, nodelist=dept_nodes, node_shape="s")
        nx.draw_networkx_nodes(G, pos, nodelist=app_nodes, node_shape="o")

        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=8)

        st.pyplot(plt)

    else:
        st.error("No data extracted")
