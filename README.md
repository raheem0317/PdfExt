# AI Department & Application Analyzer

An AI-powered tool that extracts and visualizes relationships between organizational departments and software applications from PDF documents.

## 🚀 Features
- **PDF Text Extraction**: Uses `pdfplumber` for robust text recovery.
- **AI-Powered Analysis**: Leverages Groq (Llama 3) to identify entities and relationships in parallel.
- **Interactive Knowledge Graph**: Visualizes connections using `NetworkX` and `Matplotlib`.
- **Streamlit Dashboard**: Easy-to-use web interface for document processing.

## 🛠 Tech Stack
- **Frontend**: Streamlit
- **Processing**: Python, ThreadPoolExecutor
- **AI**: Groq SDK (Llama 3.3 70B)
- **Graphing**: NetworkX, Matplotlib

## 🏁 Getting Started

### Prerequisites
- Python 3.9+
- A Groq API Key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/raheem0317/PdfExt.git
   cd PdfExt
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_actual_key_here
   ```

### Running the App
```bash
streamlit run app.py
```

## 📊 Approach
The system uses a parallel processing pipeline:
1. **Chunking**: Splits PDF text into segments.
2. **Extraction**: AI identifies `(Department, Application, Relationship)` triplets.
3. **Synthesis**: Pandas deduplicates and cleans the data.
4. **Visualization**: NetworkX renders the bipartite graph.
