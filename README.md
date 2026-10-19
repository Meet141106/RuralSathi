# 🌾 RuralSathi — AI for Bharat's Farmers

An enterprise-grade, retrieval-augmented generation (RAG) assistant designed for Indian agriculture, rural advisory, weather intelligence, and government welfare schemes.

![RuralSathi Dashboard](https://img.shields.io/badge/RuralSathi-v1.0.0-22c55e?style=for-the-badge&logo=farm)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Hybrid%20RAG-green?style=for-the-badge)

---

## 📸 Interface Preview
```
+-----------------------------------------------------------------------------------------+
|  🌾 RuralSathi                   [ 🌾 Chat ]  [ 🌦️ Weather ]  [ 📊 Evaluation ]         |
+-----------------------------------------------------------------------------------------+
|  [Sidebar]                       Ask RuralSathi                                         |
|  - 75 Chunks | 8 Sources         Grounded AI Agricultural Assistant                     |
|  - Quick Question Chips          ------------------------------------------------------ |
|  - Mini Weather Card             🧑‍🌾 Farmer: What fertilizer should I use for wheat?    |
|  - About RuralSathi (v1.0.0)                                                            |
|                                  🌾 RuralSathi AI:                                      |
|                                  Apply top dressing of Urea (around 60 kg/ha nitrogen)  |
|                                  at Crown Root Initiation (20-25 DAS)...                |
|                                  [📚 Verified Sources Used: KCC Dataset, ICAR Wheat]    |
+-----------------------------------------------------------------------------------------+
```
*[Add screenshot here]*

---

## 🌟 Key Features

1. **🌾 Hybrid Dense-Sparse Retrieval**
   - Pairs **FAISS** (Inner Product cosine similarity on normalized dense embeddings) with **BM25Okapi** (exact agronomic terminology & chemical names).
   - Fused using **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF Score} = \sum_{m \in \{\text{Dense, Sparse}\}} \frac{1}{\text{rank}_m + 60}$$

2. **🎯 Cross-Encoder Reranking & Out-of-Domain Guardrails**
   - Reranks top 8 candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Rejects off-topic inquiries (e.g., sports, general trivia) and gracefully directs farmers to the national **Kisan Call Centre (1800-180-1551)**.

3. **🌦️ Real-Time Agro-Weather Intelligence**
   - 100% free Open-Meteo API integration covering 30 major Indian agricultural districts.
   - 5-day forecasts with automated heavy rainfall warnings (>15mm) and field action indicators (favorable spraying & irrigation schedules).

4. **📊 Comprehensive Evaluation Dashboard**
   - Built-in benchmark suite testing 25 real-world queries (`answerable`, `multi_chunk`, `off_topic`).
   - Interactive Plotly analytics for latency, category success rates, and reranker distribution.

---

## 🛠️ Technology Stack

| Layer | Component / Library | Purpose |
| :--- | :--- | :--- |
| **User Interface** | Streamlit + Custom Dark Glassmorphism CSS | Responsive dashboard & chat |
| **Dense Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional dense semantic vectors |
| **Vector Store** | FAISS (`IndexFlatIP`) | Fast dense vector index |
| **Sparse Retrieval** | `rank-bm25` (`BM25Okapi`) | Exact keyword and chemical matching |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-attention query-chunk scoring |
| **LLM Inference** | Groq API (`llama-3.3-70b-versatile`) | Factual, grounded answer generation |
| **Document Parsing** | PyMuPDF (`fitz`), Pandas, JSONL | PDF, TXT, and KCC Q&A ingestion |
| **Weather API** | Open-Meteo Forecast API | Real-time agro-meteorology |
| **Visualizations** | Plotly Express & Graph Objects | Dynamic evaluation charts |

---

## 📂 Datasets & Knowledge Base

- **Kisan Call Centre (KCC) Q&A Dataset**: Farmer queries on crop pests, sowing times, and fertilizer application.
- **ICAR Technical Bulletins**: Comprehensive disease management guides for Wheat, Rice, and Cotton.
- **Government Schemes**: Official guidelines for **PM-KISAN** (income support) and **PMFBY** (crop insurance).

---

## 🧠 How RAG Works in RuralSathi

1. **Ask**: The farmer submits an agricultural question (e.g., *"How to treat yellow rust in wheat?"*).
2. **Find**: RuralSathi fetches top candidates using FAISS + BM25, and reranks them with a Cross-Encoder to evaluate semantic relevance.
3. **Answer**: The LLM synthesizes a grounded answer citing the source document, or abstains if the query is unrelated to the knowledge base.

---

## 🚀 Setup & Installation

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/RuralSathi.git
cd RuralSathi

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and add your Groq API key:
```bash
cp .env.example .env
# Edit .env:
# GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Ingest Data & Build the Hybrid Index
```bash
# 1. Download and create datasets
python scripts/download_data.py

# 2. Chunk documents and build FAISS + BM25 index
python scripts/build_index.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📈 Evaluation Benchmark Results

Evaluated on 25 standardized agricultural and out-of-domain questions:

| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Answerable Accuracy** | **100% (15/15)** | > 90% |
| **Off-Topic Abstention** | **100% (5/5)** | 100% |
| **Multi-Chunk Reasoning** | **80% (4/5)** | > 75% |
| **Average Latency** | **0.51s** | < 2.0s |
| **Source Citation Rate** | **76.0%** | > 70% |

---

## 📚 References

1. **Vaswani, A., et al. (2017)**. *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS).
2. **Reimers, N., & Gurevych, I. (2019)**. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
3. **Lewis, P., et al. (2020)**. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
4. **Robertson, S., & Zaragoza, H. (2009)**. *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.

---

## ⚖️ License
Released under the MIT License. Developed for rural empowerment and academic AI research.
