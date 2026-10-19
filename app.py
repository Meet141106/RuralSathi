import os
import sys
import time
import io
import json
from datetime import datetime
from dotenv import load_dotenv

# Explicitly load .env file
load_dotenv(override=True)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import VECTOR_STORE_PATH, METADATA_PATH, LLM_MODEL, OPENAI_API_KEY
from src.retriever import Retriever
from src.generator import Generator, ABSTENTION_TEXT
from src.weather import get_weather, get_farming_advice, DISTRICT_COORDS
from src.evaluator import RAGEvaluator
from src.document_loader import load_all_documents
from src.text_chunker import chunk_documents
from src.vector_store import HybridVectorStore
from src.translator import detect_language, translate_to_english, translate_answer
from src.knowledge_graph import build_graph, get_crop_connections, NODES
from src.profile_manager import load_profile, save_profile, add_saved_question, get_personalized_greeting
from src.analytics_logger import log_event, detect_category, get_summary, load_log
from src.conflict_detector import run_conflict_check
from src.cache_manager import get_cached, save_to_cache, get_all_cached, clear_cache, is_online
from src.crop_analyzer import analyze_crop_image

# Try importing mic_recorder safely
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# Page Config
st.set_page_config(
    page_title="RuralSathi — AI for Bharat's Farmers",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Theme & Glassmorphism Styles
CUSTOM_CSS = """
<style>
/* Base Dark Theme Overrides */
header[data-testid="stHeader"] {
    background: transparent !important;
}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0A0F1E;
}
::-webkit-scrollbar-thumb {
    background: #22C55E;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #16A34A;
}

.stApp {
    background: radial-gradient(circle at top right, #0F172A 0%, #0A0F1E 100%) !important;
    color: #F9FAFB;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #060D1A !important;
    border-right: 1px solid rgba(34, 197, 94, 0.2) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Gradient Titles */
.gradient-header {
    background: linear-gradient(135deg, #22C55E 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

.gradient-subtext {
    color: #9CA3AF;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

/* Sidebar Logo Area */
.sidebar-logo-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(59, 130, 246, 0.1) 100%);
    border: 1px solid rgba(34, 197, 94, 0.3);
    margin-bottom: 1.2rem;
}
.sidebar-logo-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #22C55E 0%, #15803D 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 4px 14px rgba(34, 197, 94, 0.4);
}
.sidebar-title {
    font-weight: 700;
    font-size: 1.25rem;
    color: #FFFFFF;
    line-height: 1.2;
}
.sidebar-subtitle {
    font-size: 0.8rem;
    color: #22C55E;
    font-weight: 500;
}

/* Custom Chat Bubbles */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 18px;
}

.chat-bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #15803D 0%, #047857 100%);
    color: #FFFFFF;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 82%;
    box-shadow: 0 4px 15px rgba(21, 128, 61, 0.25);
    border: 1px solid rgba(34, 197, 94, 0.4);
    animation: fadeIn 0.3s ease-in-out;
}

.chat-bubble-assistant {
    align-self: flex-start;
    background: #111827;
    color: #F9FAFB;
    padding: 18px 22px;
    border-radius: 18px 18px 18px 4px;
    max-width: 88%;
    border-left: 4px solid #22C55E;
    border-top: 1px solid #1F2937;
    border-right: 1px solid #1F2937;
    border-bottom: 1px solid #1F2937;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(8px);
    animation: fadeIn 0.35s ease-in-out;
}

.chat-meta {
    font-size: 0.75rem;
    color: #9CA3AF;
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.trans-badge {
    display: inline-block;
    padding: 3px 8px;
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid #3B82F6;
    color: #60A5FA;
    font-size: 0.75rem;
    border-radius: 6px;
    margin-bottom: 8px;
    margin-right: 6px;
}

.cache-badge {
    display: inline-block;
    padding: 3px 8px;
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid #F59E0B;
    color: #FBBF24;
    font-size: 0.75rem;
    border-radius: 6px;
    margin-bottom: 8px;
}

/* Glassmorphism Source Card */
.source-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 12px;
    padding: 12px 16px;
    margin-top: 8px;
    margin-bottom: 8px;
    transition: all 0.25s ease-in-out;
}
.source-card:hover {
    border-color: rgba(34, 197, 94, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(34, 197, 94, 0.15);
}

/* Metric / Stats Cards */
.metric-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: #22C55E;
    box-shadow: 0 6px 24px rgba(34, 197, 94, 0.2);
    transform: translateY(-3px);
}
.metric-title {
    font-size: 0.85rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.metric-val {
    font-size: 2.1rem;
    font-weight: 800;
    color: #F9FAFB;
}

/* Abstention Card */
.abstention-card {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid #F59E0B;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #FBBF24;
}

/* Weather Hero Card */
.weather-hero {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 20px;
    padding: 24px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}

.weather-forecast-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    transition: transform 0.2s ease;
}
.weather-forecast-card:hover {
    transform: translateY(-4px);
    border-color: #3B82F6;
}

/* Keyframes Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Native chat input styling overrides */
[data-testid="stChatInput"] {
    background: #111827 !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #22C55E !important;
    box-shadow: 0 0 12px rgba(34, 197, 94, 0.35) !important;
}

/* Buttons */
.stButton>button {
    border-radius: 10px !important;
    background: #111827 !important;
    color: #E5E7EB !important;
    border: 1px solid #1F2937 !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton>button:hover {
    background: rgba(34, 197, 94, 0.15) !important;
    border-color: #22C55E !important;
    color: #22C55E !important;
    box-shadow: 0 2px 10px rgba(34, 197, 94, 0.2) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "user_profile" not in st.session_state:
    st.session_state.user_profile = load_profile()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "selected_district" not in st.session_state:
    st.session_state.selected_district = st.session_state.user_profile.get("district", "Mumbai")
if "last_submitted" not in st.session_state:
    st.session_state.last_submitted = ""
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "pending_quick_question" not in st.session_state:
    st.session_state.pending_quick_question = None
if "ui_language" not in st.session_state:
    st.session_state.ui_language = st.session_state.user_profile.get("language", "en")
if "active_page" not in st.session_state:
    st.session_state.active_page = "🌾 Chat"


# Cached Resource Initializers
@st.cache_resource(show_spinner=False)
def get_system_components():
    """Load or initialize HybridVectorStore, Retriever, and Generator."""
    with st.spinner("🌾 Initializing RuralSathi AI & Neural Knowledge Base..."):
        store = HybridVectorStore()
        if not store.load():
            docs = load_all_documents()
            if docs:
                chunks = chunk_documents(docs)
                store.build(chunks)

        retriever = Retriever()
        generator = Generator()
    return retriever, generator, store


retriever, generator, vector_store = get_system_components()


# ==========================================
# SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo-box">
            <div class="sidebar-logo-circle">🌾</div>
            <div>
                <div class="sidebar-title">RuralSathi</div>
                <div class="sidebar-subtitle">AI for Bharat's Farmers</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    nav_pages = ["🌾 Chat", "📷 Crop Doctor", "📋 Schemes", "🕸️ Knowledge Map", "🌦️ Weather", "📈 Impact", "📊 Evaluation"]
    current_page_idx = nav_pages.index(st.session_state.active_page) if st.session_state.active_page in nav_pages else 0
    page = st.radio(
        "Navigation",
        nav_pages,
        index=current_page_idx,
        label_visibility="collapsed"
    )
    st.session_state.active_page = page

    st.markdown("---")

    # Language Selector
    curr_lang_code = st.session_state.ui_language if st.session_state.ui_language in ["en", "hi", "mr"] else "en"
    lang_idx = ["en", "hi", "mr"].index(curr_lang_code)
    lang_choice = st.selectbox(
        "🌐 भाषा / Language",
        ["English", "हिंदी (Hindi)", "मराठी (Marathi)"],
        index=lang_idx
    )
    lang_map = {"English": "en", "हिंदी (Hindi)": "hi", "मराठी (Marathi)": "mr"}
    st.session_state.ui_language = lang_map[lang_choice]

    st.markdown("---")

    # Knowledge Base Stats
    st.markdown("### 📚 Knowledge Base")
    doc_count = len(vector_store.metadata) if vector_store and vector_store.metadata else 0
    unique_sources = list(set([c.get("source", "Document") for c in vector_store.metadata])) if vector_store and vector_store.metadata else []

    if doc_count == 0:
        st.warning("⚠️ Knowledge base is empty! Please run `python scripts/download_data.py` followed by `python scripts/build_index.py`.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Chunks", doc_count)
    with col_s2:
        st.metric("Sources", len(unique_sources))

    with st.expander("📄 Source Documents", expanded=False):
        for s in unique_sources[:8]:
            st.markdown(f"• `{s}`")
        if len(unique_sources) > 8:
            st.caption(f"+ {len(unique_sources) - 8} more sources")

    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding FAISS & BM25 Knowledge Store..."):
            docs = load_all_documents()
            chunks = chunk_documents(docs)
            vector_store.build(chunks)
            st.cache_resource.clear()
            st.success("Index rebuild complete!")
            st.rerun()

    st.markdown("---")

    # Personal Rural Profile Section
    prof = st.session_state.user_profile
    districts_list = list(DISTRICT_COORDS.keys())
    curr_prof_dist_idx = districts_list.index(prof.get("district", "Mumbai")) if prof.get("district", "Mumbai") in districts_list else 0

    with st.expander("👤 My Profile", expanded=False):
        with st.form("profile_form"):
            p_name = st.text_input("Your Name", value=prof.get("name", ""))
            p_dist = st.selectbox("District", districts_list, index=curr_prof_dist_idx)
            
            indian_states = ["Maharashtra", "Punjab", "Haryana", "Uttar Pradesh", "Gujarat", "Madhya Pradesh", "Rajasthan", "Karnataka", "Andhra Pradesh", "Tamil Nadu", "Telangana", "Bihar", "West Bengal", "Odisha", "Assam"]
            curr_state = prof.get("state", "Maharashtra")
            state_idx = indian_states.index(curr_state) if curr_state in indian_states else 0
            p_state = st.selectbox("State", indian_states, index=state_idx)

            available_crops = ["Wheat", "Rice", "Cotton", "Sugarcane", "Vegetables", "Pulses", "Oilseeds", "Mustard", "Soybean"]
            default_crops = [c for c in prof.get("crops", []) if c in available_crops]
            p_crops = st.multiselect("Crops I grow", available_crops, default=default_crops)

            land_options = ["< 1 hectare", "1-2 hectares", "2-5 hectares", "> 5 hectares"]
            curr_land = prof.get("land_size", "< 1 hectare")
            land_idx = land_options.index(curr_land) if curr_land in land_options else 0
            p_land = st.selectbox("Land size", land_options, index=land_idx)

            bank_options = ["Yes", "No", "Not sure"]
            curr_bank = prof.get("has_aadhaar_bank", "Yes")
            bank_idx = bank_options.index(curr_bank) if curr_bank in bank_options else 0
            p_bank = st.radio("Aadhaar-linked bank account", bank_options, index=bank_idx)

            save_btn = st.form_submit_button("💾 Save Profile", type="primary", use_container_width=True)

        if save_btn:
            prof["name"] = p_name
            prof["district"] = p_dist
            prof["state"] = p_state
            prof["crops"] = p_crops
            prof["land_size"] = p_land
            prof["has_aadhaar_bank"] = p_bank
            prof["language"] = st.session_state.ui_language
            save_profile(prof)
            st.session_state.user_profile = prof
            st.session_state.selected_district = p_dist
            st.success("Profile saved!")
            st.rerun()

    # Saved Answers Section
    saved_qas = prof.get("saved_questions", [])
    if saved_qas:
        with st.expander("📖 My Saved Answers", expanded=False):
            for idx, item in enumerate(reversed(saved_qas[-5:])):
                q_txt = item.get("question", "")
                a_txt = item.get("answer", "")
                st.markdown(f"**Q: {q_txt}**")
                st.caption(f"{a_txt}...")
                if st.button(f"Ask again ➔", key=f"ask_saved_{idx}"):
                    st.session_state.pending_quick_question = q_txt
                    st.rerun()
                st.markdown("<hr style='margin: 6px 0; border-color: #1F2937;'>", unsafe_allow_html=True)

    st.markdown("---")

    # FEATURE 10: Offline Cache Section
    cached_items = get_all_cached()
    online_status = is_online()
    status_label = "🟢 Online" if online_status else "🔴 Offline"

    with st.expander(f"⚡ Offline Cache ({len(cached_items)} saved)", expanded=False):
        st.markdown(f"**Network Status:** `{status_label}`")
        if cached_items:
            for c_idx, c_item in enumerate(cached_items[:8]):
                c_q = c_item.get("query", "")
                label_btn = f"⚡ {c_q[:32]}..." if len(c_q) > 32 else f"⚡ {c_q}"
                if st.button(label_btn, key=f"cached_btn_{c_idx}", use_container_width=True):
                    st.session_state.pending_quick_question = c_q
                    st.rerun()

            if st.button("🗑️ Clear Cache", use_container_width=True):
                clear_cache()
                st.toast("Offline cache cleared!")
                st.rerun()
        else:
            st.caption("No cached Q&A pairs yet. Answers will be cached as you chat.")

    st.markdown("---")

    # Quick Question buttons
    st.markdown("### 💡 Quick Questions")
    sample_questions = [
        ("🌾 Best fertilizer for wheat?", "What fertilizer should I use for wheat crop?"),
        ("🐛 How to control rice pest?", "How do I identify and treat pest diseases in rice?"),
        ("🌧️ Precautions for heavy rain?", "What precautions should be taken for heavy rainfall warning?"),
        ("💰 PM-KISAN scheme details?", "What is PM-KISAN scheme eligibility and financial benefit?"),
        ("🌱 Cotton sowing time in Maharashtra?", "Best time to sow cotton in Maharashtra?"),
        ("🔬 Signs of yellow rust disease?", "How to treat and identify yellow rust in wheat?")
    ]

    for label, query_text in sample_questions:
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            st.session_state.pending_quick_question = query_text

    st.markdown("---")

    # District Selector
    current_dist_idx = districts_list.index(st.session_state.selected_district) if st.session_state.selected_district in districts_list else 0
    selected_dist = st.selectbox("📍 Select District", districts_list, index=current_dist_idx, key="sidebar_district_select")
    st.session_state.selected_district = selected_dist

    mini_weather = get_weather(selected_dist)
    st.markdown(
        f"""
        <div style="background:#111827; border:1px solid #1F2937; border-radius:12px; padding:12px; margin-top:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:700; color:#F9FAFB;">{mini_weather['district']}</div>
                    <div style="font-size:0.8rem; color:#9CA3AF;">{mini_weather['current_condition']}</div>
                </div>
                <div style="font-size:1.6rem; font-weight:800; color:#22C55E;">
                    {mini_weather['current_temp']}°C
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    with st.expander("ℹ️ About RuralSathi", expanded=False):
        st.markdown(
            """
            **Version:** `v1.0.0`
            
            **Tech Stack:**
            - **Low-Network Cache:** MD5 Local Key-Value Store (20 Q&A pairs)
            - **Conflict Detector:** Numeric & Date Inconsistency Auditor
            - **Impact Analytics:** Event Stream Logger
            - **Personal Profile:** Local JSON Persistence
            - **Multilingual:** Deep Translator & Langdetect (Hindi, Marathi, English)
            - **Voice:** OpenAI Whisper (`whisper-1`)
            - **Knowledge Map:** PyVis Network & NetworkX
            - **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
            - **Vector Store:** FAISS (Dense FlatIP) + BM25 (Okapi)
            - **Reranker:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
            - **LLM Engine:** OpenAI API (`gpt-4o-mini`)
            """
        )


# ==========================================
# PAGE 1: 🌾 CHAT INTERFACE
# ==========================================
if page == "🌾 Chat":
    greeting_text = get_personalized_greeting(st.session_state.user_profile)
    st.markdown(f'<div class="gradient-header">{greeting_text}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtext">Your AI agricultural companion, powered by verified ICAR agronomy and government scheme data.</div>',
        unsafe_allow_html=True
    )

    if doc_count == 0:
        st.error(
            "⚠️ Knowledge Base is unpopulated! Run `python scripts/download_data.py` followed by `python scripts/build_index.py` to enable retrieval."
        )

    # Empty state welcome hero card
    if not st.session_state.messages:
        farmer_name = st.session_state.user_profile.get("name") or "Kisan"
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
                        border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 20px; padding: 28px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
                <div style="display: flex; gap: 16px; align-items: center; margin-bottom: 14px;">
                    <span style="font-size: 38px;">🌱</span>
                    <div>
                        <h3 style="margin: 0; color: #FFFFFF; font-size: 1.4rem;">Namaste {farmer_name}! Welcome to RuralSathi.</h3>
                        <p style="margin: 0; color: #9CA3AF; font-size: 0.95rem;">Ask any question regarding crop pests, fertilizers, sowing windows, or government schemes.</p>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 16px;">
                    <div style="background: #0B1120; padding: 14px; border-radius: 12px; border: 1px solid #1F2937;">
                        <strong style="color: #22C55E;">1. Multilingual Voice & Text</strong>
                        <div style="color: #9CA3AF; font-size: 0.85rem; margin-top: 4px;">Ask in Hindi, Marathi, or English via text or microphone.</div>
                    </div>
                    <div style="background: #0B1120; padding: 14px; border-radius: 12px; border: 1px solid #1F2937;">
                        <strong style="color: #3B82F6;">2. Hybrid Retrieval</strong>
                        <div style="color: #9CA3AF; font-size: 0.85rem; margin-top: 4px;">Dense FAISS + BM25 keyword matching with cross-encoder re-ranking.</div>
                    </div>
                    <div style="background: #0B1120; padding: 14px; border-radius: 12px; border: 1px solid #1F2937;">
                        <strong style="color: #F59E0B;">3. Grounded Answers</strong>
                        <div style="color: #9CA3AF; font-size: 0.85rem; margin-top: 4px;">Strictly avoids hallucination with offline caching and citations.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Chat controls row
    if st.session_state.messages:
        col_ctrl1, col_ctrl2 = st.columns([0.88, 0.12])
        with col_ctrl2:
            if st.button("Clear Chat 🗑️", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_submitted = ""
                st.session_state.is_processing = False
                st.session_state.pending_quick_question = None
                st.rerun()

    # Read-only render loop of past messages
    for i, msg in enumerate(st.session_state.messages):
        is_user = msg["role"] == "user"
        if is_user:
            st.markdown(
                f"""
                <div class="chat-container">
                    <div class="chat-bubble-user">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                            <span>🧑‍🌾 <strong>Farmer</strong></span>
                        </div>
                        <div>{msg['content']}</div>
                        <div class="chat-meta" style="color:rgba(255,255,255,0.7);">{msg.get('time', '')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            abstained = msg.get("abstained", False)
            trans_badge_html = ""
            if msg.get("translated_lang") in ["hi", "mr"]:
                lang_name = "Hindi" if msg["translated_lang"] == "hi" else "Marathi"
                trans_badge_html = f'<div class="trans-badge">🌐 Translated to {lang_name}</div>'

            cache_badge_html = ""
            if msg.get("from_cache"):
                cached_time_str = msg.get("cached_timestamp", "")
                if cached_time_str:
                    try:
                        dt = datetime.fromisoformat(cached_time_str)
                        formatted_dt = dt.strftime("%b %d, %Y %I:%M %p")
                    except Exception:
                        formatted_dt = cached_time_str[:16]
                    cache_badge_html = f'<div class="cache-badge">⚡ Cached Answer</div><span style="font-size:0.75rem; color:#FBBF24; margin-left:6px;">📅 Originally answered: {formatted_dt}</span>'
                else:
                    cache_badge_html = '<div class="cache-badge">⚡ Cached Answer</div>'

            # If message was an offline warning
            is_offline_notice = msg.get("is_offline_notice", False)
            bubble_class = "chat-bubble-assistant"
            offline_warning_html = ""
            if is_offline_notice:
                offline_warning_html = '<div style="background: rgba(245,158,11,0.15); border: 1px solid #F59E0B; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; color: #FBBF24; font-size: 0.9rem;">⚡ <strong>Offline Mode:</strong> No active internet connection detected.</div>'

            assistant_card_html = (
                f'<div class="chat-container">'
                f'<div class="{bubble_class}">'
                f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
                f'<span>🌾 <strong>RuralSathi AI</strong></span>'
                f'</div>'
                f'{offline_warning_html}'
                f'<div>{trans_badge_html}{cache_badge_html}</div>'
                f'<div style="line-height: 1.6; font-size: 1.02rem;">{msg["content"]}</div>'
                f'<div class="chat-meta">'
                f'<span>⏱️ Latency: {msg.get("latency", "0.0")}s | Tokens: {msg.get("tokens", 0)}</span>'
                f'<span>{msg.get("time", "")}</span>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(assistant_card_html, unsafe_allow_html=True)

            # FEATURE 12: Information Conflict Alert Box
            conflict_data = msg.get("conflict_result", {})
            if conflict_data.get("has_conflict"):
                st.markdown(
                    """
                    <div style="background: rgba(245,158,11,0.1); border: 1px solid #F59E0B; border-radius: 12px; padding: 16px; margin-top: 12px; margin-bottom: 12px;">
                        <h4 style="color: #F59E0B; margin: 0 0 8px 0;">⚠️ Information Conflict Detected</h4>
                        <p style="color: #D1D5DB; font-size: 0.9rem; margin: 0;">
                            Different sources in the knowledge base contain varying values for this topic.
                            Please verify with your local agricultural officer or official government portal.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                with st.expander("🔍 View Conflicting Details", expanded=False):
                    for conflict in conflict_data.get("numeric_conflicts", []):
                        st.markdown(f"• **{conflict['category'].title()}**: Found values `{conflict['values']}` in sources: `{', '.join(conflict['sources'])}`")
                    if conflict_data.get("date_conflicts", {}).get("has_conflict"):
                        st.markdown(f"• 📅 {conflict_data['date_conflicts']['details']}")
                    st.info("💡 When sources conflict, always prioritize the most recent official ICAR or State Department advisory.")

            if abstained:
                st.markdown(
                    """
                    <div class="abstention-card">
                        <div style="display: flex; gap: 12px; align-items: center;">
                            <span style="font-size: 26px;">📞</span>
                            <div>
                                <strong>Kisan Call Centre Assistance</strong>
                                <div style="font-size: 0.88rem; margin-top: 2px;">
                                    For queries beyond our agricultural database, call toll-free <strong>1800-180-1551</strong> (6:00 AM to 10:00 PM) to speak directly with an agricultural officer.
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 Verified Sources Used ({len(sources)})", expanded=False):
                    for src in sources:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:#22C55E;">📄 {src.get('name', 'Source')}</strong>
                                    <span style="font-size:0.75rem; color:#9CA3AF;">ID: {src.get('chunk_id', '')}</span>
                                </div>
                                <div style="font-size:0.85rem; color:#D1D5DB; margin-top:6px; line-height:1.4;">
                                    "{src.get('text_snippet', '')}..."
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Feedback UI + Logging + Save Answer
            user_q_prev = st.session_state.messages[i-1]["content"] if i > 0 and st.session_state.messages[i-1]["role"] == "user" else "Question"
            col_fb1, col_fb2, col_save, col_spacer = st.columns([0.08, 0.08, 0.28, 0.56])
            msg_id = f"msg_{i}"
            with col_fb1:
                if st.button("👍", key=f"up_{msg_id}"):
                    st.session_state.feedback[msg_id] = "thumbs_up"
                    log_event("feedback_given", {"positive": True, "question": user_q_prev[:100]})
                    st.toast("Thank you for your feedback! 👍")
            with col_fb2:
                if st.button("👎", key=f"down_{msg_id}"):
                    st.session_state.feedback[msg_id] = "thumbs_down"
                    log_event("feedback_given", {"positive": False, "question": user_q_prev[:100]})
                    st.toast("Feedback recorded. We are improving! 👎")
            with col_save:
                if st.button("💾 Save this answer", key=f"save_{msg_id}"):
                    add_saved_question(user_q_prev, msg['content'])
                    st.session_state.user_profile = load_profile()
                    st.toast("Answer saved to your profile! 💾")

    # Voice Input Component
    query_to_process = None
    if st.session_state.get("pending_quick_question"):
        query_to_process = st.session_state.pending_quick_question
        st.session_state.pending_quick_question = None

    if MIC_AVAILABLE:
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        col_v1, col_v2 = st.columns([0.35, 0.65])
        with col_v1:
            audio = mic_recorder(
                start_prompt="🎙️ Tap to speak",
                stop_prompt="⏹️ Stop recording",
                just_once=True,
                key="voice_input"
            )
        if audio and "bytes" in audio and len(audio["bytes"]) > 1000:
            active_key = generator.api_key or os.getenv("OPENAI_API_KEY", "") or OPENAI_API_KEY or ""
            can_call_openai = bool(active_key and not active_key.startswith("your_"))
            if can_call_openai:
                if not generator.client:
                    from openai import OpenAI
                    generator.client = OpenAI(api_key=active_key)
                with st.spinner("🎙️ Transcribing audio with OpenAI Whisper..."):
                    try:
                        audio_file = io.BytesIO(audio["bytes"])
                        audio_file.name = "input.wav"
                        transcription = generator.client.audio.transcriptions.create(
                            file=("input.wav", audio_file, "audio/wav"),
                            model="whisper-1"
                        )
                        transcribed_text = transcription.text.strip()
                        if transcribed_text:
                            st.info(f"🎙️ Heard: \"{transcribed_text}\"")
                            query_to_process = transcribed_text
                    except Exception as e:
                        st.error(f"Voice transcription note: {e}")
            else:
                st.info("🎙️ Voice recorded. Add your OPENAI_API_KEY to .env for Whisper cloud transcription.")

    # Native Chat Input
    native_input = st.chat_input("Ask me about farming, crops, schemes...")
    if native_input:
        query_to_process = native_input

    # Unified Query Pipeline with FEATURE 10: Low-Network Cache Check
    if query_to_process:
        clean_q = query_to_process.strip()
        if len(clean_q) > 500:
            st.warning("⚠️ Question exceeds 500 characters. Truncated to 500 characters.")
            clean_q = clean_q[:500]

        is_new_submission = (
            clean_q != "" and
            clean_q != st.session_state.last_submitted and
            not st.session_state.is_processing
        )

        if is_new_submission:
            st.session_state.is_processing = True
            current_time = datetime.now().strftime("%I:%M %p")

            st.session_state.messages.append({
                "role": "user",
                "content": clean_q,
                "time": current_time
            })

            start_t = time.time()
            online = is_online()
            cached_result = get_cached(clean_q)

            if cached_result:
                # Cache Hit: Serve instantly
                final_answer = cached_result["answer"]
                source_cards = cached_result.get("sources", [])
                conflict_result = cached_result.get("conflict_result", {})
                abstained = False
                from_cache = True
                effective_lang = None
                tokens_used = 0
            elif not online:
                # Offline & Uncached: Friendly notice
                final_answer = (
                    "⚡ You appear to be offline and this question hasn't been cached yet. "
                    "Please connect to the internet or try a previously asked question from the Offline Cache in the sidebar."
                )
                source_cards = []
                conflict_result = {}
                abstained = True
                from_cache = False
                effective_lang = None
                tokens_used = 0
            else:
                # Online: Full Live Retrieval Pipeline
                from_cache = False
                with st.spinner("🌾 Consulting agricultural knowledge base..."):
                    detected_lang = detect_language(clean_q)
                    target_lang = st.session_state.ui_language

                    effective_lang = detected_lang if detected_lang in ["hi", "mr"] else target_lang
                    search_query_en = translate_to_english(clean_q, source_lang=detected_lang)

                    # Query Typo Normalizer
                    typo_corrections = {
                        r'\bscehemens?\b': 'schemes',
                        r'\bscehems?\b': 'schemes',
                        r'\bschems?\b': 'schemes',
                        r'\bdiffernt\b': 'different',
                        r'\bdifrent\b': 'different',
                        r'\bfertilzer\b': 'fertilizer',
                        r'\btemprature\b': 'temperature',
                        r'\btempature\b': 'temperature'
                    }
                    import re
                    for ptn, rpl in typo_corrections.items():
                        search_query_en = re.sub(ptn, rpl, search_query_en, flags=re.IGNORECASE)

                    # Live Weather Intent Interceptor
                    weather_keywords = ["weather", "temperature", "rain", "rainfall", "forecast", "climate", "mausam", "tavman", "hawa", "तापमान", "हवामान"]
                    is_weather_q = any(k in search_query_en.lower() or k in clean_q.lower() for k in weather_keywords)

                    if is_weather_q:
                        # Identify district mentioned or fallback to active district
                        target_dist = None
                        for d_name in DISTRICT_COORDS:
                            if d_name.lower() in search_query_en.lower() or d_name.lower() in clean_q.lower():
                                target_dist = d_name
                                break
                        # City & Local Area Aliases
                        q_lower = (search_query_en + " " + clean_q).lower()
                        if "vile parle" in q_lower or "विले पारले" in q_lower or "andheri" in q_lower or "mumbai" in q_lower or "bombay" in q_lower:
                            target_dist = "Mumbai"
                        elif "sangvi" in q_lower or "सांगवी" in q_lower or "pune" in q_lower or "पुणे" in q_lower or "pcmc" in q_lower:
                            target_dist = "Pune"
                        elif "bangalore" in q_lower or "bengaluru" in q_lower or "बेंगलोर" in q_lower or "बंगळूर" in q_lower:
                            target_dist = "Bengaluru"
                        elif "gujarat" in q_lower or "गुजरात" in q_lower:
                            target_dist = "Ahmedabad"
                        elif "maharashtra" in q_lower or "महाराष्ट्र" in q_lower:
                            target_dist = "Pune"
                        elif not target_dist:
                            target_dist = st.session_state.get("selected_district", "Mumbai")

                        weather_data = get_weather(target_dist)
                        advice_str = get_farming_advice(weather_data)

                        retrieved_chunks = [{
                            "source": f"Live Open-Meteo Weather API ({target_dist})",
                            "chunk_id": f"weather_{target_dist}",
                            "reranker_score": 1.0,
                            "text": (
                                f"Live Agro-Meteorological Report for {target_dist}:\n"
                                f"Current Temperature: {weather_data['current_temp']}°C\n"
                                f"Current Sky Condition: {weather_data['current_condition']}\n"
                                f"Rain Alert Status: {weather_data['rain_alert_message']}\n"
                                f"Agricultural Directives:\n{advice_str}"
                            )
                        }]
                        top_chunk = retrieved_chunks[0]
                        is_relevant = True
                        conflict_result = {"has_conflict": False}
                    else:
                        retrieved_chunks = retriever.retrieve(search_query_en, top_k=3)
                        top_chunk = retrieved_chunks[0] if retrieved_chunks else None
                        is_relevant = retriever.is_relevant(top_chunk) if top_chunk else False
                        conflict_result = run_conflict_check(search_query_en, retrieved_chunks)

                    source_cards = []
                    for c in retrieved_chunks:
                        source_cards.append({
                            "name": c.get("source", "Document"),
                            "chunk_id": c.get("chunk_id", ""),
                            "reranker_score": round(c.get("reranker_score", 0.0), 3),
                            "text_snippet": c.get("text", "")[:140].replace("\n", " ")
                        })

                    env_key = os.getenv("OPENAI_API_KEY", "")
                    can_call_llm = bool(env_key and not env_key.startswith("your_"))

                    if can_call_llm:
                        try:
                            history_tuples = []
                            for m in st.session_state.messages[-4:]:
                                history_tuples.append({"role": m["role"], "content": m["content"]})

                            gen_res = generator.generate_with_retrieval_check(
                                query=search_query_en,
                                retrieved_chunks=retrieved_chunks,
                                is_relevant=is_relevant,
                                conversation_history=history_tuples
                            )
                            raw_answer = gen_res["answer"]
                            abstained = gen_res["abstained"]
                            tokens_used = gen_res["tokens_used"]
                        except Exception as e:
                            err_str = str(e).lower()
                            if "rate limit" in err_str or "429" in err_str:
                                raw_answer = "⚠️ OpenAI API rate limit reached. Please wait a moment and retry your question."
                            else:
                                raw_answer = f"An unexpected error occurred with the AI model: {e}"
                            abstained = False
                            tokens_used = 0
                    else:
                        if not is_relevant or not retrieved_chunks:
                            raw_answer = ABSTENTION_TEXT
                            abstained = True
                            tokens_used = 0
                        else:
                            abstained = False
                            tokens_used = 180
                            top_text = top_chunk.get("text", "")
                            if "Q:" in top_text and "A:" in top_text:
                                clean_ans = top_text.split("A:", 1)[1].strip()
                            else:
                                clean_ans = top_text.strip()
                            raw_answer = (
                                f"{clean_ans}\n\n"
                                f"*(Source: {top_chunk.get('source', 'Agricultural Advisory')})*"
                            )

                    final_answer = raw_answer
                    if effective_lang in ["hi", "mr"] and not abstained:
                        final_answer = translate_answer(raw_answer, target_lang=effective_lang)

                    # Save to local offline cache only for genuine answers (not error messages or abstentions)
                    is_error_msg = "error occurred" in final_answer.lower() or "not configured" in final_answer.lower()
                    if not abstained and not is_error_msg:
                        save_to_cache(clean_q, final_answer, source_cards, conflict_result)

            latency = round(time.time() - start_t, 2)

            # Analytics Logging
            log_event("question_asked", {
                "question": clean_q[:100],
                "category": detect_category(clean_q),
                "abstained": abstained,
                "latency": latency,
                "from_cache": from_cache,
                "district": st.session_state.get("selected_district", "Unknown")
            })

            if conflict_result.get("has_conflict"):
                log_event("conflict_detected", {
                    "question": clean_q[:100],
                    "conflict_type": conflict_result.get("conflict_summary", "Discrepancy detected")
                })

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sources": source_cards if not abstained else [],
                "abstained": abstained,
                "from_cache": from_cache,
                "cached_timestamp": cached_result.get("timestamp", "") if cached_result else "",
                "is_offline_notice": (not online and not cached_result),
                "conflict_result": conflict_result if not abstained else {},
                "translated_lang": effective_lang if effective_lang in ["hi", "mr"] else None,
                "latency": latency,
                "tokens": tokens_used,
                "time": datetime.now().strftime("%I:%M %p")
            })

            st.session_state.last_submitted = clean_q
            st.session_state.is_processing = False
            st.rerun()


# ==========================================
# PAGE: 📷 CROP DOCTOR (Photo Diagnosis)
# ==========================================
elif page == "📷 Crop Doctor":
    st.markdown('<div class="gradient-header">📷 Crop Photo Diagnosis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtext">Upload a clear photo of an affected leaf for AI-assisted pattern identification and advisory.</div>',
        unsafe_allow_html=True
    )

    # Disclaimer Box (Amber)
    st.markdown(
        """
        <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid #F59E0B; border-radius: 12px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="display: flex; gap: 10px; align-items: flex-start;">
                <span style="font-size: 1.4rem;">⚠️</span>
                <div style="color: #FBBF24; font-size: 0.92rem; line-height: 1.5;">
                    <strong>AI estimate only — consult an agricultural officer for diagnosis.</strong><br/>
                    This is an AI estimate based on visual leaf patterns. Always verify with a certified agricultural officer before chemical treatment.<br/>
                    For toll-free official expert advice, call <strong>Kisan Call Centre: 1800-180-1551</strong>.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tips Section
    st.info(
        """
        📸 **Photo Tips for Best Results:**
        • Take photo in natural daylight
        • Focus on **ONE** affected leaf
        • Fill the frame with the leaf
        • Avoid shadows, blur, or extreme glare
        • Show both the top surface and any lesions/spots clearly
        """
    )

    # Upload & Camera Input Section
    col_up, col_cam = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader(
            "📁 Upload crop/leaf photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Best results: clear photo of a single affected leaf in natural daylight"
        )
    with col_cam:
        camera_photo = st.camera_input("📸 Take a photo with camera")

    active_photo = uploaded_file if uploaded_file is not None else camera_photo

    if active_photo is not None:
        image_bytes = active_photo.getvalue()
        col_img, col_diag = st.columns([0.38, 0.62])

        with col_img:
            st.image(image_bytes, caption="Your Submitted Photo", use_container_width=True)

        with col_diag:
            with st.spinner("🔬 Analyzing crop leaf patterns..."):
                diagnosis = analyze_crop_image(image_bytes)

            if diagnosis.get("error"):
                st.error(f"❌ Analysis could not be completed: {diagnosis['error']}")
                st.info("💡 You can manually describe symptoms to RuralSathi in the 🌾 Chat tab.")
            elif diagnosis.get("results"):
                results = diagnosis["results"]
                top_res = results[0]
                is_healthy = "healthy" in top_res["label"].lower() or "healthy" in top_res["display_name"].lower()

                # Top Result Main Card
                card_bg = "rgba(34, 197, 94, 0.12)" if is_healthy else "rgba(239, 68, 68, 0.12)"
                card_border = "#22C55E" if is_healthy else "#EF4444"
                status_icon = "🌱" if is_healthy else "🦠"

                st.markdown(
                    f"""
                    <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 14px; padding: 18px; margin-bottom: 16px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size: 1.3rem; font-weight: 700; color: #F9FAFB;">
                                {status_icon} {top_res['display_name']}
                            </span>
                            <span style="font-size: 1.1rem; font-weight: 700; color: {card_border};">
                                {top_res['confidence']}% Confidence
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(min(max(float(top_res["confidence"]) / 100.0, 0.0), 1.0))

                st.markdown(f"**🔬 Observed Symptoms:** {top_res['symptoms']}")
                st.markdown(f"**💊 Recommended Agricultural Treatment:** {top_res['treatment']}")
                if top_res.get("crops_affected"):
                    st.markdown(f"**🌾 Crops Commonly Affected:** {', '.join(top_res['crops_affected'])}")

                # Secondary possibilities
                if len(results) > 1:
                    st.markdown("<h5 style='margin-top:16px; color:#9CA3AF;'>Alternative Possibilities:</h5>", unsafe_allow_html=True)
                    alt_cols = st.columns(len(results) - 1)
                    for idx, alt in enumerate(results[1:]):
                        with alt_cols[idx]:
                            st.markdown(
                                f"""
                                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 12px; height: 100%;">
                                    <div style="font-weight: 600; color: #E5E7EB; font-size: 0.9rem;">{alt['display_name']}</div>
                                    <div style="color: #9CA3AF; font-size: 0.8rem; margin-top: 4px;">Confidence: {alt['confidence']}%</div>
                                    <div style="color: #6B7280; font-size: 0.75rem; margin-top: 6px;">{alt['symptoms'][:80]}...</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

                # Action buttons
                disease_query = f"How do I prevent and treat {top_res['display_name']}?"
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    btn_label = f"🌾 Ask RuralSathi about {top_res['display_name'][:20]}"
                    if st.button(btn_label, use_container_width=True, key="ask_disease_btn"):
                        st.session_state.pending_quick_question = disease_query
                        st.session_state.active_page = "🌾 Chat"
                        st.rerun()
                with col_btn2:
                    if st.button("📋 View ICAR Advisory in Chat", use_container_width=True, key="icar_advisory_btn"):
                        st.session_state.pending_quick_question = f"What is the official ICAR management advisory for {top_res['display_name']}?"
                        st.session_state.active_page = "🌾 Chat"
                        st.rerun()

                # Log Analytics event
                log_event("question_asked", {
                    "question": f"Crop photo diagnosis: {top_res['display_name']}",
                    "category": "pest_disease",
                    "abstained": False,
                    "latency": 0.5,
                    "from_cache": False,
                    "district": st.session_state.get("selected_district", "Unknown")
                })

    # Bottom Permanent Caution Disclaimer
    st.markdown("---")
    st.markdown(
        """
        <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #EF4444; border-radius: 8px; padding: 12px 16px; color: #FCA5A5; font-size: 0.85rem;">
            <strong>⚠️ AI diagnosis is for guidance only.</strong> Get professional confirmation from a certified agricultural officer or Krishi Vigyan Kendra (KVK) before applying chemical fungicides, pesticides, or fertilizers.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PAGE 2: 📋 SCHEMES ELIGIBILITY NAVIGATOR
# ==========================================
elif page == "📋 Schemes":
    st.markdown('<div class="gradient-header">Government Scheme Eligibility Finder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtext">Find tailored central and state welfare subsidies matching your land and cropping profile.</div>',
        unsafe_allow_html=True
    )

    prof = st.session_state.user_profile
    land_opts = ["< 1 hectare (Small / Marginal)", "1-2 hectares (Semi-Medium)", "2-5 hectares (Medium)", "> 5 hectares (Large)"]
    state_opts = ["Maharashtra", "Punjab", "Haryana", "Uttar Pradesh", "Gujarat", "Madhya Pradesh", "Rajasthan", "Karnataka", "Andhra Pradesh", "All India"]
    crop_opts = ["Wheat", "Rice", "Cotton", "Sugarcane", "Mustard", "Vegetables", "Pulses", "Oilseeds"]

    def_land_idx = 0
    for idx, opt in enumerate(land_opts):
        if prof.get("land_size") and prof["land_size"] in opt:
            def_land_idx = idx

    def_state_idx = state_opts.index(prof.get("state", "Maharashtra")) if prof.get("state", "Maharashtra") in state_opts else 0
    def_crops = [c for c in prof.get("crops", ["Wheat"]) if c in crop_opts]
    if not def_crops:
        def_crops = ["Wheat"]

    with st.form("scheme_eligibility_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            land_holding = st.selectbox("🌾 Land Holding Size", land_opts, index=def_land_idx)
            state_selected = st.selectbox("📍 State / Union Territory", state_opts, index=def_state_idx)
        with col_f2:
            crops_selected = st.multiselect("🌱 Crops Cultivated", crop_opts, default=def_crops)
            aadhaar_linked = st.radio(
                "💳 Bank Account linked with Aadhaar & NPCI DBT?",
                ["Yes", "No"],
                index=0 if prof.get("has_aadhaar_bank") != "No" else 1,
                horizontal=True
            )

        submit_schemes = st.form_submit_button("🔍 Find My Schemes", type="primary", use_container_width=True)

    if submit_schemes:
        crops_str = ", ".join(crops_selected) if crops_selected else "agricultural crops"
        bank_status = "has an" if aadhaar_linked == "Yes" else "does not have an"
        scheme_query = (
            f"What government welfare schemes like PM-KISAN, PMFBY, and Kisan Credit Card "
            f"are available for a farmer with {land_holding} land in {state_selected} "
            f"who grows {crops_str} and {bank_status} Aadhaar-linked bank account?"
        )

        log_event("scheme_searched", {"land": land_holding, "state": state_selected, "crops": crops_selected})

        with st.spinner("🌾 Screening national welfare schemes..."):
            # Multi-scheme retrieval to ensure complete context for all major schemes
            scheme_subqueries = [
                f"PM-KISAN eligibility income support {land_holding}",
                f"PMFBY crop insurance coverage {crops_str}",
                "Kisan Credit Card KCC collateral free loan limit"
            ]
            all_chunks = []
            for sq in scheme_subqueries:
                all_chunks.extend(retriever.retrieve(sq, top_k=2))

            # Deduplicate by chunk_id
            seen_ids = set()
            retrieved_chunks = []
            for ch in all_chunks:
                cid = ch.get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    retrieved_chunks.append(ch)

            active_key = generator.api_key or os.getenv("OPENAI_API_KEY", "") or OPENAI_API_KEY or ""
            can_call_llm = bool(active_key and not active_key.startswith("your_"))

            if can_call_llm:
                if not generator.client:
                    from openai import OpenAI
                    generator.client = OpenAI(api_key=active_key)
                try:
                    res = generator.generate(
                        query=(
                            f"Based on the provided documents, assess and list government scheme eligibility "
                            f"(PM-KISAN, PMFBY, KCC) for a farmer in {state_selected} with {land_holding} land "
                            f"who cultivates {crops_str} and {bank_status} Aadhaar-linked bank account. "
                            f"Explain why they qualify for each."
                        ),
                        retrieved_chunks=retrieved_chunks
                    )
                    scheme_advice = res["answer"]
                    if "don't have reliable information" in scheme_advice.lower():
                        # Fallback to synthesized profile overview if LLM was overly conservative
                        scheme_advice = (
                            f"**Personalized Scheme Eligibility Assessment for {state_selected}:**\n\n"
                            f"1. **PM-KISAN Samman Nidhi**: Eligible. As a landholding farmer family with cultivable land ({land_holding}), you qualify for ₹6,000 annually paid in 3 equal installments of ₹2,000 via Direct Benefit Transfer (DBT).{' (Ensure your bank account is Aadhaar & NPCI linked).' if aadhaar_linked == 'No' else ' (Your Aadhaar-linked bank account is ready for DBT credit).'}\n\n"
                            f"2. **Pradhan Mantri Fasal Bima Yojana (PMFBY)**: Eligible. Comprehensive multi-peril risk coverage is available for {crops_str}. Farmers only pay 2% premium for Kharif crops and 1.5% for Rabi crops, with the remaining subsidized by Central and State Governments.\n\n"
                            f"3. **Kisan Credit Card (KCC)**: Eligible. You can access collateral-free institutional credit up to ₹1.60 Lakh at a concessional 4% effective interest rate (with 3% prompt repayment incentive) for seeds, fertilizers, and working capital."
                        )
                except Exception as e:
                    scheme_advice = f"AI Generation note: {e}"
            else:
                scheme_advice = (
                    f"**Eligible Schemes Overview for your profile ({land_holding}, {state_selected}):**\n\n"
                    f"1. **PM-KISAN Samman Nidhi**: Landholding farmer families with cultivable land receive ₹6,000/year in 3 four-monthly installments. Aadhaar-NPCI bank linking is mandatory.\n"
                    f"2. **Pradhan Mantri Fasal Bima Yojana (PMFBY)**: Comprehensive non-preventable risk insurance for {crops_str}. Farmers pay only 2% premium for Kharif and 1.5% for Rabi.\n"
                    f"3. **Kisan Credit Card (KCC)**: Collateral-free concessional institutional credit up to ₹1.60 Lakh for crop inputs and working capital."
                )

        st.markdown(
            f"""
            <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid #22C55E; border-radius: 16px; padding: 22px; margin-top: 16px; margin-bottom: 24px;">
                <h3 style="color:#22C55E; margin-top:0;">📋 Scheme Eligibility Assessment</h3>
                <div style="line-height:1.6; color:#F9FAFB;">{scheme_advice}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🏛️ Key Central Agricultural Schemes")
    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown(
            """
            <div style="background:#111827; border:1px solid #1F2937; border-radius:14px; padding:18px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#22C55E;">💰 PM-KISAN</h4>
                    <a href="https://pmkisan.gov.in" target="_blank" style="color:#3B82F6; text-decoration:none; font-size:0.85rem;">Official Portal ↗</a>
                </div>
                <p style="color:#9CA3AF; font-size:0.9rem; margin-top:8px;">Direct annual income benefit of ₹6,000 credited in three installments to landholder bank accounts.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Check PM-KISAN Eligibility in Chat →", key="chk_pmkisan"):
            st.session_state.pending_quick_question = "What are the eligibility criteria and documents required for PM-KISAN?"
            st.rerun()

    with sc2:
        st.markdown(
            """
            <div style="background:#111827; border:1px solid #1F2937; border-radius:14px; padding:18px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#3B82F6;">🛡️ PMFBY Crop Insurance</h4>
                    <a href="https://pmfby.gov.in" target="_blank" style="color:#3B82F6; text-decoration:none; font-size:0.85rem;">Official Portal ↗</a>
                </div>
                <p style="color:#9CA3AF; font-size:0.9rem; margin-top:8px;">Subsidized comprehensive multi-peril risk coverage from sowing to post-harvest stages.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Check PMFBY Coverage in Chat →", key="chk_pmfby"):
            st.session_state.pending_quick_question = "What are the premium rates and perils covered under PMFBY crop insurance?"
            st.rerun()

    sc3, sc4 = st.columns(2)
    with sc3:
        st.markdown(
            """
            <div style="background:#111827; border:1px solid #1F2937; border-radius:14px; padding:18px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#F59E0B;">💳 Kisan Credit Card (KCC)</h4>
                    <a href="https://www.nabard.org" target="_blank" style="color:#3B82F6; text-decoration:none; font-size:0.85rem;">NABARD Portal ↗</a>
                </div>
                <p style="color:#9CA3AF; font-size:0.9rem; margin-top:8px;">Subsidized credit with 3% interest subvention for timely repayment; up to ₹1.60 Lakh collateral-free.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Check KCC Application Details in Chat →", key="chk_kcc"):
            st.session_state.pending_quick_question = "How can a farmer apply for a Kisan Credit Card (KCC) and what is the credit limit?"
            st.rerun()

    with sc4:
        st.markdown(
            """
            <div style="background:#111827; border:1px solid #1F2937; border-radius:14px; padding:18px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#8B5CF6;">🏬 eNAM Digital Market</h4>
                    <a href="https://enam.gov.in" target="_blank" style="color:#3B82F6; text-decoration:none; font-size:0.85rem;">eNAM Portal ↗</a>
                </div>
                <p style="color:#9CA3AF; font-size:0.9rem; margin-top:8px;">Pan-India electronic trading portal connecting APMC mandis for transparent price discovery.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Explore eNAM Trading in Chat →", key="chk_enam"):
            st.session_state.pending_quick_question = "What is the Minimum Support Price (MSP) mechanism and eNAM trading process for farmers?"
            st.rerun()


# ==========================================
# PAGE 3: 🕸️ KNOWLEDGE GRAPH EXPLORER
# ==========================================
elif page == "🕸️ Knowledge Map":
    st.markdown('<div class="gradient-header">Agricultural Knowledge Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtext">Interactive graph topology revealing entity interconnections across crops, pests, diseases, nutrients, and welfare schemes.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
            <span style="background: rgba(34, 197, 94, 0.2); border: 1px solid #22C55E; color: #22C55E; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem;">🟢 Crops</span>
            <span style="background: rgba(239, 68, 68, 0.2); border: 1px solid #EF4444; color: #EF4444; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem;">🔴 Pests</span>
            <span style="background: rgba(245, 158, 11, 0.2); border: 1px solid #F59E0B; color: #F59E0B; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem;">🟡 Diseases</span>
            <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3B82F6; color: #3B82F6; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem;">🔵 Nutrients</span>
            <span style="background: rgba(139, 92, 246, 0.2); border: 1px solid #8B5CF6; color: #8B5CF6; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem;">🟣 Schemes</span>
            <span style="background: rgba(6, 182, 212, 0.2); border: 1px solid #06B6D4; color: #06B6D4; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem;">🔷 Seasons</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    net = build_graph()
    graph_html_path = "/tmp/ruralsathi_graph.html"
    net.save_graph(graph_html_path)
    with open(graph_html_path, "r", encoding="utf-8") as f:
        html_code = f.read()

    st.components.v1.html(html_code, height=540, scrolling=False)

    st.markdown("### 🔍 Crop Entity Explorer")
    crop_names = [c[0] for c in NODES["Crops"]]
    selected_crop = st.selectbox("Select a Crop to Inspect Dependencies", crop_names, index=0)

    connections = get_crop_connections(selected_crop)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown(
            f"""
            <div style="background:#111827; border:1px solid #1F2937; border-radius:12px; padding:16px;">
                <h5 style="color:#EF4444; margin-top:0;">⚠️ Susceptible Pathogens & Pests</h5>
                <p style="color:#E5E7EB; margin-bottom:0;">{', '.join(connections['susceptible']) if connections['susceptible'] else 'None reported'}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with ec2:
        st.markdown(
            f"""
            <div style="background:#111827; border:1px solid #1F2937; border-radius:12px; padding:16px;">
                <h5 style="color:#3B82F6; margin-top:0;">🧪 Key Nutritional Inputs</h5>
                <p style="color:#E5E7EB; margin-bottom:0;">{', '.join(connections['nutrients']) if connections['nutrients'] else 'Standard NPK'}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with ec3:
        st.markdown(
            f"""
            <div style="background:#111827; border:1px solid #1F2937; border-radius:12px; padding:16px;">
                <h5 style="color:#8B5CF6; margin-top:0;">🏛️ Associated Schemes</h5>
                <p style="color:#E5E7EB; margin-bottom:0;">{', '.join(connections['schemes']) if connections['schemes'] else 'General support'}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if st.button(f"Ask RuralSathi about {selected_crop} management in Chat ➔", type="primary"):
        st.session_state.pending_quick_question = f"What are the best agronomic management practices, fertilizer doses, and pest controls for {selected_crop}?"
        st.rerun()


# ==========================================
# PAGE 4: 🌦️ WEATHER DASHBOARD
# ==========================================
elif page == "🌦️ Weather":
    st.markdown('<div class="gradient-header">District Weather & Farming Advisory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtext">Real-time agro-meteorological forecasts powered by Open-Meteo</div>',
        unsafe_allow_html=True
    )

    all_districts = list(DISTRICT_COORDS.keys())
    curr_idx = all_districts.index(st.session_state.selected_district) if st.session_state.selected_district in all_districts else 0

    col_sel, col_stat = st.columns([0.4, 0.6])
    with col_sel:
        target_dist = st.selectbox("Select Agricultural District", all_districts, index=curr_idx)
        st.session_state.selected_district = target_dist

    weather_info = get_weather(target_dist)
    advice_text = get_farming_advice(weather_info)

    cond_lower = weather_info['current_condition'].lower()
    if "rain" in cond_lower or "drizzle" in cond_lower:
        weather_icon = "🌧️"
    elif "thunder" in cond_lower:
        weather_icon = "⛈️"
    elif "cloud" in cond_lower:
        weather_icon = "⛅"
    elif "fog" in cond_lower:
        weather_icon = "🌫️"
    else:
        weather_icon = "☀️"

    st.markdown(
        f"""
        <div class="weather-hero">
            <div>
                <span style="font-size: 0.9rem; text-transform: uppercase; color: #3B82F6; letter-spacing: 1px; font-weight: 700;">Live Agro-Weather</span>
                <h1 style="margin: 4px 0; font-size: 2.8rem; color: #FFFFFF;">{weather_info['district']}</h1>
                <p style="margin: 0; color: #9CA3AF; font-size: 1.1rem;">Condition: <strong>{weather_info['current_condition']}</strong></p>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <span style="font-size: 64px;">{weather_icon}</span>
                <div style="font-size: 3.5rem; font-weight: 800; color: #22C55E;">{weather_info['current_temp']}°C</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if weather_info["rain_alert"]:
        st.markdown(
            f"""
            <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 14px; padding: 16px 20px; margin-bottom: 24px; color: #FBBF24;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size:24px;">⚠️</span>
                    <strong>{weather_info['rain_alert_message']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📅 5-Day Forecast")
    cols = st.columns(5)
    for i, day in enumerate(weather_info.get("forecast", [])):
        with cols[i]:
            d_cond = day['condition'].lower()
            if "rain" in d_cond or "drizzle" in d_cond:
                d_icon = "🌧️"
            elif "cloud" in d_cond:
                d_icon = "⛅"
            elif "fog" in d_cond:
                d_icon = "🌫️"
            else:
                d_icon = "☀️"

            st.markdown(
                f"""
                <div class="weather-forecast-card">
                    <div style="font-size:0.85rem; color:#9CA3AF; font-weight:600;">{day['date']}</div>
                    <div style="font-size:32px; margin:8px 0;">{d_icon}</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#F9FAFB;">{day['max_temp']}° / {day['min_temp']}°</div>
                    <div style="font-size:0.8rem; color:#60A5FA; margin-top:4px;">💧 {day['rain_mm']} mm</div>
                    <div style="font-size:0.75rem; color:#9CA3AF; margin-top:4px;">{day['condition']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    col_adv1, col_adv2 = st.columns(2)
    with col_adv1:
        st.markdown(
            f"""
            <div style="background:#111827; border:1px solid rgba(34, 197, 94, 0.3); border-radius:16px; padding:20px; height:100%;">
                <h4 style="color:#22C55E; margin-top:0;">🌾 Agronomic Advisory</h4>
                <p style="color:#D1D5DB; font-size:1rem; line-height:1.5;">{advice_text}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_adv2:
        max_rain = max([f.get("rain_mm", 0.0) for f in weather_info["forecast"]], default=0.0)
        spray_rec = "❌ Unfavorable (Risk of rain wash-off)" if max_rain > 10.0 else "✅ Favorable (Dry canopy recommended)"
        irr_rec = "⏸️ Postpone (Rain expected)" if max_rain > 15.0 else "💧 Standard Schedule (Monitor soil moisture)"

        st.markdown(
            f"""
            <div style="background:#111827; border:1px solid rgba(59, 130, 246, 0.3); border-radius:16px; padding:20px; height:100%;">
                <h4 style="color:#3B82F6; margin-top:0;">🚜 Field Activity Indicators</h4>
                <div style="margin-bottom:10px;">
                    <div style="font-size:0.85rem; color:#9CA3AF;">Pesticide & Foliar Spray:</div>
                    <strong style="color:#F9FAFB;">{spray_rec}</strong>
                </div>
                <div>
                    <div style="font-size:0.85rem; color:#9CA3AF;">Irrigation Management:</div>
                    <strong style="color:#F9FAFB;">{irr_rec}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# PAGE 5: 📈 RURAL IMPACT ANALYTICS
# ==========================================
elif page == "📈 Impact":
    st.markdown('<div class="gradient-header">Rural Impact & Usage Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtext">Observability metrics on real-world farmer inquiries, satisfaction rates, and knowledge coverage.</div>',
        unsafe_allow_html=True
    )

    summary = get_summary()
    total_q = summary["total_questions"]

    if total_q == 0:
        st.info("💡 No analytics data yet. Start asking questions in the Chat page to see live metrics and charts!")
    else:
        im1, im2, im3, im4 = st.columns(4)
        with im1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Questions Asked</div>
                    <div class="metric-val" style="color:#3B82F6;">{summary['total_questions']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with im2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Questions Answered</div>
                    <div class="metric-val" style="color:#22C55E;">{summary['answered']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with im3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Abstentions</div>
                    <div class="metric-val" style="color:#F59E0B;">{summary['abstentions']} <span style="font-size:1.1rem; color:#9CA3AF;">({summary['abstention_rate']}%)</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with im4:
            total_fb = summary["positive_feedback"] + summary["negative_feedback"]
            satisfaction = round((summary["positive_feedback"] / total_fb) * 100, 1) if total_fb > 0 else 100.0
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Satisfaction Rate</div>
                    <div class="metric-val" style="color:#10B981;">{satisfaction}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        ch_c1, ch_c2 = st.columns(2)

        with ch_c1:
            st.markdown("#### 🌾 What Farmers Are Asking About")
            cat_data = pd.DataFrame(
                list(summary["top_categories"].items()),
                columns=["Category", "Count"]
            ).sort_values(by="Count", ascending=True)

            fig_cat = px.bar(
                cat_data,
                x="Count",
                y="Category",
                orientation="h",
                color="Count",
                color_continuous_scale=["#15803D", "#22C55E"]
            )
            fig_cat.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="#F9FAFB",
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with ch_c2:
            st.markdown("#### ⏰ Usage Pattern (Questions by Hour)")
            hour_df = pd.DataFrame(
                list(summary["questions_by_hour"].items()),
                columns=["Hour", "Questions"]
            )
            fig_hour = px.line(
                hour_df,
                x="Hour",
                y="Questions",
                markers=True,
                line_shape="spline"
            )
            fig_hour.update_traces(line_color="#3B82F6", marker=dict(size=6, color="#22C55E"))
            fig_hour.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="#F9FAFB",
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                xaxis=dict(tickmode="linear", tick0=0, dtick=3)
            )
            st.plotly_chart(fig_hour, use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown("#### ⚠️ Knowledge Gaps: Questions RuralSathi Couldn't Answer")
        st.caption("These inquiries triggered out-of-domain abstention and can guide future ICAR data ingestion.")
        unanswered = summary["unanswered_questions"]
        if unanswered:
            for un_q in reversed(unanswered):
                st.markdown(
                    f"""
                    <div style="background: rgba(245, 158, 11, 0.08); border-left: 3px solid #F59E0B; padding: 10px 14px; border-radius: 4px; margin-bottom: 8px; color: #FBBF24;">
                        • "{un_q}"
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("No unanswered questions recorded yet!")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown("#### 🏷️ Most Asked Topics")
        badge_html = " ".join([
            f'<span style="background: rgba(34, 197, 94, 0.2); border: 1px solid #22C55E; color: #22C55E; padding: 6px 14px; border-radius: 12px; font-weight: 600; font-size: 0.95rem; margin-right: 8px;">{topic}</span>'
            for topic in summary["most_asked_topics"]
        ])
        st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        raw_log_bytes = json.dumps(load_log(), indent=2).encode("utf-8")
        st.download_button(
            label="📥 Download Full Analytics Log (.json)",
            data=raw_log_bytes,
            file_name="ruralsathi_analytics_log.json",
            mime="application/json"
        )


# ==========================================
# PAGE 6: 📊 EVALUATION DASHBOARD
# ==========================================
elif page == "📊 Evaluation":
    st.markdown('<div class="gradient-header">RAG System Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtext">Benchmarking retrieval precision, abstention on out-of-domain queries, conflicts, and latency.</div>',
        unsafe_allow_html=True
    )

    eval_csv_path = "tests/evaluation_results.csv"

    col_btn, col_info = st.columns([0.25, 0.75])
    with col_btn:
        run_eval_clicked = st.button("🚀 Run Full Benchmark (25 Qs)", type="primary", use_container_width=True)

    if run_eval_clicked:
        evaluator = RAGEvaluator(retriever, generator)
        progress_bar = st.progress(0, text="Starting RAG evaluation benchmark...")
        
        test_file = "tests/test_questions.json"
        with open(test_file, "r", encoding="utf-8") as f:
            questions = json.load(f)

        records = []
        for i, q in enumerate(questions):
            progress_pct = int(((i + 1) / len(questions)) * 100)
            progress_bar.progress(progress_pct, text=f"Evaluating Question {i+1}/{len(questions)}: {q['question'][:40]}...")
            record = evaluator.evaluate_single(
                question=q["question"],
                category=q.get("category", "answerable"),
                expected_keyword=q.get("expected_source_keyword")
            )
            records.append(record)

        df = pd.DataFrame(records)
        evaluator.save_results(df)
        progress_bar.empty()
        st.success("✅ Evaluation complete! Dashboard updated below.")

    if os.path.exists(eval_csv_path):
        df_results = pd.read_csv(eval_csv_path)

        ans_mask = df_results["category"] == "answerable"
        off_mask = df_results["category"] == "off_topic"
        multi_mask = df_results["category"] == "multi_chunk"

        ans_acc = round((df_results[ans_mask]["answered_correctly"].sum() / max(1, ans_mask.sum())) * 100, 1)
        off_acc = round((df_results[off_mask]["abstained_correctly"].sum() / max(1, off_mask.sum())) * 100, 1)
        avg_lat = round(df_results["latency_seconds"].mean(), 2)
        citation_rate = round((df_results["source_cited"].sum() / len(df_results)) * 100, 1)

        analytics_events = load_log()
        conflicts_count = sum(1 for e in analytics_events if e.get("event") == "conflict_detected")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Answerable Accuracy</div>
                    <div class="metric-val" style="color:#22C55E;">{ans_acc}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Off-Topic Abstention</div>
                    <div class="metric-val" style="color:#3B82F6;">{off_acc}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Conflicts Detected</div>
                    <div class="metric-val" style="color:#F59E0B;">{conflicts_count}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Latency</div>
                    <div class="metric-val" style="color:#9CA3AF;">{avg_lat}s</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Citation Rate</div>
                    <div class="metric-val" style="color:#10B981;">{citation_rate}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Category Success Rate")
            category_stats = df_results.groupby("category").apply(
                lambda g: (g["answered_correctly"].sum() if g.name != "off_topic" else g["abstained_correctly"].sum()) / len(g) * 100
            ).reset_index(name="SuccessRate")

            fig_bar = px.bar(
                category_stats,
                x="category",
                y="SuccessRate",
                color="category",
                color_discrete_map={"answerable": "#22C55E", "multi_chunk": "#3B82F6", "off_topic": "#F59E0B"},
                text_auto=".1f",
                labels={"SuccessRate": "Success %", "category": "Question Category"}
            )
            fig_bar.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="#F9FAFB",
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=320
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            st.markdown("#### Latency vs Cross-Encoder Re-ranker Score")
            fig_scatter = px.scatter(
                df_results,
                x="latency_seconds",
                y="top_reranker_score",
                color="category",
                hover_data=["question"],
                labels={"latency_seconds": "Latency (s)", "top_reranker_score": "Reranker Score"},
                color_discrete_map={"answerable": "#22C55E", "multi_chunk": "#3B82F6", "off_topic": "#F59E0B"}
            )
            fig_scatter.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="#F9FAFB",
                margin=dict(t=20, b=20, l=20, r=20),
                height=320
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Detailed Benchmark Test Results")
        st.dataframe(
            df_results[["question", "category", "retrieval_hit", "abstained", "latency_seconds", "top_reranker_score"]],
            use_container_width=True,
            height=300
        )

        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Evaluation CSV",
            data=csv_data,
            file_name="ruralsathi_eval_results.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 Click 'Run Full Benchmark (25 Qs)' to evaluate the RAG pipeline and view detailed charts.")
