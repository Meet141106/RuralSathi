import os
import json
import glob
import logging
import requests
import pandas as pd
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def load_pdf_file(filepath: str) -> list[dict]:
    """Load PDF files using PyMuPDF (fitz) page-by-page."""
    docs = []
    if fitz is None:
        logger.warning("PyMuPDF (fitz) is not installed. Skipping PDF: %s", filepath)
        return docs

    filename = os.path.basename(filepath)
    try:
        doc = fitz.open(filepath)
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text()
                if text and len(text.strip()) >= 50:
                    docs.append({
                        "text": text.strip(),
                        "source": filename,
                        "page": page_num + 1,
                        "type": "pdf"
                    })
            except Exception as page_err:
                logger.warning("Error reading page %d of %s: %s", page_num + 1, filename, page_err)
        doc.close()
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", filepath, e)

    return docs


def load_kcc_file(filepath: str, limit: int = 1000) -> list[dict]:
    """Load KCC JSON, JSONL, or CSV dataset up to limit rows."""
    docs = []
    if not os.path.exists(filepath):
        return docs

    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
            df = df.head(limit)
            for _, row in df.iterrows():
                query = str(row.get("QueryText") or row.get("query") or "").strip()
                ans = str(row.get("KccAns") or row.get("answer") or "").strip()
                crop = str(row.get("CropName") or row.get("crop") or "").strip()
                state = str(row.get("StateName") or row.get("state") or "").strip()

                if query and ans and ans.lower() != "nan":
                    docs.append({
                        "text": f"Q: {query}\nA: {ans}",
                        "source": "KCC Dataset",
                        "crop": crop,
                        "state": state,
                        "type": "kcc"
                    })

        elif filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        query = str(item.get("query") or item.get("QueryText") or item.get("question") or "").strip()
                        ans = str(item.get("answer") or item.get("KccAns") or item.get("response") or "").strip()
                        crop = str(item.get("crop") or item.get("CropName") or "").strip()
                        state = str(item.get("state") or item.get("StateName") or "").strip()
                        if query and ans:
                            docs.append({
                                "text": f"Q: {query}\nA: {ans}",
                                "source": "KCC Dataset",
                                "crop": crop,
                                "state": state,
                                "type": "kcc"
                            })
                    except json.JSONDecodeError:
                        continue

        elif filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = data[:limit]
                elif isinstance(data, dict):
                    items = data.get("data", [])[:limit]
                else:
                    items = []

                for item in items:
                    query = str(item.get("query") or item.get("QueryText") or item.get("question") or "").strip()
                    ans = str(item.get("answer") or item.get("KccAns") or item.get("response") or "").strip()
                    crop = str(item.get("crop") or item.get("CropName") or "").strip()
                    state = str(item.get("state") or item.get("StateName") or "").strip()
                    if query and ans:
                        docs.append({
                            "text": f"Q: {query}\nA: {ans}",
                            "source": "KCC Dataset",
                            "crop": crop,
                            "state": state,
                            "type": "kcc"
                        })
    except Exception as e:
        logger.error("Error loading KCC dataset from %s: %s", filepath, e)

    return docs


def load_txt_file(filepath: str) -> list[dict]:
    """Load plain TXT file and split by double newline into paragraphs."""
    docs = []
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) >= 50]
        for p in paragraphs:
            docs.append({
                "text": p,
                "source": filename,
                "type": "txt"
            })
    except Exception as e:
        logger.error("Failed to read TXT %s: %s", filepath, e)

    return docs


def create_fallback_kcc_data(target_path: str):
    """Fallback 20 realistic Indian farming Q&A pairs covering key topics."""
    qa_pairs = [
        {"query": "What fertilizer should I apply for wheat crop during crown root initiation?", "answer": "Apply top dressing of Urea (around 60 kg/ha nitrogen) at the time of first irrigation (Crown Root Initiation stage, 20-25 days after sowing). Ensure adequate soil moisture.", "crop": "Wheat", "state": "Punjab"},
        {"query": "How to manage yellow rust disease in wheat crop?", "answer": "Spray Propiconazole 25 EC (Tilt) @ 0.1% (1 ml per litre of water) or Tebuconazole 250 EC at first appearance of yellow pustules. Repeat after 15 days if cloudy weather persists.", "crop": "Wheat", "state": "Haryana"},
        {"query": "What are the recommended irrigation stages for wheat?", "answer": "Wheat requires 4 to 6 irrigations at critical growth stages: Crown Root Initiation (21 DAS), Tillering (40-45 DAS), Jointing (60-65 DAS), Flowering (80-85 DAS), and Dough stage (100-105 DAS).", "crop": "Wheat", "state": "Uttar Pradesh"},
        {"query": "How to control stem borer and leaf folder in rice paddy?", "answer": "Apply Cartap Hydrochloride 4G @ 10 kg/acre or spray Chlorantraniliprole 18.5 SC @ 60 ml/acre in 200 litres of water. Maintain a thin layer of standing water.", "crop": "Rice", "state": "Andhra Pradesh"},
        {"query": "What is the water management schedule for transplanted rice?", "answer": "Keep 2-3 cm standing water for the initial 2 weeks after transplanting for root establishment. Later adopt Alternate Wetting and Drying (AWD) to conserve water and prevent root rot.", "crop": "Rice", "state": "West Bengal"},
        {"query": "How to control aphids and whitefly in cotton?", "answer": "Use yellow sticky traps (10/acre) and spray Imidacloprid 17.8 SL @ 0.3 ml/litre or Flonicamid 50 WG @ 0.3 g/litre water. Avoid excess nitrogen fertilizer which attracts sucking pests.", "crop": "Cotton", "state": "Gujarat"},
        {"query": "What is the best sowing time for cotton in Maharashtra?", "answer": "Sow Bt cotton between May 25 to June 15 for irrigated crop, or immediately with the onset of regular monsoon rains in late June for rainfed black cotton soils.", "crop": "Cotton", "state": "Maharashtra"},
        {"query": "How much irrigation water is needed for sugarcane crop?", "answer": "Sugarcane requires approximately 1500 to 2500 mm water per season. In dry summer months, irrigate every 8-10 days, and during winter, irrigate every 15-20 days.", "crop": "Sugarcane", "state": "Maharashtra"},
        {"query": "How to control early shoot borer in sugarcane?", "answer": "Plant setts with Chlorpyriphos 20 EC @ 2 ml/litre drenching. Earthing up at 45 days after planting and release of Trichogramma chilonis egg parasitoid @ 20,000/acre at weekly intervals.", "crop": "Sugarcane", "state": "Uttar Pradesh"},
        {"query": "How to identify nitrogen deficiency in agricultural crops?", "answer": "Older lower leaves turn pale green to yellow (chlorosis) progressing upwards from leaf tip along the midrib in a V-shaped pattern, with stunted overall crop growth.", "crop": "General Crops", "state": "All India"},
        {"query": "How can farmers manage acidic or alkaline soil pH for better yield?", "answer": "For acidic soil (pH < 6.0), apply agricultural lime (calcium carbonate) or dolomite. For sodic/alkaline soil (pH > 8.0), apply gypsum (calcium sulphate) accompanied by deep ploughing and green manuring with Dhaincha.", "crop": "General Crops", "state": "Madhya Pradesh"},
        {"query": "What are the proper grain and seed storage techniques post-harvest?", "answer": "Sun dry seeds until moisture drops below 10-12%. Clean and store in airtight metal bins or hermetic bags. Treat storage area with Malathion 50 EC spray and keep neem leaves inside containers to deter weevils.", "crop": "General Crops", "state": "Rajasthan"},
        {"query": "Who is eligible for PM-KISAN Samman Nidhi scheme?", "answer": "All landholding farmer families with cultivable land in their names are eligible for Rs 6000/year in 3 equal installments of Rs 2000, subject to exclusions like institutional landholders and income tax payees.", "crop": "Government Scheme", "state": "All India"},
        {"query": "What documents are required to apply for Pradhan Mantri Fasal Bima Yojana (PMFBY)?", "answer": "Documents include Aadhaar card, Bank passbook, Land Record (Khasra/Khatauni/7-12 extract), Sowing Certificate or self-declaration, and Land tenancy/lease agreement if tenant farmer.", "crop": "Government Scheme", "state": "All India"},
        {"query": "How can a farmer apply for a Kisan Credit Card (KCC)?", "answer": "Submit the simplified KCC application at the nearest commercial, cooperative, or rural regional bank along with land title records, Aadhaar, PAN/Voter ID, and passport photos. KCC gives collateral-free credit up to Rs 1.6 lakh.", "crop": "Government Scheme", "state": "All India"},
        {"query": "What precautions should be taken when heavy rainfall or storm warning is issued?", "answer": "Ensure proper drainage channels in crop fields to drain excess standing water quickly, postpone fertilizer and pesticide spraying, harvest mature crops immediately, and move livestock to elevated sheltered areas.", "crop": "Weather Advisory", "state": "Odisha"},
        {"query": "How to control mustard aphids (Lipaphis erysimi) effectively?", "answer": "Sow mustard early (before mid-October) to escape aphid attack. Spray Dimethoate 30 EC @ 1 ml/litre or Thiamethoxam 25 WG @ 0.2 g/litre when 20% plants show infestation during cloudy weather.", "crop": "Mustard", "state": "Rajasthan"},
        {"query": "What vegetables are suitable for cultivation in Kharif season?", "answer": "Suitable Kharif vegetables include Okra (Bhindi), Bitter Gourd, Bottle Gourd, Cowpea, Tomato, Brinjal, and Chilli. Ensure raised bed planting to prevent waterlogging during monsoon.", "crop": "Vegetables", "state": "Karnataka"},
        {"query": "What is the Minimum Support Price (MSP) mechanism for paddy and wheat?", "answer": "MSP is announced prior to sowing season by the Government of India based on CACP recommendations, guaranteeing a baseline price covering at least 1.5 times the cost of production (A2+FL).", "crop": "Government Scheme", "state": "All India"},
        {"query": "How to prepare and apply Jeevamrit for organic soil health enhancement?", "answer": "Mix 10 kg cow dung, 10 litres cow urine, 2 kg jaggery, 2 kg pulse flour, and a handful of fertile farm soil in 200 litres water. Ferment for 48 hours stirring twice daily, and apply with irrigation water or spray @ 10% concentration.", "crop": "Organic Farming", "state": "All India"}
    ]
    with open(target_path, "w", encoding="utf-8") as f:
        for item in qa_pairs:
            f.write(json.dumps(item) + "\n")
    logger.info("Saved %d fallback Q&A pairs to %s", len(qa_pairs), target_path)


def download_sample_kcc_data(target_dir: str = RAW_DATA_DIR) -> str:
    """Download KCC sample dataset or create high quality fallback."""
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "kcc_sample.jsonl")
    url = "https://huggingface.co/datasets/prabhu09/kcc-agricultural-qa/resolve/main/data/train.jsonl"

    print("Checking / Downloading KCC sample dataset...")
    try:
        response = requests.get(url, timeout=12)
        if response.status_code == 200 and len(response.text.strip()) > 100:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Successfully downloaded KCC dataset to {target_path}")
            return target_path
        else:
            print(f"Download returned status {response.status_code}. Generating fallback sample Q&A.")
            create_fallback_kcc_data(target_path)
            return target_path
    except Exception as e:
        print(f"Dataset download failed ({e}). Creating fallback sample dataset.")
        create_fallback_kcc_data(target_path)
        return target_path


def load_all_documents(raw_dir: str = RAW_DATA_DIR) -> list[dict]:
    """Load all PDFs, TXTs, and KCC files from raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)

    # Check if empty, then download sample
    existing_files = [f for f in os.listdir(raw_dir) if not f.startswith(".")]
    if not existing_files:
        download_sample_kcc_data(raw_dir)

    all_docs = []
    pdf_count = 0
    kcc_count = 0
    txt_count = 0

    # 1. PDFs
    pdf_files = glob.glob(os.path.join(raw_dir, "*.pdf"))
    for pdf in pdf_files:
        docs = load_pdf_file(pdf)
        pdf_count += len(docs)
        all_docs.extend(docs)

    # 2. KCC data (.jsonl, .json, .csv)
    kcc_files = (
        glob.glob(os.path.join(raw_dir, "*kcc*.jsonl")) +
        glob.glob(os.path.join(raw_dir, "*kcc*.csv")) +
        glob.glob(os.path.join(raw_dir, "*kcc*.json")) +
        glob.glob(os.path.join(raw_dir, "train.jsonl"))
    )
    # Deduplicate paths
    kcc_files = list(set(kcc_files))
    for kcc in kcc_files:
        docs = load_kcc_file(kcc)
        kcc_count += len(docs)
        all_docs.extend(docs)

    # 3. TXT files
    txt_files = glob.glob(os.path.join(raw_dir, "*.txt"))
    for txt in txt_files:
        docs = load_txt_file(txt)
        txt_count += len(docs)
        all_docs.extend(docs)

    print(f"Loaded {pdf_count} PDF pages, {kcc_count} KCC entries, {txt_count} text blocks = Total {len(all_docs)} documents")
    return all_docs


if __name__ == "__main__":
    docs = load_all_documents()
