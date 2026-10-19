import re
import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list) -> list:
    """Split documents into chunks using RecursiveCharacterTextSplitter while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_counter = {}

    for doc in documents:
        text = doc.get("text", "").strip()
        if not text:
            continue

        source = doc.get("source", "unknown")
        doc_type = doc.get("type", "general")

        # Metadata extraction
        metadata = {}
        for key in ["page", "crop", "state", "type", "source"]:
            if key in doc and doc[key]:
                metadata[key] = doc[key]

        # For short items (like individual KCC Q&As), if they fit within chunk size, keep them intact
        if len(text) <= CHUNK_SIZE:
            text_splits = [text]
        else:
            text_splits = splitter.split_text(text)

        for split in text_splits:
            cleaned = split.strip()
            if len(cleaned) < 100:
                continue

            clean_source_tag = re.sub(r'[^a-zA-Z0-9_-]', '_', source)
            chunk_counter[clean_source_tag] = chunk_counter.get(clean_source_tag, 0) + 1
            chunk_id = f"{clean_source_tag}_{chunk_counter[clean_source_tag]}"

            chunks.append({
                "text": cleaned,
                "source": source,
                "type": doc_type,
                "chunk_id": chunk_id,
                "metadata": metadata
            })

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
