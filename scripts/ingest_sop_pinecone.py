import os
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from pinecone import Pinecone, ServerlessSpec
from google import genai
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. PATH RESOLUTION & SETUP
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
load_dotenv(project_root / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cold-chain-sops")
TARGET_DIMENSION = 3072  # Matches models/gemini-embedding-001

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY or GEMINI_API_KEY in .env")

# Cache directory to avoid re-embedding unchanged files
cache_dir = project_root / "data" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
HASH_CACHE_FILE = cache_dir / "ingestion_hash_cache.json"

hash_cache = {}
if HASH_CACHE_FILE.exists():
    try:
        with open(HASH_CACHE_FILE, "r") as f:
            hash_cache = json.load(f)
    except Exception:
        hash_cache = {}

# 2. CLIENT INITIALIZATION & PINECONE INDEX SETUP
pc = Pinecone(api_key=PINECONE_API_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

existing_indexes = pc.list_indexes().names()

if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != TARGET_DIMENSION:
        print(f"⚠️ Fixing dimension: Recreating {INDEX_NAME} for {TARGET_DIMENSION} dimensions...")
        pc.delete_index(INDEX_NAME)
        existing_indexes = [name for name in existing_indexes if name != INDEX_NAME]

if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME} ({TARGET_DIMENSION} Dim)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=TARGET_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index_client = pc.Index(INDEX_NAME)

# 3. DOCUMENT PARSER
def parse_and_chunk_document(doc_path: Path):
    ext = doc_path.suffix.lower()
    raw_texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)

    if ext == ".md":
        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        raw_text = doc_path.read_text(encoding="utf-8")
        header_docs = md_splitter.split_text(raw_text)
        split_chunks = text_splitter.split_documents(header_docs)
        raw_texts = [c.page_content for c in split_chunks]

    elif ext == ".txt":
        raw_text = doc_path.read_text(encoding="utf-8")
        raw_texts = text_splitter.split_text(raw_text)

    elif ext in [".csv", ".xlsx"]:
        try:
            df = pd.read_csv(doc_path) if ext == ".csv" else pd.read_excel(doc_path)
            for _, row in df.iterrows():
                row_items = [f"{col}: {val}" for col, val in row.to_dict().items() if pd.notna(val) and str(val).strip()]
                if row_items:
                    raw_texts.append(" | ".join(row_items))
        except Exception as e:
            print(f"  ❌ Error reading table {doc_path.name}: {e}")
            return []

    return [t.strip() for t in raw_texts if t.strip()]

# 4. INCREMENTAL PIPELINE WITH HASH CACHING
# Checking both possible SOP directories
sops_dir = project_root / "data" / "source" / "sops"
if not sops_dir.exists():
    sops_dir = project_root / "data" / "policy"

target_patterns = ["*.md", "*.txt", "*.csv", "*.xlsx"]
current_files = {}
for pattern in target_patterns:
    for file_path in sops_dir.glob(pattern):
        current_files[file_path.name] = file_path

print(f"Found {len(current_files)} SOP file(s) in {sops_dir}...")

updated_cache = dict(hash_cache)
cache_modified = False

# Remove deleted files from Pinecone
deleted_files = set(hash_cache.keys()) - set(current_files.keys())
for deleted_file in deleted_files:
    print(f"🗑️ Detected deleted file: {deleted_file}. Purging from Pinecone...")
    try:
        index_client.delete(filter={"source_file": {"$eq": deleted_file}})
        updated_cache.pop(deleted_file, None)
        cache_modified = True
    except Exception as e:
        print(f"  ❌ Failed to purge {deleted_file}: {e}")

# Process current files
for file_name, file_path in current_files.items():
    file_bytes = file_path.read_bytes()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if hash_cache.get(file_name) == file_hash:
        print(f"✨ Skipped (Unchanged): {file_name}")
        continue

    print(f"🔄 Processing updates: {file_name}...")
    cache_modified = True

    chunks = parse_and_chunk_document(file_path)
    if not chunks:
        print(f"  ⚠️ No valid text chunks extracted from {file_name}.")
        continue

    records = []
    for idx, chunk in enumerate(chunks):
        resp = ai_client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=chunk
        )
        
        if hasattr(resp, "embedding") and resp.embedding and hasattr(resp.embedding, "values"):
            vector = resp.embedding.values
        elif hasattr(resp, "embeddings") and resp.embeddings:
            vector = resp.embeddings[0].values
        else:
            vector = resp.values

        records.append({
            "id": f"{file_name}-chunk-{idx}",
            "values": vector,
            "metadata": {
                "source_file": file_name,
                "file_format": file_path.suffix.replace(".", "").upper(),
                "text": chunk
            }
        })

    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        index_client.upsert(vectors=batch)

    updated_cache[file_name] = file_hash

# 5. SYNC CACHE FILE
if cache_modified:
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(updated_cache, f, indent=4)
    print("✅ Ingestion & cache update complete.")
else:
    print("🌴 Index is already up-to-date.")