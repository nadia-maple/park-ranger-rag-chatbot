from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "chunks_embeddings_outputs"


@dataclass(frozen=True)
class VectorStoreConfig:
    input_files: Sequence[str] = (
        str(PROCESSED_DIR / "canada-top-parks.json"),
        str(PROCESSED_DIR / "us-top-parks.json"),
    )
    chroma_dir: str = str(PROJECT_ROOT / "chroma_parks_db")
    collection_name: str = "parks_rag_v1"
    reset_collection: bool = True
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1200
    chunk_overlap: int = 220
    min_chunk_len: int = 60
    output_dir: str = str(OUTPUT_DIR)



def normalize_text(text: Any) -> str:
    text = str(text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text



def split_preserve_tables(text: str) -> List[Tuple[str, str]]:
    normalized = str(text)
    parts: List[Tuple[str, str]] = []
    table_pattern = re.compile(r"(?:^|\n)(?:[^\n]*\|[^\n]*\n){2,}", re.MULTILINE)
    position = 0

    for match in table_pattern.finditer(normalized):
        before = normalized[position:match.start()]
        table = match.group(0)
        if before.strip():
            parts.append(("text", before.strip()))
        parts.append(("table", table.strip()))
        position = match.end()

    remainder = normalized[position:]
    if remainder.strip():
        parts.append(("text", remainder.strip()))
    if not parts and normalized.strip():
        parts.append(("text", normalized.strip()))
    return parts



def recursive_sentence_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_text(text)
    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
            continue

        if current:
            chunks.append(current)

        if len(sentence) <= chunk_size:
            current = sentence
            continue

        step = max(1, chunk_size - overlap)
        for index in range(0, len(sentence), step):
            piece = sentence[index : index + chunk_size].strip()
            if piece:
                chunks.append(piece)
        current = ""

    if current:
        chunks.append(current)
    return chunks



def add_context_header(title: str, chunk_text: str, url: str = "", source: str = "") -> str:
    title = normalize_text(title)
    chunk_text = normalize_text(chunk_text)
    url = normalize_text(url)
    source = normalize_text(source)

    header_parts = []
    if title:
        header_parts.append(f"Title: {title}")
    if source:
        header_parts.append(f"Source: {source}")
    if url:
        header_parts.append(f"URL: {url}")

    header = "\n".join(header_parts)
    return f"{header}\n\n{chunk_text}" if header else chunk_text



def load_documents(input_files: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for file_path_str in input_files:
        file_path = Path(file_path_str)
        if not file_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of documents in {file_path}, got {type(data).__name__}")

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "doc_id": f"{file_path.stem}_{index}",
                    "file": str(file_path),
                    "title": normalize_text(item.get("title", "")),
                    "url": normalize_text(item.get("url", "")),
                    "source": normalize_text(item.get("source", "")),
                    "content": normalize_text(item.get("content", "")),
                }
            )

    documents_df = pd.DataFrame(rows)
    if documents_df.empty:
        raise ValueError("No usable source documents were loaded.")
    return documents_df



def build_chunks(documents_df: pd.DataFrame, config: VectorStoreConfig) -> pd.DataFrame:
    chunks: List[Dict[str, Any]] = []
    for _, row in documents_df.iterrows():
        if not row["content"]:
            continue
        chunk_num = 0
        parts = split_preserve_tables(row["content"])
        for part_type, part_text in parts:
            if part_type == "table":
                piece_list = [normalize_text(part_text)]
            else:
                piece_list = recursive_sentence_chunk(
                    part_text,
                    chunk_size=config.chunk_size,
                    overlap=config.chunk_overlap,
                )

            for piece in piece_list:
                if len(piece) < config.min_chunk_len:
                    continue
                contextual_text = add_context_header(
                    row["title"],
                    piece,
                    url=row["url"],
                    source=row["source"],
                )
                chunks.append(
                    {
                        "id": f"{row['doc_id']}_c{chunk_num}",
                        "doc_id": row["doc_id"],
                        "file": row["file"],
                        "title": row["title"],
                        "title_normalized": normalize_text(row["title"]).lower(),
                        "url": row["url"],
                        "source": row["source"],
                        "text": contextual_text,
                        "raw_text": normalize_text(piece),
                        "chars": len(contextual_text),
                    }
                )
                chunk_num += 1

    chunks_df = pd.DataFrame(chunks)
    if chunks_df.empty:
        raise ValueError("Chunking produced no usable chunks.")
    return chunks_df



def save_outputs(chunks_df: pd.DataFrame, config: VectorStoreConfig):
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunks_csv = output_path / "parks_chunks.csv"
    summary_csv = output_path / "parks_summary.csv"
    manifest_json = output_path / "vector_store_manifest.json"

    chunks_df.to_csv(chunks_csv, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "documents": int(chunks_df["doc_id"].nunique()),
                "chunks_created": int(len(chunks_df)),
                "avg_chunk_chars": round(float(chunks_df["chars"].mean()), 2),
                "embedding_model": config.embed_model,
                "vector_db": "Chroma",
                "collection_name": config.collection_name,
                "reset_collection": config.reset_collection,
            }
        ]
    )
    summary_df.to_csv(summary_csv, index=False)

    manifest = {
        "embedding_model": config.embed_model,
        "collection_name": config.collection_name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "min_chunk_len": config.min_chunk_len,
        "documents": int(chunks_df["doc_id"].nunique()),
        "chunks_created": int(len(chunks_df)),
    }
    with open(manifest_json, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return str(chunks_csv), str(summary_csv), str(manifest_json)



def create_embeddings(chunks_df: pd.DataFrame, config: VectorStoreConfig):
    logger.info("Loading embedding model: %s", config.embed_model)
    model = SentenceTransformer(config.embed_model)
    logger.info("Creating embeddings for %s chunks", len(chunks_df))
    embeddings = model.encode(
        chunks_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    ).tolist()
    return model, embeddings



def get_or_reset_collection(client, collection_name: str, reset: bool = False):
    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection: %s", collection_name)
        except Exception:
            logger.info("Collection %s did not exist prior to reset", collection_name)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )



def store_in_chroma(chunks_df: pd.DataFrame, embeddings: List[List[float]], config: VectorStoreConfig):
    logger.info("Creating/using Chroma DB at: %s", config.chroma_dir)
    client = chromadb.PersistentClient(path=config.chroma_dir)
    collection = get_or_reset_collection(client, config.collection_name, reset=config.reset_collection)
    collection.upsert(
        ids=chunks_df["id"].tolist(),
        documents=chunks_df["text"].tolist(),
        embeddings=embeddings,
        metadatas=chunks_df[
            ["title", "title_normalized", "url", "source", "file", "doc_id"]
        ].to_dict(orient="records"),
    )
    logger.info("Stored %s chunks in Chroma collection: %s", collection.count(), config.collection_name)
    return client, collection



def truncate_preview(text: str, max_chars: int = 200) -> str:
    normalized = normalize_text(text)
    return normalized if len(normalized) <= max_chars else normalized[: max_chars - 3] + "..."



def test_query(model, collection, query: str = "What parks mention camping discounts or park passes?") -> Dict[str, Any]:
    logger.info("Running test query: %s", query)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=["documents", "metadatas"],
    )

    for index, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
        logger.info(
            "Result %s | title=%s | source=%s | file=%s | url=%s | preview=%s",
            index,
            meta.get("title"),
            meta.get("source"),
            meta.get("file"),
            meta.get("url"),
            truncate_preview(doc),
        )
    return results



def main() -> None:
    config = VectorStoreConfig()
    logger.info("Loading source documents...")
    documents_df = load_documents(config.input_files)
    logger.info("Loaded %s source documents", len(documents_df))

    logger.info("Building chunks...")
    chunks_df = build_chunks(documents_df, config)
    logger.info("Created %s chunks", len(chunks_df))

    chunks_csv, summary_csv, manifest_json = save_outputs(chunks_df, config)
    logger.info("Saved chunks CSV: %s", chunks_csv)
    logger.info("Saved summary CSV: %s", summary_csv)
    logger.info("Saved manifest JSON: %s", manifest_json)

    model, embeddings = create_embeddings(chunks_df, config)
    _, collection = store_in_chroma(chunks_df, embeddings, config)
    _ = test_query(model, collection)
    logger.info("Vector store build completed successfully.")


if __name__ == "__main__":
    main()
