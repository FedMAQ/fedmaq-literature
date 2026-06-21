"""LlamaIndex ingestion pipeline with ChromaDB and Hugging Face embeddings."""

from __future__ import annotations

import os
from pathlib import Path
import chromadb
import yaml

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from fedmaq_literature.paths import markdown_dir, storage_dir, registry_path
from fedmaq_literature.registry import parse_registry
from fedmaq_literature.ingest import (
    DEFAULT_EMBED_MODEL,
    EMBED_BATCH_SIZE,
    RETRIEVAL_INSTRUCT,
)


def run_ingest(
    slug: str | None = None,
    *,
    root: Path | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> int:
    """Run LlamaIndex + Chroma ingestion for converted markdown files.

    If slug is specified, ingest only that paper (deleting any existing entries for it first).
    If slug is None, ingest all papers in the registry that are marked 'ready'.
    """
    model_name = model_name or DEFAULT_EMBED_MODEL
    batch_size = batch_size or EMBED_BATCH_SIZE

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure float16 for Qwen models
    model_kwargs = {}
    if "Qwen" in model_name:
        model_kwargs["dtype"] = torch.float16

    print(f"Initializing embedding model: {model_name} on {device}...")
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        trust_remote_code=True,
        embed_batch_size=batch_size,
        query_instruction=RETRIEVAL_INSTRUCT,
        model_kwargs=model_kwargs,
    )

    db_path = storage_dir(root) / "chroma"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Chroma DB at {db_path}...")
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    chroma_collection = chroma_client.get_or_create_collection("literature")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    entries = parse_registry(registry_path(root))

    if slug:
        target_entries = [e for e in entries if e.slug == slug]
        if not target_entries:
            print(f"Error: Slug '{slug}' not found in registry.")
            return 1
    else:
        target_entries = [e for e in entries if e.conversion == "ready"]

    if not target_entries:
        print("No papers ready for ingestion.")
        return 0

    documents = []
    md_dir = markdown_dir(root)

    for entry in target_entries:
        paper_dir = md_dir / entry.slug
        paper_file = paper_dir / "paper.md"
        meta_file = paper_dir / "meta.yaml"

        if not paper_file.is_file():
            print(
                f"Warning: Converted paper markdown not found for '{entry.slug}' at {paper_file}. Skipping."
            )
            continue

        text = paper_file.read_text(encoding="utf-8")

        meta_data = {}
        if meta_file.is_file():
            try:
                meta_data = yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
            except Exception as e:
                print(f"Warning: Failed to parse meta.yaml for '{entry.slug}': {e}")

        doc_metadata = {
            "slug": entry.slug,
            "title": entry.pdf_label,
            "baseline": entry.baseline,
            "tags": entry.tags,
            "page_count": meta_data.get("page_count", 0),
            "char_count": meta_data.get("char_count", len(text)),
        }

        doc = Document(
            text=text,
            metadata=doc_metadata,
            doc_id=entry.slug,
        )
        documents.append(doc)

    if not documents:
        print("No documents successfully loaded.")
        return 0

    # Clean existing entries from Chroma to prevent duplicates
    for doc in documents:
        s = doc.metadata["slug"]
        print(f"Cleaning existing Chroma entries for slug '{s}'...")
        try:
            chroma_collection.delete(where={"slug": s})
        except Exception:
            pass

    print(f"Running IngestionPipeline for {len(documents)} document(s)...")
    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(chunk_size=512, chunk_overlap=128),
            embed_model,
        ],
        vector_store=vector_store,
    )

    pipeline.run(documents=documents)
    print("Ingestion completed successfully.")
    return 0
