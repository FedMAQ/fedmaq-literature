"""Literature RAG query and synthesis workflow using LlamaIndex and OpenRouter."""

from __future__ import annotations

from pathlib import Path
import os
import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from fedmaq_literature.paths import repo_root, storage_dir
from fedmaq_literature.ingest import DEFAULT_EMBED_MODEL, RETRIEVAL_INSTRUCT
from fedmaq_literature.workflows.llm import call_llm, DEFAULT_SYNTHESIS_MODEL


def run_query(
    query_str: str,
    *,
    model: str | None = None,
    limit: int = 5,
    device: str | None = None,
    root: Path | None = None,
) -> int:
    """Query Chroma DB using Hugging Face embeddings and synthesize response via OpenRouter."""
    root_path = repo_root(root)
    model_name = model or DEFAULT_SYNTHESIS_MODEL

    # Connect to Chroma DB
    db_path = storage_dir(root_path) / "chroma"
    if not db_path.is_dir():
        print(
            f"Error: Chroma DB directory not found at {db_path}. Please run 'fedmaq-lit ingest' first."
        )
        return 1

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure model precision/dtype
    model_kwargs = {}
    dtype_str = os.environ.get("FEDMAQ_EMBED_DTYPE", "float16").lower()
    if device == "cuda":
        if dtype_str == "float16":
            model_kwargs["dtype"] = torch.float16
        elif dtype_str == "bfloat16":
            model_kwargs["dtype"] = torch.bfloat16
        elif dtype_str == "float32":
            model_kwargs["dtype"] = torch.float32
    else:
        # Fallback to float32 on CPU unless bfloat16 is specified
        if dtype_str == "bfloat16":
            model_kwargs["dtype"] = torch.bfloat16
        else:
            model_kwargs["dtype"] = torch.float32

    print(
        f"Initializing embedding model for retrieval: {DEFAULT_EMBED_MODEL} on {device}..."
    )
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=DEFAULT_EMBED_MODEL,
            device=device,
            trust_remote_code=True,
            query_instruction=RETRIEVAL_INSTRUCT,
            model_kwargs=model_kwargs,
        )
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return 1

    print("Connecting to Chroma DB collection 'literature'...")
    try:
        chroma_client = chromadb.PersistentClient(path=str(db_path))
        chroma_collection = chroma_client.get_collection("literature")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    except Exception as e:
        print(
            f"Error connecting to Chroma DB collection: {e}. Please ensure papers are indexed."
        )
        return 1

    print("Building search index...")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    print(f"Retrieving top {limit} passages for query: '{query_str}'...")
    try:
        retriever = index.as_retriever(similarity_top_k=limit)
        nodes = retriever.retrieve(query_str)
    except Exception as e:
        print(f"Error retrieving from vector store: {e}")
        return 1

    if not nodes:
        print("No matching passages found in the literature database.")
        return 0

    # Build context from retrieved nodes
    context_parts = []
    print("\nRetrieved Passages:")
    for idx, node in enumerate(nodes, 1):
        slug = node.metadata.get("slug", "unknown")
        title = node.metadata.get("title", "unknown")
        score = node.score if node.score is not None else 0.0
        print(f"  [{idx}] {slug} (similarity: {score:.4f})")
        context_parts.append(
            f"Source [{slug}] - Title: {title}\n" f"Content:\n{node.text}\n"
        )
    print()

    context = "\n---\n".join(context_parts)

    system_prompt = (
        "You are an expert academic assistant for the FedMAQ thesis project (Communication-Efficient "
        "Federated Learning via Multi-Adaptive Quantization and Knowledge Distillation). "
        "Use the provided literature context to answer the research question. "
        "Synthesize the findings across different papers, highlight consensus or contradictions, "
        "and draft a detailed, professional, and academic answer. "
        "Cite the specific papers (using their slugs, e.g., [mcmahan-2017-fedavg]) when referencing "
        "their methods, results, or equations. If the provided context is insufficient, state that."
    )

    prompt = (
        f"Context information from the literature database:\n"
        f"=====================\n"
        f"{context}\n"
        f"=====================\n\n"
        f"Question: {query_str}\n\n"
        f"Please write a comprehensive, synthesis-based academic answer citing the source papers above:"
    )

    print(f"Synthesizing answer using OpenRouter model '{model_name}'...")
    try:
        answer = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=0.2,
        )
        print("\n=== Answer ===\n")
        print(answer)
        print("\n==============\n")
        return 0
    except Exception as e:
        print(f"Error synthesizing answer from OpenRouter: {e}")
        return 1
