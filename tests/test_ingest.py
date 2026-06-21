"""Tests for LlamaIndex + ChromaDB ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml

from fedmaq_literature.ingest.pipeline import run_ingest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding


def test_run_ingest(tmp_path: Path) -> None:
    # 1. Setup mock directories
    (tmp_path / "papers").mkdir()
    (tmp_path / "markdown").mkdir()
    cursor_dir = tmp_path / ".cursor" / "project"
    cursor_dir.mkdir(parents=True)

    # 2. Setup mock registry (7-column layout)
    registry_file = cursor_dir / "paper_registry.md"
    registry_content = (
        "# Paper Registry\n\n"
        "| Slug | PDF | Baseline | Conversion | Indexing | Summary | Tags |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| test-paper | Test Paper | FedAvg | ready | none | none | test |\n"
    )
    registry_file.write_text(registry_content, encoding="utf-8")

    # 3. Setup mock converted markdown
    paper_dir = tmp_path / "markdown" / "test-paper"
    paper_dir.mkdir()
    (paper_dir / "paper.md").write_text(
        "# Test Title\n\nThis is a test paper.", encoding="utf-8"
    )

    meta_content = {"page_count": 5, "char_count": 22, "confidence": 0.99}
    (paper_dir / "meta.yaml").write_text(yaml.safe_dump(meta_content), encoding="utf-8")

    # 4. Mock the Chroma db persistent client and collection
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    # Patch chromadb and HuggingFaceEmbedding
    with (
        patch("chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("fedmaq_literature.ingest.pipeline.HuggingFaceEmbedding") as mock_hfe,
    ):

        # Set HuggingFaceEmbedding to return MockEmbedding
        mock_hfe.return_value = MockEmbedding(embed_dim=384)

        # Run the ingest pipeline for a specific slug
        ret = run_ingest(slug="test-paper", root=tmp_path, device="cpu")
        assert ret == 0

        # Verify collection delete was called to clear duplicate entries
        mock_collection.delete.assert_called_once_with(where={"slug": "test-paper"})

        # Check if registry was updated to 'ready' for indexing
        text = registry_file.read_text(encoding="utf-8")
        assert (
            "| test-paper | Test Paper | FedAvg | ready | ready | none | test |" in text
        )

        # Reset registry to 'none' for testing next run
        from fedmaq_literature.registry import update_registry_indexing

        update_registry_indexing("test-paper", "none", root=tmp_path)

        # Verify collection reset and run with slug=None
        mock_collection.reset_mock()
        ret = run_ingest(slug=None, root=tmp_path, device="cpu")
        assert ret == 0
        mock_collection.delete.assert_called_once_with(where={"slug": "test-paper"})

        # Verify it is ready again
        text = registry_file.read_text(encoding="utf-8")
        assert (
            "| test-paper | Test Paper | FedAvg | ready | ready | none | test |" in text
        )


def test_run_ingest_failure(tmp_path: Path) -> None:
    # 1. Setup mock directories
    (tmp_path / "papers").mkdir()
    (tmp_path / "markdown").mkdir()
    cursor_dir = tmp_path / ".cursor" / "project"
    cursor_dir.mkdir(parents=True)

    # 2. Setup mock registry (7-column layout)
    registry_file = cursor_dir / "paper_registry.md"
    registry_content = (
        "# Paper Registry\n\n"
        "| Slug | PDF | Baseline | Conversion | Indexing | Summary | Tags |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| test-paper | Test Paper | FedAvg | ready | none | none | test |\n"
    )
    registry_file.write_text(registry_content, encoding="utf-8")

    # 3. Setup mock converted markdown
    paper_dir = tmp_path / "markdown" / "test-paper"
    paper_dir.mkdir()
    (paper_dir / "paper.md").write_text(
        "# Test Title\n\nThis is a test paper.", encoding="utf-8"
    )

    meta_content = {"page_count": 5, "char_count": 22, "confidence": 0.99}
    (paper_dir / "meta.yaml").write_text(yaml.safe_dump(meta_content), encoding="utf-8")

    # 4. Mock the Chroma db persistent client and collection
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    # Patch chromadb, HuggingFaceEmbedding, and IngestionPipeline
    with (
        patch("chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("fedmaq_literature.ingest.pipeline.HuggingFaceEmbedding") as mock_hfe,
        patch(
            "fedmaq_literature.ingest.pipeline.IngestionPipeline"
        ) as mock_pipeline_class,
    ):
        mock_hfe.return_value = MockEmbedding(embed_dim=384)

        # Make pipeline.run raise an exception
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.side_effect = RuntimeError("Ingestion failed")
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Run the ingest pipeline and check for failure return code
        ret = run_ingest(slug="test-paper", root=tmp_path, device="cpu")
        assert ret == 1

        # Check if registry was updated to 'failed' for indexing
        text = registry_file.read_text(encoding="utf-8")
        assert (
            "| test-paper | Test Paper | FedAvg | ready | failed | none | test |"
            in text
        )
