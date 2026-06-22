"""Tests for literature workflows: summarize, approve, and query."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from fedmaq_literature.registry import parse_registry
from fedmaq_literature.workflows.summarize import run_summarize
from fedmaq_literature.workflows.approve import run_approve
from fedmaq_literature.workflows.query import run_query


@pytest.fixture
def mock_repo_env(tmp_path: Path):
    # Setup standard folder structure
    (tmp_path / "papers").mkdir()
    (tmp_path / "markdown").mkdir()
    (tmp_path / "summaries" / "drafts").mkdir(parents=True)
    cursor_dir = tmp_path / ".cursor" / "project"
    cursor_dir.mkdir(parents=True)

    # Write a sample registry
    registry_file = cursor_dir / "paper_registry.md"
    registry_content = (
        "# Paper Registry\n\n"
        "| Slug | PDF | Baseline | Conversion | Indexing | Summary | Tags |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| test-paper | Test Paper | FedAvg | ready | none | none | test |\n"
    )
    registry_file.write_text(registry_content, encoding="utf-8")

    # Write a sample paper markdown
    paper_dir = tmp_path / "markdown" / "test-paper"
    paper_dir.mkdir()
    (paper_dir / "paper.md").write_text(
        "# Test Title\n\nThis is a converted research paper content.", encoding="utf-8"
    )
    (paper_dir / "meta.yaml").write_text(
        "page_count: 10\nchar_count: 50\n", encoding="utf-8"
    )

    return tmp_path


@patch("fedmaq_literature.workflows.summarize.call_llm")
def test_run_summarize_success(mock_call_llm: MagicMock, mock_repo_env: Path) -> None:
    mock_call_llm.return_value = (
        "## Structured Summary\n\nThis is a mock LLM generated summary."
    )

    # Run summarize for slug
    ret = run_summarize(slug="test-paper", root=mock_repo_env)
    assert ret == 0

    # Verify summary draft file is created
    draft_file = mock_repo_env / "summaries" / "drafts" / "test-paper.md"
    assert draft_file.is_file()
    assert "mock LLM generated summary" in draft_file.read_text(encoding="utf-8")

    # Verify registry is updated
    entries = parse_registry(
        mock_repo_env / ".cursor" / "project" / "paper_registry.md"
    )
    assert entries[0].summary == "draft"


def test_run_approve_success(mock_repo_env: Path) -> None:
    # Setup a draft file and set summary status in registry to "draft"
    draft_file = mock_repo_env / "summaries" / "drafts" / "test-paper.md"
    draft_file.write_text("## Test Draft Summary Content", encoding="utf-8")

    from fedmaq_literature.registry import update_registry_summary

    update_registry_summary("test-paper", "draft", root=mock_repo_env)

    # Run approve
    ret = run_approve(slug="test-paper", root=mock_repo_env)
    assert ret == 0

    # Verify draft file is deleted, and approved file is created
    assert not draft_file.exists()
    approved_file = mock_repo_env / "summaries" / "test-paper.md"
    assert approved_file.is_file()
    assert "Test Draft Summary Content" in approved_file.read_text(encoding="utf-8")

    # Verify registry is updated to "approved"
    entries = parse_registry(
        mock_repo_env / ".cursor" / "project" / "paper_registry.md"
    )
    assert entries[0].summary == "approved"


@patch("fedmaq_literature.workflows.query.call_llm")
@patch("fedmaq_literature.workflows.query.VectorStoreIndex")
@patch("fedmaq_literature.workflows.query.chromadb.PersistentClient")
@patch("fedmaq_literature.workflows.query.HuggingFaceEmbedding")
def test_run_query_success(
    mock_hfe: MagicMock,
    mock_chroma: MagicMock,
    mock_vsi: MagicMock,
    mock_call_llm: MagicMock,
    mock_repo_env: Path,
) -> None:
    # Create the storage/chroma directory so RAG thinks index exists
    (mock_repo_env / "storage" / "chroma").mkdir(parents=True)

    # Mock components
    mock_node = MagicMock()
    mock_node.text = "This is some retrieved context about KD."
    mock_node.score = 0.95
    mock_node.metadata = {"slug": "test-paper", "title": "Test Title"}

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [mock_node]

    mock_index_instance = MagicMock()
    mock_index_instance.as_retriever.return_value = mock_retriever
    mock_vsi.from_vector_store.return_value = mock_index_instance

    mock_call_llm.return_value = "Synthesized Answer: KD is knowledge distillation."

    # Run query
    ret = run_query(
        query_str="what is KD?",
        root=mock_repo_env,
        device="cpu",
    )
    assert ret == 0
    mock_call_llm.assert_called_once()
    assert "what is KD?" in mock_call_llm.call_args[1]["prompt"]
    assert (
        "This is some retrieved context about KD."
        in mock_call_llm.call_args[1]["prompt"]
    )
