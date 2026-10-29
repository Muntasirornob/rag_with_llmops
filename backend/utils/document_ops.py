from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from backend.logger import GLOBAL_LOGGER as log
from backend.exception.custom_exception import DocumentPortalException


def _loader_for(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".docx":
        return Docx2txtLoader(str(path))
    if ext in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    if ext == ".csv":
        return CSVLoader(str(path), encoding="utf-8")
    return None


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load text from each file path into LangChain Documents."""
    docs: List[Document] = []
    for p in paths:
        path = Path(p)
        loader = _loader_for(path)
        if loader is None:
            log.warning("No loader for file type, skipped", path=str(path))
            continue
        try:
            docs.extend(loader.load())
        except Exception as e:
            log.error("Failed to load document", path=str(path), error=str(e))
            raise DocumentPortalException(f"Failed to load document: {path.name}", sys) from e
    log.info("Documents loaded", count=len(docs))
    return docs
