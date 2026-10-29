from __future__ import annotations
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.logger import GLOBAL_LOGGER as log
from backend.exception.custom_exception import DocumentPortalException
from backend.model.model import ChatRequest, ChatResponse, UploadResponse
from backend.src.doc_chat.retrieval import ConversationalRAG
from backend.src.doc_ingest.data_ingest import ChatIngestor
from backend.utils.model_loader import ModelLoader
from backend.utils.tracing import build_langfuse_handler, flush_langfuse
from backend.utils.config_loader import load_config
from backend.utils.file_io import SUPPORTED_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = PROJECT_ROOT / "faiss_index"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES_PER_UPLOAD = 10
MAX_HISTORY_MESSAGES = 20  # 10 user/assistant turns kept per session

SESSION_ID_PATTERN = re.compile(r"^session_\d{8}_\d{6}_[0-9a-f]{8}$")

CONFIG = load_config()
RETRIEVER_CFG = CONFIG.get("retriever", {})
MODEL_LOADER = ModelLoader()
LANGFUSE_HANDLER = build_langfuse_handler(MODEL_LOADER.api_key_mgr)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    flush_langfuse()


app = FastAPI(title="Document Chat RAG", lifespan=lifespan)

# In-memory state: per-session chat history and loaded RAG chains.
_state_lock = threading.Lock()
_ingest_lock = threading.Lock()
_histories: Dict[str, List[BaseMessage]] = {}
_rag_cache: Dict[str, ConversationalRAG] = {}


def _validate_session_id(session_id: str) -> None:
    if not SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")


def _require_session(session_id: str) -> None:
    _validate_session_id(session_id)
    if not (FAISS_DIR / session_id).is_dir():
        raise HTTPException(status_code=404, detail="Session not found. Upload documents first.")


def _get_rag(session_id: str) -> ConversationalRAG:
    with _state_lock:
        rag = _rag_cache.get(session_id)
        if rag is None:
            rag = ConversationalRAG(session_id=session_id, model_loader=MODEL_LOADER)
            rag.load_retriever_from_faiss(
                index_path=str(FAISS_DIR / session_id),
                k=RETRIEVER_CFG.get("top_k", 5),
                search_type=RETRIEVER_CFG.get("search_type", "mmr"),
                fetch_k=RETRIEVER_CFG.get("fetch_k", 20),
                lambda_mult=RETRIEVER_CFG.get("lambda_mult", 0.5),
            )
            _rag_cache[session_id] = rag
        return rag


def _validate_upload(files: List[UploadFile]) -> None:
    if not files or len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status_code=400, detail=f"Upload 1 to {MAX_FILES_PER_UPLOAD} files")
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")
        f.file.seek(0, 2)
        size = f.file.tell()
        f.file.seek(0)
        if size > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large: {f.filename}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
def upload(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
) -> UploadResponse:
    _validate_upload(files)
    if session_id:
        _require_session(session_id)

    try:
        with _ingest_lock:
            ingestor = ChatIngestor(
                temp_base=str(DATA_DIR),
                faiss_base=str(FAISS_DIR),
                use_session_dirs=True,
                session_id=session_id or None,
                model_loader=MODEL_LOADER,
            )
            ingestor.built_retriver(
                files,
                chunk_size=1000,
                chunk_overlap=200,
                k=RETRIEVER_CFG.get("top_k", 5),
                search_type=RETRIEVER_CFG.get("search_type", "mmr"),
                fetch_k=RETRIEVER_CFG.get("fetch_k", 20),
                lambda_mult=RETRIEVER_CFG.get("lambda_mult", 0.5),
            )
    except DocumentPortalException as e:
        log.error("Upload indexing failed", error=str(e))
        raise HTTPException(
            status_code=422,
            detail="Could not index the uploaded documents. Check that they contain extractable text.",
        )

    # The cached chain holds the old index in memory, so drop it.
    with _state_lock:
        _rag_cache.pop(ingestor.session_id, None)

    log.info("Documents indexed", session_id=ingestor.session_id, files=len(files))
    return UploadResponse(
        session_id=ingestor.session_id,
        indexed=True,
        message=f"Indexed {len(files)} file(s)",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    _require_session(req.session_id)
    rag = _get_rag(req.session_id)

    with _state_lock:
        history = list(_histories.get(req.session_id, []))

    try:
        answer = rag.invoke(
            req.message,
            chat_history=history,
            callbacks=[LANGFUSE_HANDLER] if LANGFUSE_HANDLER else [],
        )
    except DocumentPortalException as e:
        log.error("Chat failed", session_id=req.session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate an answer")

    with _state_lock:
        msgs = _histories.setdefault(req.session_id, [])
        msgs.extend([HumanMessage(content=req.message), AIMessage(content=answer)])
        del msgs[:-MAX_HISTORY_MESSAGES]

    return ChatResponse(answer=answer)


# Mounted last so API routes take precedence over the static frontend.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
