import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from backend.evaluation.dataset import EvalItem, load_dataset
from backend.logger import GLOBAL_LOGGER as log
from backend.src.doc_chat.retrieval import ConversationalRAG
from backend.utils.model_loader import ModelLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAISS_DIR = PROJECT_ROOT / "faiss_index"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SESSION_ID_PATTERN = re.compile(r"^session_\d{8}_\d{6}_[0-9a-f]{8}$")
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]


def build_judge(model_loader: ModelLoader) -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Ragas judge models. Kept separate from the production LLM via the `evaluation` config section."""
    eval_cfg = model_loader.config["evaluation"]
    api_key = model_loader.api_key_mgr.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model=eval_cfg["llm_model_name"], temperature=0, api_key=api_key)
    embeddings = OpenAIEmbeddings(model=eval_cfg["embedding_model_name"], api_key=api_key)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def collect_samples(rag: ConversationalRAG, items: List[EvalItem]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Run each question through the existing RAG pipeline. Failed questions are recorded, not scored."""
    samples: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for item in items:
        try:
            answer, contexts = rag.invoke_with_contexts(item.question)
        except Exception as e:
            log.error("Evaluation question failed", question=item.question, error=str(e))
            errors.append({"question": item.question, "error": str(e)})
            continue
        samples.append(
            {
                "user_input": item.question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item.reference,
                "reference_contexts": item.reference_contexts,
            }
        )
    return samples, errors


def _clean(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def score_samples(samples: List[Dict[str, Any]], judge_llm, judge_embeddings) -> Tuple[Dict[str, Optional[float]], List[Dict[str, Any]]]:
    """Run the Ragas metrics and return (aggregate scores, per-question scores)."""
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()

    aggregate = {name: _clean(df[name].mean()) for name in METRIC_NAMES if name in df.columns}
    per_question = [
        {
            "question": row["user_input"],
            "scores": {name: _clean(df.iloc[i][name]) for name in METRIC_NAMES if name in df.columns},
        }
        for i, row in enumerate(samples)
    ]
    return aggregate, per_question


def run(session_id: str, dataset_path: Path, output_path: Optional[Path] = None) -> Path:
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError("Invalid session id")
    index_path = FAISS_DIR / session_id
    if not index_path.is_dir():
        raise FileNotFoundError(f"No index for session {session_id}. Upload documents for it first.")

    items = load_dataset(dataset_path)
    model_loader = ModelLoader()
    retriever_cfg = model_loader.config.get("retriever", {})

    rag = ConversationalRAG(session_id=session_id, model_loader=model_loader)
    rag.load_retriever_from_faiss(
        index_path=str(index_path),
        k=retriever_cfg.get("top_k", 5),
        search_type=retriever_cfg.get("search_type", "mmr"),
        fetch_k=retriever_cfg.get("fetch_k", 20),
        lambda_mult=retriever_cfg.get("lambda_mult", 0.5),
    )

    samples, errors = collect_samples(rag, items)
    aggregate, per_question = ({}, [])
    if samples:
        judge_llm, judge_embeddings = build_judge(model_loader)
        aggregate, per_question = score_samples(samples, judge_llm, judge_embeddings)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "session_id": session_id,
        "dataset": str(dataset_path),
        "generated_at": timestamp,
        "judge_llm": model_loader.config["evaluation"]["llm_model_name"],
        "judge_embedding_model": model_loader.config["evaluation"]["embedding_model_name"],
        "questions_total": len(items),
        "questions_scored": len(samples),
        "aggregate": aggregate,
        "per_question": per_question,
        "errors": errors,
    }

    output_path = output_path or RESULTS_DIR / f"eval_{session_id}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Evaluation finished", output=str(output_path), aggregate=aggregate)
    return output_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on one document collection (session).")
    parser.add_argument("--session-id", required=True, help="Session id whose FAISS index to evaluate")
    parser.add_argument("--dataset", required=True, type=Path, help="JSON file with question/reference/reference_contexts")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: backend/evaluation/results)")
    args = parser.parse_args(argv)

    try:
        output = run(args.session_id, args.dataset, args.output)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    print(f"Evaluation report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
