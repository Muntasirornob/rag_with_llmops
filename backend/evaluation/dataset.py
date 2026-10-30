import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class EvalItem:
    question: str
    reference: str
    reference_contexts: List[str]


def _item_problems(index: int, row: dict) -> List[str]:
    problems = []
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        problems.append(f"item {index}: missing 'question'")
    if not isinstance(row.get("reference"), str) or not row["reference"].strip():
        problems.append(f"item {index}: missing 'reference'")
    contexts = row.get("reference_contexts")
    if not isinstance(contexts, list) or not contexts or not all(
        isinstance(c, str) and c.strip() for c in contexts
    ):
        problems.append(f"item {index}: 'reference_contexts' must be a non-empty list of strings")
    return problems


def load_dataset(path: Path) -> List[EvalItem]:
    """
    Load and validate an evaluation dataset (a JSON list).

    Each item needs: question (str), reference (str), reference_contexts (non-empty list of str).
    Extra keys such as "_note" are ignored. All problems are reported together.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty JSON list")

    problems: List[str] = []
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            problems.append(f"item {index}: must be a JSON object")
            continue
        problems.extend(_item_problems(index, row))
    if problems:
        raise ValueError("Invalid dataset:\n" + "\n".join(problems))

    return [
        EvalItem(
            question=row["question"],
            reference=row["reference"],
            reference_contexts=row["reference_contexts"],
        )
        for row in data
    ]
