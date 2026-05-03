"""Harness de evals para o pipeline RAG.

Executa um dataset de perguntas, coleta respostas do pipeline e avalia
com LLM-as-judge (relevância, faithfulness, completeness).

Uso:
    python -m evals.evaluate

Variáveis de ambiente necessárias:
    OPENAI_API_KEY   — para o pipeline RAG e para o juiz LLM

Variáveis opcionais (observabilidade):
    LANGSMITH_API_KEY + LANGCHAIN_TRACING_V2=true — envia traces ao LangSmith
    LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY     — envia traces ao Langfuse
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

# ── Dataset ───────────────────────────────────────────────────────────────


def load_dataset() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── LLM-as-judge ─────────────────────────────────────────────────────────


def llm_as_judge(
    question: str,
    answer: str,
    expected_themes: str | None = None,
) -> dict:
    """Avalia a resposta em três dimensões com um LLM juiz.

    Dimensões:
        relevance    — A resposta endereça a pergunta? (1–5)
        faithfulness — A resposta é baseada no contexto, sem alucinações? (1–5)
        completeness — A resposta cobre os pontos esperados? (1–5)

    Args:
        question: Pergunta do dataset.
        answer: Resposta gerada pelo pipeline RAG.
        expected_themes: Temas esperados na resposta (opcional).

    Returns:
        Dict com scores e reasoning.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    expected_section = (
        f"\nExpected themes: {expected_themes}" if expected_themes else ""
    )

    prompt = f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Rate the following answer on three dimensions using a 1-5 integer scale.

Question: {question}
Answer: {answer}{expected_section}

Scoring guide:
  1 = Very poor  |  2 = Poor  |  3 = Acceptable  |  4 = Good  |  5 = Excellent

Dimensions:
  relevance    — Does the answer directly address the question?
  faithfulness — Is every claim grounded in the retrieved context (no hallucinations)?
  completeness — Does the answer cover the expected themes without major gaps?

Respond with valid JSON only:
{{"relevance": <int>, "faithfulness": <int>, "completeness": <int>, "reasoning": "<one sentence>"}}"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "relevance": 0,
            "faithfulness": 0,
            "completeness": 0,
            "reasoning": response.content,
        }


# ── Eval runner ───────────────────────────────────────────────────────────


def run_evals(data_path: str = "data/sample_docs.txt") -> list[dict]:
    """Executa o dataset completo e retorna resultados com scores.

    Args:
        data_path: Corpus a indexar (mesmo usado no pipeline RAG).

    Returns:
        Lista de resultados com question, answer e scores.
    """
    from app import build_rag_graph  # lazy import — evita indexar na importação

    dataset = load_dataset()
    rag = build_rag_graph(data_path)
    results = []

    print(f"▶ Rodando {len(dataset)} evals...\n")

    for sample in dataset:
        initial_state = {
            "query": sample["question"],
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
        }
        result = rag.invoke(initial_state)
        scores = llm_as_judge(
            sample["question"],
            result["answer"],
            sample.get("expected_themes"),
        )

        entry = {
            "id": sample.get("id", "?"),
            "question": sample["question"],
            "answer": result["answer"],
            "sources_used": len(result["reranked_docs"]),
            "scores": scores,
        }
        results.append(entry)

        rel = scores["relevance"]
        faith = scores["faithfulness"]
        comp = scores["completeness"]
        print(f"[{entry['id']}] R:{rel} F:{faith} C:{comp} — {scores['reasoning']}")

    # Aggregate metrics
    rel_scores = [r["scores"]["relevance"] for r in results]
    faith_scores = [r["scores"]["faithfulness"] for r in results]
    comp_scores = [r["scores"]["completeness"] for r in results]

    summary = {
        "n": len(results),
        "avg_relevance": round(statistics.mean(rel_scores), 2),
        "avg_faithfulness": round(statistics.mean(faith_scores), 2),
        "avg_completeness": round(statistics.mean(comp_scores), 2),
    }

    print(f"\n📊 Resumo ({summary['n']} perguntas)")
    print(f"   Relevance:    {summary['avg_relevance']:.2f} / 5")
    print(f"   Faithfulness: {summary['avg_faithfulness']:.2f} / 5")
    print(f"   Completeness: {summary['avg_completeness']:.2f} / 5")

    output = {"summary": summary, "results": results}
    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n💾 Resultados salvos em {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("⚠️  Defina OPENAI_API_KEY no arquivo .env")
    run_evals()
