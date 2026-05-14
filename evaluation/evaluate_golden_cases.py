from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

from api.services.ruleset_service import execute_all_rulesets, load_execution_rulesets


DEFAULT_DATASET_PATH = Path("data/golden_dataset/golden_cases.csv")
DEFAULT_RULESET_DIR = Path("data/ruleset")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/golden_eval_results.csv")
DEFAULT_SUMMARY_OUTPUT_PATH = Path("evaluation/results/golden_eval_summary.json")
DEFAULT_JUDGE_MODEL = "gpt-4o"


JUDGE_SYSTEM_PROMPT = """
You are a careful clinical guideline evaluation judge.
Your job is to compare a system output against a reference answer for a
medical decision-support golden case.

Return only JSON with:
{
  "verdict": "correct" | "partial" | "incorrect",
  "score": 1.0 | 0.5 | 0.0,
  "rationale": "short explanation"
}

Judging rules:
- Mark correct when the system output contains all mandatory components from
  the reference and does not contradict restrictions.
- When the reference allows alternatives with "ou" / "or", any valid
  alternative is sufficient.
- Mark partial when the output is clinically related but incomplete.
- Mark incorrect when it omits an essential recommendation, recommends a
  prohibited item, contradicts the reference, or is unrelated.
- Judge only against the supplied reference answer, not against outside
  clinical knowledge.
""".strip()


def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Evaluate executable ruleset outputs against golden cases using LLM-as-a-judge."
    )

    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--ruleset-dir", type=Path, default=DEFAULT_RULESET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument(
        "--judge-model",
        default=os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help="OpenAI model used as judge. Defaults to EVAL_JUDGE_MODEL or gpt-4o.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of golden cases to evaluate.",
    )

    return parser


def load_cases(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:

    with path.open(newline="", encoding="utf-8") as file_obj:

        reader = csv.DictReader(file_obj)
        cases = list(reader)

    print('---'*10)
    print('reader.fieldnames: ', reader.fieldnames)
    reader.fieldnames[0] = 'id'
    print('reader.fieldnames: ', reader.fieldnames)
    print('---'*10)


    required_columns = {"id", "parametros_clinicos", "resposta_correta"}
    missing_columns = required_columns - set(reader.fieldnames or [])

    if missing_columns:

        missing = ", ".join(sorted(missing_columns))

        raise ValueError(f"Dataset is missing required columns: {missing}")

    if limit is not None:

        cases = cases[:limit]

    return cases


def parse_parameters(raw_parameters: str) -> dict[str, Any]:

    payload = json.loads(raw_parameters)

    if not isinstance(payload, dict):

        raise ValueError("parametros_clinicos must be a JSON object.")
    
    return payload


def build_ruleset_executor():

    try:

        from markdown_vector_indexer.pipeline import RulesetPipeline

    except Exception:

        return LocalRulesetExecutor()
    
    return object.__new__(RulesetPipeline)


class LocalRulesetExecutor:

    @staticmethod
    def _normalize_context(context: dict[str, Any]) -> dict[str, Any]:
        return {
            key.lower() if isinstance(key, str) else key: (
                value.lower() if isinstance(value, str) else value
            )
            for key, value in context.items()
        }


    def evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:

        if not isinstance(condition, str) or not condition.strip():

            raise ValueError("Condition must be a non-empty string.")

        result = eval(condition, {"__builtins__": {}}, {"context": context})

        if not isinstance(result, bool):

            raise ValueError(f"Condition did not return a boolean: {condition}")
        
        return result


    def solve(self, ruleset: dict[str, Any], query: dict[str, Any], max_steps: int = 100):

        if not isinstance(ruleset, dict):

            raise ValueError("`ruleset` must be a dict.")

        flow_id = ruleset.get("flow_id")
        start_node = ruleset.get("start_node")
        nodes = ruleset.get("nodes")

        if not flow_id:

            raise ValueError("Ruleset must contain `flow_id`.")
        
        if not start_node:

            raise ValueError("Ruleset must contain `start_node`.")
        
        if not isinstance(nodes, dict) or not nodes:

            raise ValueError("Ruleset must contain non-empty `nodes` dict.")

        query = self._normalize_context(query)
        current_node_id = start_node
        trace: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        warnings: list[str] = []

        for step_index in range(max_steps):

            node = nodes.get(current_node_id)

            if node is None:

                raise KeyError(f"Node `{current_node_id}` not found in ruleset.")

            node_type = node.get("type")

            if node_type == "terminal":

                trace.append(
                    {
                        "step": step_index + 1,
                        "node_id": current_node_id,
                        "node_type": "terminal",
                        "result": node.get("result"),
                        "description": node.get("description"),
                    }
                )

                return {
                    "flow_id": flow_id,
                    "final_node": current_node_id,
                    "status": "completed",
                    "trace": trace,
                    "actions": actions,
                    "output": node.get("result"),
                    "warnings": warnings,
                }


            if node_type != "decision":

                raise ValueError(f"Node `{current_node_id}` has invalid type: {node_type!r}.")

            warning = node.get("warning")

            if warning:

                warnings.append(f"{current_node_id}: {warning}")

            condition_result = self.evaluate_condition(node.get("condition"), query)

            branch = "true" if condition_result else "false"

            selected_payload = (
                node.get("action_if_true") if condition_result else node.get("action_if_false")
            )

            next_node_id = (
                selected_payload
                if isinstance(selected_payload, str) and selected_payload in nodes
                else None
            )

            actions.append(
                {
                    "node_id": current_node_id,
                    "branch": branch,
                    "action": selected_payload,
                }
            )

            trace.append(
                {
                    "step": step_index + 1,
                    "node_id": current_node_id,
                    "node_type": "decision",
                    "condition": node.get("condition"),
                    "condition_result": condition_result,
                    "branch_taken": branch,
                    "next_node": next_node_id,
                    "selected_action": selected_payload,
                    "description": node.get("description"),
                }
            )

            if not isinstance(selected_payload, str) or not selected_payload.strip():

                raise ValueError(f"Node `{current_node_id}` did not define output for `{branch}`.")

            if next_node_id is None:
                return {
                    "flow_id": flow_id,
                    "final_node": None,
                    "status": "completed",
                    "trace": trace,
                    "actions": actions,
                    "output": selected_payload,
                    "warnings": warnings,
                }

            current_node_id = next_node_id

        raise RuntimeError(f"Execution stopped after {max_steps} steps. Possible cycle.")


def execute_case(
    executor: Any,
    rulesets: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    
    return execute_all_rulesets(rulesets, parameters, pipeline=executor)


def build_judge_user_prompt(
    *,
    case_id: str,
    parameters: dict[str, Any],
    reference_output: str,
    system_output: str,
    execution: dict[str, Any],
) -> str:
    
    payload = {
        "case_id": case_id,
        "parametros_clinicos": parameters,
        "resposta_correta_referencia": reference_output,
        "system_output": system_output,
        "system_actions": execution.get("actions", []),
        "system_trace": execution.get("trace", []),
        "system_warnings": execution.get("warnings", []),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def judge_output(
    *,
    client: OpenAI,
    model: str,
    case_id: str,
    parameters: dict[str, Any],
    reference_output: str,
    system_output: str,
    execution: dict[str, Any],
) -> dict[str, Any]:
    
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_judge_user_prompt(
                    case_id=case_id,
                    parameters=parameters,
                    reference_output=reference_output,
                    system_output=system_output,
                    execution=execution,
                ),
            },
        ],
    )

    content = completion.choices[0].message.content or "{}"

    return normalize_judge_payload(json.loads(content))


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:

    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"correct", "partial", "incorrect"}:
        verdict = "incorrect"

    expected_scores = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    score = payload.get("score", expected_scores[verdict])
    
    try:
    
        score = float(score)
    
    except (TypeError, ValueError):
    
    
        score = expected_scores[verdict]

    if verdict == "correct":
    
        score = 1.0
    
    elif verdict == "partial":
    
        score = 0.5
    
    else:
    
        score = 0.0

    rationale = str(payload.get("rationale", "")).strip()
    
    return {"verdict": verdict, "score": score, "rationale": rationale}


def evaluate_cases(
    *,
    cases: list[dict[str, str]],
    rulesets: list[dict[str, Any]],
    judge_model: str,
) -> list[dict[str, Any]]:
    client = OpenAI()
    executor = build_ruleset_executor()
    rows: list[dict[str, Any]] = []

    for case in cases:

        case_id = str(case.get("id", "")).strip()
        raw_parameters = case.get("parametros_clinicos", "")
        reference_output = case.get("resposta_correta", "")
        execution: dict[str, Any] = {}
        execution_error = ""
        system_output = ""
        parameters: dict[str, Any] = {}

        try:

            parameters = parse_parameters(raw_parameters)
            execution = execute_case(executor, rulesets, parameters)
            system_output = str(execution.get("output", "") or "")

        except Exception as exc:

            execution_error = str(exc)

        if execution_error:

            judge_payload = {
                "verdict": "incorrect",
                "score": 0.0,
                "rationale": f"Ruleset execution failed: {execution_error}",
            }

        else:

            judge_payload = judge_output(
                client=client,
                model=judge_model,
                case_id=case_id,
                parameters=parameters,
                reference_output=reference_output,
                system_output=system_output,
                execution=execution,
            )

        rows.append(
            {
                "id": case_id,
                "parametros_clinicos": json.dumps(parameters, ensure_ascii=False),
                "resposta_correta": reference_output,
                "system_output": system_output,
                "verdict": judge_payload["verdict"],
                "score": judge_payload["score"],
                "rationale": judge_payload["rationale"],
                "raw_execution_status": execution.get("status", "error"),
                "warnings": json.dumps(execution.get("warnings", []), ensure_ascii=False),
                "execution_error": execution_error,
            }
        )

    return rows


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "parametros_clinicos",
        "resposta_correta",
        "system_output",
        "verdict",
        "score",
        "rationale",
        "raw_execution_status",
        "warnings",
        "execution_error",
    ]

    with path.open("w", newline="", encoding="utf-8") as file_obj:

        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    rows: list[dict[str, Any]],
    dataset_path: Path,
    ruleset_dir: Path,
    ruleset_count: int,
    output_path: Path,
    summary_output_path: Path,
    judge_model: str,
) -> dict[str, Any]:
    
    total = len(rows)
    verdict_counts = Counter(str(row["verdict"]) for row in rows)
    score_sum = sum(float(row["score"]) for row in rows)

    return {
        "total_cases": total,
        "verdict_counts": dict(verdict_counts),
        "accuracy_strict": verdict_counts.get("correct", 0) / total if total else 0.0,
        "partial_rate": verdict_counts.get("partial", 0) / total if total else 0.0,
        "mean_score": score_sum / total if total else 0.0,
        "dataset_path": str(dataset_path),
        "ruleset_dir": str(ruleset_dir),
        "ruleset_count": ruleset_count,
        "output_path": str(output_path),
        "summary_output_path": str(summary_output_path),
        "judge_model": judge_model,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:

    load_dotenv()

    args = build_argument_parser().parse_args(argv)

    rulesets = load_execution_rulesets(args.ruleset_dir)
    cases = load_cases(args.dataset, limit=args.limit)
    rows = evaluate_cases(cases=cases, rulesets=rulesets, judge_model=args.judge_model)

    write_results(args.output, rows)

    summary = build_summary(
        rows=rows,
        dataset_path=args.dataset,
        ruleset_dir=args.ruleset_dir,
        ruleset_count=len(rulesets),
        output_path=args.output,
        summary_output_path=args.summary_output,
        judge_model=args.judge_model,
    )

    write_summary(args.summary_output, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    
    print('Evaluating golden cases...')

    raise SystemExit(main())

    print('Evaluation completed.')
