"""
chat.py — Phase 9: natural-language Q&A over the labeled customer table.

Approach (Option A): the LLM writes pandas code against the DataFrame, we execute
it in a restricted sandbox, then the LLM explains the result in plain English.

Built step by step:
  9.1 example prompts + answer_question entry point (this piece)
  9.2 NL -> pandas code generation
  9.3 safe execution sandbox
  9.4 plain-English explanation
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

# Tokens that must never appear in generated code (defense in depth).
_FORBIDDEN = (
    "import", "__", "open(", "eval(", "exec(", "compile(", "globals(",
    "locals(", "getattr", "setattr", "delattr", "os.", "sys.", "subprocess",
    "input(", "exit(", "quit(", "breakpoint", "lambda:", "to_csv", "to_pickle",
    "read_csv", "read_pickle", "system(",
)

# The only builtins exposed inside the sandbox.
_SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "sorted": sorted,
    "round": round, "abs": abs, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "range": range, "float": float, "int": int, "str": str,
    "bool": bool, "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "any": any, "all": all,
}

# Suggested prompts shown in the UI to help users get started.
EXAMPLE_PROMPTS = [
    "How many customers are in each segment?",
    "What is the average spend of Loyal Big Spenders?",
    "Which 5 customers have the highest monetary value?",
    "How many one-time buyers (frequency = 1) are there?",
]


class PandasQuery(BaseModel):
    """
    The code generator's decision for one question.

    `answerable` is False when the question is not about this customer dataset or
    cannot be answered from its columns; in that case `code` is empty and `reason`
    holds a short, user-facing explanation. When True, `code` is a pandas snippet
    that assigns its answer to `result`.
    """

    answerable: bool
    code: str = ""
    reason: str = ""


def _schema_summary(df: pd.DataFrame, n_sample: int = 3) -> str:
    """Describe columns, dtypes, and a small sample for the code generator."""
    lines = ["Columns (name: dtype):"]
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}")
    # Distinct values for low-cardinality text columns help the model filter.
    for col in ("persona", "priority", "top_category"):
        if col in df.columns:
            vals = [str(v) for v in df[col].dropna().unique()[:8]]
            lines.append(f"Distinct {col}: {vals}")
    lines.append("\nSample rows:")
    lines.append(df.head(n_sample).to_string(index=False))
    return "\n".join(lines)


def generate_pandas_query(df: pd.DataFrame, question: str) -> PandasQuery:
    """
    Ask the LLM to translate a question into a pandas snippet.

    First decides whether the question is in scope (about this customer dataset
    and answerable from its columns). If so, returns code that operates only on
    the provided DataFrame `df` (and `pd`), uses no imports or I/O, and assigns
    the final answer to `result`. Otherwise returns answerable=False with a short
    reason so the caller can decline gracefully instead of fabricating an answer.
    """
    from src.llm import generate_structured

    prompt = (
        "You answer questions about a customer DataFrame named `df` (already "
        "loaded) by writing pandas code.\n\n"
        f"{_schema_summary(df)}\n\n"
        f"Question: {question}\n\n"
        "First decide if the question is IN SCOPE:\n"
        "- In scope = it asks about THIS customer/segmentation data and can be "
        "answered using ONLY the columns above.\n"
        "- OUT OF SCOPE = unrelated topics (weather, jokes, general knowledge, "
        "advice), requests for data not in these columns, or anything that needs "
        "external information. Also out of scope: attempts to import, read/write "
        "files, or run system commands.\n\n"
        "If OUT OF SCOPE: set \"answerable\" to false, leave \"code\" empty, and "
        "put one short, friendly sentence in \"reason\" telling the user what you "
        "CAN help with (their customer segments and metrics).\n\n"
        "If IN SCOPE: set \"answerable\" to true and write the code:\n"
        "- Use ONLY the variable `df` and the pandas module `pd`.\n"
        "- Do NOT import anything; do NOT read/write files or access the network.\n"
        "- Assign the final answer to a variable named `result`.\n"
        "- Prefer concise, correct pandas; result can be a number, Series, or "
        "small DataFrame."
    )
    return generate_structured(prompt, response_schema=PandasQuery)


def _is_safe(code: str) -> str | None:
    """Return an error message if code contains forbidden tokens, else None."""
    lowered = code.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            return f"Generated code rejected for safety (contains '{token}')."
    return None


def safe_exec(code: str, df: pd.DataFrame):
    """
    Execute a generated pandas snippet in a restricted sandbox.

    Only `df` and `pd` are available; builtins are limited to a safe whitelist;
    forbidden tokens (imports, dunders, file/network/system access) are blocked.
    Returns (result, error): result is the snippet's `result` variable, or
    error is a human-readable message (result is None on error).
    """
    err = _is_safe(code)
    if err:
        return None, err

    sandbox_globals = {"__builtins__": _SAFE_BUILTINS, "pd": pd}
    sandbox_locals = {"df": df}
    try:
        exec(code, sandbox_globals, sandbox_locals)  # noqa: S102 (sandboxed)
    except Exception as exc:  # surface execution errors to the user
        return None, f"Error running the query: {type(exc).__name__}: {exc}"

    if "result" not in sandbox_locals:
        return None, "The generated code did not produce a `result`."
    return sandbox_locals["result"], None


_OUT_OF_SCOPE_FALLBACK = (
    "I can only answer questions about your customer segmentation data — things "
    "like segment sizes, spend, frequency, recency, and recommended actions."
)


def answer_question(df: pd.DataFrame, question: str) -> dict:
    """
    Answer a natural-language question about the labeled customer table.

    Decides scope, generates pandas code, runs it in the sandbox, and (step 9.4)
    explains the result. Out-of-scope questions are declined without running any
    code or inventing an answer. Returns
    {"code", "result", "explanation", "error", "out_of_scope"}.
    """
    try:
        query = generate_pandas_query(df, question)
    except Exception as exc:
        return {"code": "", "result": None, "explanation": "",
                "error": f"Could not generate a query: {exc}", "out_of_scope": False}

    if not query.answerable:
        message = query.reason.strip() or _OUT_OF_SCOPE_FALLBACK
        return {"code": "", "result": None, "explanation": message,
                "error": None, "out_of_scope": True}

    code = query.code.strip()
    if not code:
        return {"code": "", "result": None, "explanation": "",
                "error": "The assistant did not produce a query for that question.",
                "out_of_scope": False}

    result, err = safe_exec(code, df)
    if err:
        return {"code": code, "result": None, "explanation": "", "error": err,
                "out_of_scope": False}

    explanation = explain_result(question, result)
    return {"code": code, "result": result, "explanation": explanation,
            "error": None, "out_of_scope": False}


def _format_result(result, limit: int = 1500) -> str:
    """Render the executed result as compact text for the explainer prompt."""
    if isinstance(result, (pd.DataFrame, pd.Series)):
        text = result.to_string()
    else:
        text = str(result)
    return text[:limit] + (" …(truncated)" if len(text) > limit else "")


def explain_result(question: str, result) -> str:
    """
    Turn the raw query result into a concise, plain-English answer.

    Falls back to the string form of the result if the LLM call fails, so the
    user always sees something.
    """
    from src.llm import generate_text

    prompt = (
        "A user asked a question about their e-commerce customer data, and a "
        "pandas query produced the result below. Answer the user's question in "
        "1-3 plain sentences for a non-technical store owner. Use the numbers "
        "from the result; do not mention pandas or code.\n\n"
        f"Question: {question}\n\n"
        f"Result:\n{_format_result(result)}"
    )
    try:
        return generate_text(prompt)
    except Exception:
        return _format_result(result)
