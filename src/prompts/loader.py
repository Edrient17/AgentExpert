from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


PROMPT_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=64)
def load_prompt_spec(relative_path: str) -> Dict[str, Any]:
    prompt_path = PROMPT_ROOT / relative_path
    with prompt_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Prompt spec must be a mapping: {prompt_path}")
    return spec


def load_prompt_template(relative_path: str) -> PromptTemplate:
    spec = load_prompt_spec(relative_path)
    template = spec.get("template")
    if not template:
        raise ValueError(f"Prompt spec has no template: {relative_path}")
    return PromptTemplate.from_template(template)


def load_chat_prompt_template(relative_path: str) -> ChatPromptTemplate:
    spec = load_prompt_spec(relative_path)
    messages: List[Dict[str, str]] = spec.get("messages") or []
    if not messages:
        raise ValueError(f"Chat prompt spec has no messages: {relative_path}")
    return ChatPromptTemplate.from_messages(
        [(message["role"], message["template"]) for message in messages]
    )
