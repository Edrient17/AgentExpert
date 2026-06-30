from pathlib import Path

from src.prompts.loader import PROMPT_ROOT, load_chat_prompt_template, load_prompt_spec, load_prompt_template


def test_all_yaml_prompt_specs_are_valid_mappings():
    prompt_files = sorted(Path(PROMPT_ROOT).rglob("*.yaml"))
    assert prompt_files

    for prompt_file in prompt_files:
        relative_path = prompt_file.relative_to(PROMPT_ROOT).as_posix()
        spec = load_prompt_spec(relative_path)
        assert spec["id"]
        assert spec["version"]
        assert spec.get("template") or spec.get("messages")


def test_standard_prompt_template_loads():
    prompt = load_prompt_template("answer/evaluator.yaml")
    assert "generated_answer" in prompt.input_variables
    assert "is_simple_query" in prompt.input_variables


def test_chat_prompt_template_loads():
    prompt = load_chat_prompt_template("query/simple_query_classifier.yaml")
    assert "question" in prompt.input_variables
