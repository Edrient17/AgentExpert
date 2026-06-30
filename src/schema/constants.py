from enum import StrEnum
from typing import Literal


class Team(StrEnum):
    QUERY = "team1"
    RETRIEVAL = "team2"
    ANSWER = "team3"
    END = "end"


class NodeName(StrEnum):
    QUERY_WORKER = "team1_worker"
    QUERY_EVALUATOR = "team1_evaluator"
    RAG_SEARCH = "rag_search"
    RAG_SEARCH_RESULT = "rag_search_result"
    WEB_SEARCH = "web_search"
    WEB_SEARCH_RESULT = "web_search_result"
    RETRIEVAL_EVALUATOR = "team2_evaluator"
    ANSWER_WORKER = "team3_worker"
    ANSWER_EVALUATOR = "team3_evaluator"


class WorkflowSignal(StrEnum):
    PASS = "pass"
    RETRY = "retry"
    FAIL = "fail"
    RETRY_RAG = "retry_rag"
    FALLBACK_TO_WEB = "fallback_to_web"
    RETRY_WEB = "retry_web"


class RetrievalSource(StrEnum):
    RAG = "rag"
    WEB = "web"


class SimpleQuery(StrEnum):
    YES = "Yes"
    NO = "No"


TeamName = Literal["team1", "team2", "team3", "end"]
SimpleQueryValue = Literal["Yes", "No"]


def retry_signal(reason: str = "") -> str:
    if reason:
        return f"{WorkflowSignal.RETRY}: {reason}"
    return WorkflowSignal.RETRY


def fail_signal(reason: str) -> str:
    return f"{WorkflowSignal.FAIL}: {reason}"


def is_retry_signal(content: str) -> bool:
    return str(content).startswith(WorkflowSignal.RETRY)


def is_fail_signal(content: str) -> bool:
    return str(content).startswith(WorkflowSignal.FAIL)
