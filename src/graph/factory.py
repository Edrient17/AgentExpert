import functools

from src.graph.supervisor import create_supervisor_graph
from src.graph.teams.answer_team import create_answer_team_graph
from src.graph.teams.query_team import create_query_team_graph
from src.graph.teams.retrieval_team import create_retrieval_team_graph


@functools.lru_cache(maxsize=1)
def get_graph_app():
    print("[start] Initializing multi-agent RAG system...")

    query_team_app = create_query_team_graph()
    retrieval_team_app = create_retrieval_team_graph()
    answer_team_app = create_answer_team_graph()
    supervisor_app = create_supervisor_graph(query_team_app, retrieval_team_app, answer_team_app)

    print("[ok] System ready.")
    return supervisor_app
