from src.graph.factory import get_graph_app


def test_graph_app_builds():
    get_graph_app.cache_clear()
    app = get_graph_app()
    assert app is not None
