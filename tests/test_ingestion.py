from src.ingestion.pdf_parser import as_document, deduplicate_documents


def test_deduplicate_documents_uses_source_page_and_content():
    first = as_document("same", "source.pdf", page=1)
    duplicate = as_document("same", "source.pdf", page=1)
    different_page = as_document("same", "source.pdf", page=2)

    docs = deduplicate_documents([first, duplicate, different_page])

    assert docs == [first, different_page]
