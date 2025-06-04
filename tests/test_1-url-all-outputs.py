import json

import httpx
import pytest
import pytest_asyncio
from pytest_check import check


@pytest_asyncio.fixture
async def async_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


@pytest.mark.asyncio
async def test_convert_url(async_client):
    """Test convert URL to all outputs"""
    url = "http://localhost:5001/v1alpha/convert/source"
    payload = {
        "options": {
            "from_formats": [
                "docx",
                "pptx",
                "html",
                "image",
                "pdf",
                "asciidoc",
                "md",
                "xlsx",
            ],
            "to_formats": ["md", "json", "html", "text", "doctags"],
            "image_export_mode": "placeholder",
            "ocr": True,
            "force_ocr": False,
            "ocr_engine": "easyocr",
            "ocr_lang": ["en"],
            "pdf_backend": "dlparse_v2",
            "table_mode": "fast",
            "abort_on_error": False,
            "return_as_file": False,
        },
        "http_sources": [{"url": "https://arxiv.org/pdf/2206.01062"}],
    }
    print(json.dumps(payload, indent=2))

    response = await async_client.post(url, json=payload)
    assert response.status_code == 200, "Response should be 200 OK"

    data = response.json()

    # Response content checks
    # Helper function to safely slice strings
    def safe_slice(value, length=100):
        if isinstance(value, str):
            return value[:length]
        return str(value)  # Convert non-string values to string for debug purposes

    # Document check
    check.is_in(
        "document",
        data,
        msg=f"Response should contain 'document' key. Received keys: {list(data.keys())}",
    )
    # MD check
    check.is_in(
        "md_content",
        data.get("document", {}),
        msg=f"Response should contain 'md_content' key. Received keys: {list(data.get('document', {}).keys())}",
    )
    if data.get("document", {}).get("md_content") is not None:
        check.is_in(
            "## DocLayNet: ",
            data["document"]["md_content"],
            msg=f"Markdown document should contain 'DocLayNet: '. Received: {safe_slice(data['document']['md_content'])}",
        )
    # JSON check
    check.is_in(
        "json_content",
        data.get("document", {}),
        msg=f"Response should contain 'json_content' key. Received keys: {list(data.get('document', {}).keys())}",
    )
    if data.get("document", {}).get("json_content") is not None:
        check.is_in(
            '{"schema_name": "DoclingDocument"',
            json.dumps(data["document"]["json_content"]),
            msg=f'JSON document should contain \'{{\\n  "schema_name": "DoclingDocument\'". Received: {safe_slice(data["document"]["json_content"])}',
        )
    # HTML check
    check.is_in(
        "html_content",
        data.get("document", {}),
        msg=f"Response should contain 'html_content' key. Received keys: {list(data.get('document', {}).keys())}",
    )
    if data.get("document", {}).get("html_content") is not None:
        check.is_in(
            "<!DOCTYPE html>\n<html>\n<head>",
            data["document"]["html_content"],
            msg=f"HTML document should contain '<!DOCTYPE html>\\n<html>'. Received: {safe_slice(data['document']['html_content'])}",
        )
    # Text check
    check.is_in(
        "text_content",
        data.get("document", {}),
        msg=f"Response should contain 'text_content' key. Received keys: {list(data.get('document', {}).keys())}",
    )
    if data.get("document", {}).get("text_content") is not None:
        check.is_in(
            "DocLayNet: A Large Human-Annotated Dataset",
            data["document"]["text_content"],
            msg=f"Text document should contain 'DocLayNet: A Large Human-Annotated Dataset'. Received: {safe_slice(data['document']['text_content'])}",
        )
    # DocTags check
    check.is_in(
        "doctags_content",
        data.get("document", {}),
        msg=f"Response should contain 'doctags_content' key. Received keys: {list(data.get('document', {}).keys())}",
    )
    if data.get("document", {}).get("doctags_content") is not None:
        check.is_in(
            "<doctag><page_header><loc",
            data["document"]["doctags_content"],
            msg=f"DocTags document should contain '<doctag><page_header><loc'. Received: {safe_slice(data['document']['doctags_content'])}",
        )


@pytest.mark.asyncio
async def test_convert_url_with_markdown_chunking(async_client: httpx.AsyncClient):
    """Test convert URL to markdown with chunking enabled."""
    url_endpoint = "http://localhost:5001/v1alpha/convert/source"
    payload = {
        "options": {
            "to_formats": ["md"], # Focus on markdown
            "do_markdown_chunking": True,
            "markdown_chunking_config": {
                "max_tokens": 100,
                "overlap_tokens": 10,
                "min_chunk_tokens": 5,
                "encoding_name": "cl100k_base"
            },
            "pdf_backend": "dlparse_v2", # Keep other relevant options consistent
            "return_as_file": False,
        },
        "http_sources": [{"url": "https://arxiv.org/pdf/2206.01062"}], # Same PDF as file test
    }

    response = await async_client.post(url_endpoint, json=payload)
    check.equal(response.status_code, 200, f"Response should be 200 OK. Response: {response.text}")

    response_json = response.json()

    # Document check
    check.is_in("document", response_json)
    document_data = response_json.get("document", {})

    # MD content check
    check.is_in("md_content", document_data)
    check.is_not_none(document_data.get("md_content"), "Markdown content should exist.")
    if document_data.get("md_content"):
        check.is_in("## DocLayNet:", document_data["md_content"], "Actual md_content: " + document_data["md_content"][:200])


    # MD chunks check
    check.is_in("md_chunks", document_data)
    check.is_not_none(document_data.get("md_chunks"), "md_chunks should exist.")
    check.is_instance(document_data.get("md_chunks"), list, "md_chunks should be a list.")

    md_chunks_list = document_data.get("md_chunks", [])
    if not md_chunks_list:
        check.is_true(len(md_chunks_list) > 0, "md_chunks list should not be empty for this document with these chunking settings.")
    else:
        check.greater(len(md_chunks_list), 1, "Expected multiple chunks for the given PDF and chunking config.")
        for i, chunk in enumerate(md_chunks_list):
            check.is_in("content", chunk, f"Chunk {i} should have 'content' key.")
            check.is_in("tokens", chunk, f"Chunk {i} should have 'tokens' key.")
            check.is_not_none(chunk.get("content"), f"Chunk {i} content should not be None.")
            check.is_instance(chunk.get("content"), str, f"Chunk {i} content should be a string.")
            check.greater_equal(chunk.get("tokens"), 0, f"Chunk {i} tokens should be >= 0.")
            if chunk.get("tokens", 0) > 0:
                check.less_equal(chunk.get("tokens"), 100 + 10 + 5, # max_tokens + overlap_tokens + buffer
                                 f"Chunk {i} tokens {chunk.get('tokens')} seems too large for max_tokens=100, overlap=10.")

    # Check that other formats are not present if "to_formats" was only ["md"]
    check.is_none(document_data.get("json_content"), "JSON content should be None if not requested.")
    check.is_none(document_data.get("html_content"), "HTML content should be None if not requested.")
    check.is_none(document_data.get("text_content"), "Text content should be None if not requested.")
    check.is_none(document_data.get("doctags_content"), "DocTags content should be None if not requested.")
