import asyncio
import json
import os

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from pytest_check import check

from docling_serve.app import create_app


@pytest.fixture(scope="session")
def event_loop():
    return asyncio.get_event_loop()


@pytest_asyncio.fixture(scope="session")
async def app():
    app = create_app()

    async with LifespanManager(app) as manager:
        print("Launching lifespan of app.")
        yield manager.app


@pytest_asyncio.fixture(scope="session")
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://app.io"
    ) as client:
        print("Client is ready")
        yield client


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_convert_file(client: AsyncClient):
    """Test convert single file to all outputs"""

    endpoint = "/v1alpha/convert/file"
    options = {
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
    }

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "2206.01062v1.pdf")

    files = {
        "files": ("2206.01062v1.pdf", open(file_path, "rb"), "application/pdf"),
    }

    response = await client.post(endpoint, files=files, data=options)
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
            msg=f"HTML document should contain '<!DOCTYPE html>\n<html>\n<head>'. Received: {safe_slice(data['document']['html_content'])}",
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
            "<doctag><page_header>",
            data["document"]["doctags_content"],
            msg=f"DocTags document should contain '<doctag><page_header>'. Received: {safe_slice(data['document']['doctags_content'])}",
        )


@pytest.mark.asyncio
async def test_chunk_markdown_basic(client: AsyncClient):
    markdown_text = "# Title\\n\\nThis is a paragraph.\\n\\n- Item 1\\n- Item 2"
    request_data = {"markdown_text": markdown_text}
    response = await client.post("/v1alpha/chunk/markdown", json=request_data)
    assert response.status_code == 200
    response_json = response.json()
    check.is_in("chunks", response_json)
    check.is_in("statistics", response_json)
    check.is_none(response_json["error"])
    check.is_true(len(response_json.get("chunks", [])) > 0, "Chunks list should not be empty")
    if response_json.get("chunks"):
        check.is_not_none(response_json["chunks"][0].get("content"))
        check.is_true(response_json["chunks"][0].get("tokens", 0) > 0, "Tokens should be greater than 0")
    if response_json.get("statistics"):
        check.equal(response_json["statistics"].get("total_chunks"), len(response_json.get("chunks", [])))

@pytest.mark.asyncio
async def test_chunk_markdown_with_config(client: AsyncClient):
    markdown_text = "This is a single line of text that should not be split if max_tokens is high enough."
    chunking_config_dict = {
        "max_tokens": 1000,
        "overlap_tokens": 10,
        "min_chunk_tokens": 5,
        "prefer_semantic_boundaries": True,
        "preserve_code_blocks": True,
        "preserve_tables": True,
        "preserve_lists": True,
        "overlap_strategy": "semantic",
        "encoding_name": "cl100k_base"
    }
    request_data = {
        "markdown_text": markdown_text,
        "config": chunking_config_dict
    }
    response = await client.post("/v1alpha/chunk/markdown", json=request_data)
    assert response.status_code == 200
    response_json = response.json()
    check.is_in("chunks", response_json)
    check.is_none(response_json.get("error"))
    chunks = response_json.get("chunks", [])
    check.equal(len(chunks), 1, "Expecting one chunk due to high max_tokens")
    if len(chunks) == 1:
        check.equal(chunks[0].get("content"), markdown_text)

@pytest.mark.asyncio
async def test_chunk_markdown_empty_input(client: AsyncClient):
    request_data = {"markdown_text": ""}
    response = await client.post("/v1alpha/chunk/markdown", json=request_data)
    assert response.status_code == 200
    response_json = response.json()
    check.is_in("chunks", response_json)
    check.is_none(response_json.get("error"))
    check.equal(len(response_json.get("chunks", [])), 0)
    if response_json.get("statistics"):
        check.equal(response_json["statistics"].get("total_chunks"), 0)

@pytest.mark.asyncio
async def test_chunk_markdown_with_minimal_config(client: AsyncClient):
    markdown_text = "Short text."
    request_data = {
        "markdown_text": markdown_text,
        "config": {} # Empty config, should use defaults
    }
    response = await client.post("/v1alpha/chunk/markdown", json=request_data)
    assert response.status_code == 200
    response_json = response.json()
    check.is_in("chunks", response_json)
    check.is_none(response_json.get("error"))
    check.is_true(len(response_json.get("chunks", [])) > 0, "Chunks list should not be empty for non-empty text")

@pytest.mark.asyncio
async def test_chunk_markdown_causes_splitting(client: AsyncClient):
    long_word = "longword"
    markdown_text = (long_word + " ") * 300 # Should be roughly 300 words/tokens

    chunking_config_dict = {
        "max_tokens": 50, # Force splitting
        "overlap_tokens": 5,
        "min_chunk_tokens": 10,
        "encoding_name": "cl100k_base" # Ensure tiktoken can count
    }
    request_data = {
        "markdown_text": markdown_text,
        "config": chunking_config_dict
    }
    response = await client.post("/v1alpha/chunk/markdown", json=request_data)
    assert response.status_code == 200
    response_json = response.json()
    check.is_in("chunks", response_json)
    check.is_none(response_json.get("error"))
    chunks = response_json.get("chunks", [])
    check.is_true(len(chunks) > 1, "Expecting multiple chunks due to forced splitting")
    if response_json.get("statistics"):
        check.equal(response_json["statistics"].get("total_chunks"), len(chunks))
        check.is_true(response_json["statistics"].get("total_chunks") > 1)

    for chunk in chunks:
        check.is_true(len(chunk.get("content", "")) > 0)
        check.is_true(chunk.get("tokens", 0) > 0)
        # This check can be tricky due to how overlap is handled and if the last chunk is smaller.
        # A simple check is that tokens are not excessively over max_tokens.
        # The chunker aims for max_tokens BEFORE overlap is added back for some strategies.
        # And min_chunk_tokens is also a factor.
        check.is_true(chunk.get("tokens", 0) <= chunking_config_dict["max_tokens"] + chunking_config_dict["overlap_tokens"] + 5, # Adding a small buffer for tokenization variance
                        f"Chunk tokens {chunk.get('tokens')} vs max_tokens {chunking_config_dict['max_tokens']}")
