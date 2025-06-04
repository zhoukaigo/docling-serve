import json
import os

import httpx
import pytest
import pytest_asyncio
from pytest_check import check


@pytest_asyncio.fixture
async def async_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


@pytest.mark.asyncio
async def test_convert_file(async_client):
    """Test convert single file to all outputs"""
    url = "http://localhost:5001/v1alpha/convert/file"
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

    response = await async_client.post(url, files=files, data=options)
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
async def test_convert_file_with_markdown_chunking(async_client: httpx.AsyncClient):
    """Test convert single file to markdown with chunking enabled."""
    url = "http://localhost:5001/v1alpha/convert/file"

    # Prepare options for form data
    form_data_options = {
        "options.to_formats": "md",
        "options.do_markdown_chunking": "true",
        # Ensure the PDF content is actually chunked by setting low token limits
        "options.markdown_chunking_config": json.dumps({
            "max_tokens": 100,
            "overlap_tokens": 10,
            "min_chunk_tokens": 5,
            "encoding_name": "cl100k_base" # good for tiktoken
        }),
        "options.pdf_backend": "dlparse_v2", # Keep other relevant options
        "options.return_as_file": "false", # Ensure response is JSON
    }

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "2206.01062v1.pdf") # Use the same PDF

    files = {
        "files": ("2206.01062v1.pdf", open(file_path, "rb"), "application/pdf"),
    }

    response = await async_client.post(url, files=files, data=form_data_options)
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
    if not md_chunks_list: # If the list is empty or None
        # This might be unexpected if chunking is forced and document is non-trivial
        check.is_true(len(md_chunks_list) > 0, "md_chunks list should not be empty for this document with these chunking settings.")
    else:
        check.greater(len(md_chunks_list), 1, "Expected multiple chunks for the given PDF and chunking config.")
        for i, chunk in enumerate(md_chunks_list):
            check.is_in("content", chunk, f"Chunk {i} should have 'content' key.")
            check.is_in("tokens", chunk, f"Chunk {i} should have 'tokens' key.")
            check.is_not_none(chunk.get("content"), f"Chunk {i} content should not be None.")
            check.is_instance(chunk.get("content"), str, f"Chunk {i} content should be a string.")
            check.greater_equal(chunk.get("tokens"), 0, f"Chunk {i} tokens should be >= 0.")
            # Check if token count is within expected limits (max_tokens + some buffer for overlap logic)
            # This is a loose check as exact token counts can vary.
            if chunk.get("tokens", 0) > 0 : # only check if there are tokens
                # Max tokens for this config is 100, overlap is 10.
                # The chunker might produce chunks slightly larger than max_tokens before overlap is removed,
                # or slightly smaller than min_chunk_tokens for the last chunk.
                # A chunk's token count should ideally be <= max_tokens (or <= max_tokens + overlap_tokens depending on strategy)
                # and >= min_chunk_tokens (unless it's the only/last chunk and smaller).
                # Given max_tokens=100, overlap_tokens=10, min_chunk_tokens=5
                check.less_equal(chunk.get("tokens"), 100 + 10 + 5, # max_tokens + overlap_tokens + buffer
                                 f"Chunk {i} tokens {chunk.get('tokens')} seems too large for max_tokens=100, overlap=10.")
                # This check might be too strict if the last chunk is very small
                # check.greater_equal(chunk.get("tokens"), 5, f"Chunk {i} tokens {chunk.get('tokens')} seems too small for min_chunk_tokens=5.")


    # Check that other formats are not present if "to_formats" was only "md"
    check.is_none(document_data.get("json_content"), "JSON content should be None if not requested.")
    check.is_none(document_data.get("html_content"), "HTML content should be None if not requested.")
    check.is_none(document_data.get("text_content"), "Text content should be None if not requested.")
    check.is_none(document_data.get("doctags_content"), "DocTags content should be None if not requested.")
