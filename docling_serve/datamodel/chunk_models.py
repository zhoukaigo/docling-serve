# docling_serve/datamodel/chunk_models.py
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field


class FileItemChunk(BaseModel):
    content: Annotated[
        Optional[str],
        Field(description="The content of the file item")
    ] = None

    tokens: Annotated[
        Optional[int],
        Field(description="The tokens of the content")
    ] = None

class Chunks(BaseModel):
    chunks: Annotated[
        List[FileItemChunk],
        Field(
            default_factory=list,
            description="The chunks of the document"
        )
    ]

    error: Annotated[
        Optional[str],
        Field(description="The error that occurred during the conversion to chunks")
    ] = None
