from typing import List, Optional
from pydantic import BaseModel, Field


class LLMResponseBase(BaseModel):
    """
    Base class for all structured LLM task responses.
    """

    status: str = "success"
    message: Optional[str] = None
    processing_time: Optional[str] = None


class SummarizationResponse(LLMResponseBase):
    summary: str


class TranslationResponse(LLMResponseBase):
    original_text: str
    translated_text: str
    target_language: str


class KeywordExtractionResponse(BaseModel):
    keywords: List[str]
    processing_time_sec: float
    status: str = "success"
    error: Optional[str] = None


class TopicGenerationResponse(LLMResponseBase):
    topics: List[str]


class TextAlignmentResponse(LLMResponseBase):
    alignment: str  # Freeform structured string from the model
