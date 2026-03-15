"""LLM-powered claim extraction from natural language text.

This module provides tools for extracting structured claims from unstructured text
using Large Language Models. The LLM-based approach provides:

- Automatic claim type classification (semantic, episodic)
- Context-aware extraction
- Confidence scoring
- Evidence extraction
Requires: pip install re-collect[llm]

Example:
    from re_collect.extractors import LLMExtractor
    from re_collect.llm.providers import OpenAIProvider

    llm = OpenAIProvider(api_key="...")
    extractor = LLMExtractor(llm)
    claims = extractor.extract("I love pizza on Fridays")
"""

from re_collect.extractors.base import ClaimExtractor
from re_collect.extractors.llm import LLMExtractor

__all__ = [
    "ClaimExtractor",
    "LLMExtractor",
]
