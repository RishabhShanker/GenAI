# src/chains/sentiment_chain.py
from __future__ import annotations

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import get_gemini_chat_model
from src.models.sentiment_schema import SentimentResult
from src.prompts.sentiment_prompt import SYSTEM_PROMPT, HUMAN_TEMPLATE


def build_prompt(company_name: str, stock_code: str, newsdesc: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=HUMAN_TEMPLATE.format(
            company_name=company_name,
            stock_code=stock_code,
            newsdesc=newsdesc if newsdesc.strip() else "(no recent articles found)"
        )),
    ])


def analyze_sentiment(company_name: str, stock_code: str, newsdesc: str) -> Dict[str, Any]:
    """
    Uses Gemini 2.0 Flash (LangChain wrapper) with structured output to return a dict
    conforming to SentimentResult. If newsdesc is empty, returns a Neutral fallback.
    """
    if not newsdesc.strip():
        # Fallback (no LLM call): neutral with low confidence
        fallback = SentimentResult(
            company_name=company_name,
            stock_code=stock_code,
            newsdesc="(no recent articles found)",
            sentiment="Neutral",
            market_implications="Insufficient recent coverage; no clear directional signal.",
            confidence_score=0.2,
        )
        return fallback.model_dump()

    llm = get_gemini_chat_model(model="gemini-2.0-flash")
    # Ask LangChain to enforce the Pydantic schema directly
    structured_llm = llm.with_structured_output(SentimentResult)

    prompt = build_prompt(company_name, stock_code, newsdesc)
    result: SentimentResult = (prompt | structured_llm).invoke({})

    # Ensure a plain dict is returned (useful for JSON serialization & MLflow logging)
    return result.model_dump()
