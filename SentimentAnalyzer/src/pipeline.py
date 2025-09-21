# src/pipeline.py
from __future__ import annotations
from typing import Any, Dict

from src.config import get_settings
from src.chains.ticker_resolver import resolve_ticker
from src.chains.news_fetcher import (
    fetch_recent_news,
    to_bulleted_newsdesc,
    as_dict_list,
    optional_tool_snapshot,
)
from src.chains.sentiment_chain import analyze_sentiment
from src.prompts.sentiment_prompt import SYSTEM_PROMPT, HUMAN_TEMPLATE
from src.observability.mlflow_utils import init_mlflow, span, log_text, log_dict

def run_pipeline(company_name: str) -> Dict[str, Any]:
    """
    Orchestrates: name -> ticker -> news -> sentiment (Gemini 2.0 Flash).
    Logs params, artifacts, and metrics to MLflow, and returns final JSON dict.
    """
    import mlflow  # local import to avoid any shadowing
    s = get_settings()
    init_mlflow(experiment="market-sentiment-analyzer")

    run_name = f"analyze:{company_name}"
    with mlflow.start_run(run_name=run_name):
        # --- 1) Ticker resolution ---
        with span("ticker", {"company_name": company_name}):
            cand = resolve_ticker(company_name)
            if cand:
                ticker = cand.symbol
                mlflow.log_params({
                    "ticker.symbol": cand.symbol,
                    "ticker.exchange": cand.exchDisp or (cand.exchange or ""),
                    "ticker.display_name": cand.display_name(),
                })
            else:
                ticker = company_name
                mlflow.log_params({"ticker.symbol": ticker, "ticker.exchange": ""})

        # --- 2) News fetch ---
        with span("news", {"ticker": ticker, "lookback_days": s.news_lookback_days, "top_k": s.news_top_k}):
            items = fetch_recent_news(ticker, lookback_days=s.news_lookback_days, top_k=s.news_top_k)
            newsdesc = to_bulleted_newsdesc(items)
            log_text("news/newsdesc.txt", newsdesc or "(no recent articles found)")
            log_dict("news/news_items.json", {"ticker": ticker, "items": as_dict_list(items)})

            snapshot = optional_tool_snapshot(ticker)
            if snapshot:
                log_text("news/yahoo_tool_snapshot.txt", snapshot)

        # --- 3) Sentiment (Gemini 2.0 Flash + structured output) ---
        with span("sentiment", {"model": "gemini-2.0-flash"}):
            prompt_text = (
                SYSTEM_PROMPT
                + "\n\n"
                + HUMAN_TEMPLATE.format(
                    company_name=company_name,
                    stock_code=ticker,
                    newsdesc=newsdesc if newsdesc.strip() else "(no recent articles found)",
                )
            )
            log_text("sentiment/prompt.txt", prompt_text)

            result = analyze_sentiment(company_name=company_name, stock_code=ticker, newsdesc=newsdesc)
            log_dict("sentiment/sentiment.json", result)

            try:
                mlflow.log_metric("confidence_score", float(result.get("confidence_score", 0.0)))
            except Exception:
                pass

        return result
    

    
