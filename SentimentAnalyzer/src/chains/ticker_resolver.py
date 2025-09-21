# src/chains/ticker_resolver.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

# Use an explicit UA if provided; helps avoid anonymous-rate throttling
DEFAULT_HEADERS = {
    "User-Agent": os.getenv("USER_AGENT", "MarketSentimentBot/0.1 (+https://example.com)")
}


# ---------- Data structures ----------
@dataclass
class TickerCandidate:
    symbol: str
    shortname: str
    longname: Optional[str]
    exchange: Optional[str]
    score: float
    typeDisp: Optional[str]
    exchDisp: Optional[str]

    def display_name(self) -> str:
        return self.shortname or self.longname or self.symbol


# ---------- Utilities ----------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# Minimal curated shortcuts (expand as you like)
CURATED = {
    _norm("Apple Inc"): ("AAPL", "Apple Inc.", "NASDAQ"),
    _norm("Alphabet Inc"): ("GOOGL", "Alphabet Inc.", "NASDAQ"),
    _norm("Microsoft Corporation"): ("MSFT", "Microsoft Corporation", "NASDAQ"),
    _norm("Amazon.com, Inc."): ("AMZN", "Amazon.com, Inc.", "NASDAQ"),
    _norm("Tesla, Inc."): ("TSLA", "Tesla, Inc.", "NASDAQ"),
    _norm("Meta Platforms, Inc."): ("META", "Meta Platforms, Inc.", "NASDAQ"),
    _norm("NVIDIA Corporation"): ("NVDA", "NVIDIA Corporation", "NASDAQ"),
    # A few India examples (NSE suffix)
    _norm("Reliance Industries"): ("RELIANCE.NS", "Reliance Industries Limited", "NSE"),
    _norm("Tata Consultancy Services"): ("TCS.NS", "Tata Consultancy Services Limited", "NSE"),
    _norm("Infosys"): ("INFY.NS", "Infosys Limited", "NSE"),
    _norm("HDFC Bank"): ("HDFCBANK.NS", "HDFC Bank Limited", "NSE"),
    _norm("ICICI Bank"): ("ICICIBANK.NS", "ICICI Bank Limited", "NSE"),
}


# ---------- Yahoo search ----------
@retry(
    wait=wait_random_exponential(multiplier=0.5, max=4),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=True,
)
def yahoo_suggest(
    query: str,
    count: int = 8,
    lang: str = "en-US",
    region: str = "US",
    timeout: int = 10,
) -> List[TickerCandidate]:
    """Call Yahoo Finance search API and return a list of TickerCandidate."""
    params = {
        "q": query,
        "quotesCount": count,
        "newsCount": 0,
        "lang": lang,
        "region": region,
    }
    resp = requests.get(YAHOO_SEARCH_URL, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json() or {}
    quotes = data.get("quotes", []) or []
    out: List[TickerCandidate] = []
    for q in quotes:
        out.append(
            TickerCandidate(
                symbol=q.get("symbol", ""),
                shortname=q.get("shortname") or q.get("longname") or "",
                longname=q.get("longname"),
                exchange=q.get("exchange"),
                score=float(q.get("score", 0.0) or 0.0),
                typeDisp=q.get("typeDisp"),
                exchDisp=q.get("exchDisp"),
            )
        )
    return out


def _boost_score(c: TickerCandidate, target_norm: str, prefer_equity: bool) -> float:
    score = c.score or 0.0
    # Prefer equities/common stock-like
    if prefer_equity and str(c.typeDisp).upper() in {"EQUITY", "COMMON STOCK", "ETF"}:
        score += 1.5
    # Name token containment
    combined = _norm(f"{c.shortname} {c.longname or ''}")
    if all(tok in combined for tok in target_norm.split()):
        score += 1.0
    return score


def resolve_ticker(
    company_name: str,
    region: str = "US",
    prefer_equity: bool = True,
) -> Optional[TickerCandidate]:
    """
    Resolve a company name to a likely stock ticker.
    1) curated map
    2) Yahoo Finance search + simple scoring
    Returns best TickerCandidate or None.
    """
    key = _norm(company_name)
    if key in CURATED:
        sym, nm, exch = CURATED[key]
        return TickerCandidate(
            symbol=sym,
            shortname=nm,
            longname=nm,
            exchange=exch,
            score=1.0,
            typeDisp="EQUITY",
            exchDisp=exch,
        )

    # fallback: Yahoo search
    candidates = yahoo_suggest(company_name, count=8, region=region)
    if not candidates:
        return None

    target = _norm(company_name)
    best = max(candidates, key=lambda c: _boost_score(c, target, prefer_equity))
    return best
