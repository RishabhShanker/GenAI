# src/chains/news_fetcher.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# Optional: LangChain tool (for a human-readable snapshot only)
try:
    from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
    _HAS_LC_TOOL = True
except Exception:
    _HAS_LC_TOOL = False


@dataclass
class NewsItem:
    title: str
    link: str
    publisher: Optional[str]
    published_at: str  # ISO 8601
    related_tickers: List[str]


def _to_iso(dt: datetime | None) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _parse_epoch_to_dt(v: Union[int, float, str, None]) -> Optional[datetime]:
    if v is None:
        return None
    try:
        # handle "1699999999" as str too
        secs = int(float(v))
        return datetime.fromtimestamp(secs, tz=timezone.utc)
    except Exception:
        return None


def _parse_iso_to_dt(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    try:
        # Accept e.g. "2025-09-15T14:40:39Z"
        v = v.replace("Z", "+00:00") if v.endswith("Z") else v
        return datetime.fromisoformat(v).astimezone(timezone.utc)
    except Exception:
        return None


def _extract_published_dt(n: Dict[str, Any]) -> Optional[datetime]:
    """
    Handle multiple Yahoo news shapes:
    - n['providerPublishTime'] -> epoch seconds
    - n['content']['pubDate'] -> ISO string
    - (rare) n['timePublished'] -> epoch seconds
    """
    dt = None
    dt = dt or _parse_epoch_to_dt(n.get("providerPublishTime"))
    if dt:
        return dt

    content = n.get("content") or {}
    dt = dt or _parse_iso_to_dt(content.get("pubDate"))
    if dt:
        return dt

    dt = dt or _parse_epoch_to_dt(n.get("timePublished"))
    return dt


def _within_lookback(dt: Optional[datetime], lookback_days: int) -> bool:
    if not dt:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    return dt >= cutoff


def _extract_link(n: Dict[str, Any]) -> str:
    # Preferred order: top-level 'link', content.canonicalUrl.url, content.clickThroughUrl.url
    link = n.get("link")
    if link:
        return link
    content = n.get("content") or {}
    canon = (content.get("canonicalUrl") or {}).get("url")
    if canon:
        return canon
    click = (content.get("clickThroughUrl") or {}).get("url")
    if click:
        return click
    return ""


def _extract_publisher(n: Dict[str, Any]) -> Optional[str]:
    pub = n.get("publisher")
    if pub:
        return pub
    content = n.get("content") or {}
    prov = content.get("provider") or {}
    return prov.get("displayName")


@retry(
    wait=wait_random_exponential(multiplier=0.5, max=4),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def fetch_recent_news(
    ticker: str,
    lookback_days: int = 7,
    top_k: int = 5,
) -> List[NewsItem]:
    """
    Fetch recent company news from Yahoo Finance via yfinance, filter to last `lookback_days`,
    sort by recency, and return top_k.
    """
    tk = yf.Ticker(ticker)
    raw_news = tk.news or []  # list of dicts

    items: List[NewsItem] = []
    for n in raw_news:
        title = (n.get("title") or (n.get("content") or {}).get("title") or "").strip()
        link = _extract_link(n)
        publisher = _extract_publisher(n)
        dt = _extract_published_dt(n)
        ts_iso = _to_iso(dt)
        related = n.get("relatedTickers") or []

        if not title or not link:
            continue
        if not _within_lookback(dt, lookback_days):
            continue

        items.append(
            NewsItem(
                title=title,
                link=link,
                publisher=publisher,
                published_at=ts_iso or "",
                related_tickers=related,
            )
        )

    # Sort newest first, then truncate
    items.sort(key=lambda x: x.published_at, reverse=True)
    return items[: max(0, top_k)]


def to_bulleted_newsdesc(items: List[NewsItem]) -> str:
    lines = []
    for it in items:
        date_short = it.published_at[:10] if it.published_at else ""
        pub = f" [{it.publisher}]" if it.publisher else ""
        lines.append(f"- {date_short}{pub}: {it.title} ({it.link})")
    return "\n".join(lines)


def optional_tool_snapshot(ticker: str) -> Optional[str]:
    if not _HAS_LC_TOOL:
        return None
    try:
        tool = YahooFinanceNewsTool()
        return tool.run(ticker)
    except Exception:
        return None


def as_dict_list(items: List[NewsItem]) -> List[Dict[str, Any]]:
    return [asdict(x) for x in items]
