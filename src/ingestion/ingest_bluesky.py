"""
ingest_bluesky.py — Poll Bluesky public API → Bronze Delta table.

Searches for posts related to: news, politics, misinformation, fact check.
Unlabeled live data — label is set to "unknown".
Post AT URI is used as the row id.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import settings
from src.ingestion.ingest_static import _make_record, _merge_to_bronze

_SEARCH_QUERIES = ["news", "politics", "misinformation", "fact check"]


def _fetch_posts(client: Any, query: str, limit: int) -> list[dict[str, Any]]:
    response = client.app.bsky.feed.search_posts(
        params={"q": query, "limit": min(limit, 100)}
    )
    posts = response.posts or []
    records = []
    for post in posts:
        uri = getattr(post, "uri", "") or ""
        if not uri:
            continue

        record_obj = getattr(post, "record", None)
        if isinstance(record_obj, dict):
            text = record_obj.get("text", "")
        else:
            text = getattr(record_obj, "text", "") or ""

        author = getattr(post, "author", None)
        if isinstance(author, dict):
            author_did = author.get("did", "")
            author_handle = author.get("handle", "")
        else:
            author_did = getattr(author, "did", "") if author else ""
            author_handle = getattr(author, "handle", "") if author else ""

        meta = {
            "cid": getattr(post, "cid", ""),
            "author_did": author_did,
            "author_handle": author_handle,
            "indexed_at": str(getattr(post, "indexed_at", "")),
            "reply_count": getattr(post, "reply_count", 0),
            "repost_count": getattr(post, "repost_count", 0),
            "like_count": getattr(post, "like_count", 0),
            "query": query,
        }
        records.append(
            _make_record(
                row_id=uri,
                text=text,
                label="unknown",
                source="bluesky",
                meta=meta,
            )
        )
    return records


def run(limit: int = 100) -> None:
    """Entry point — fetch Bluesky posts and append to Bronze Delta table."""
    path = settings.delta_path("bronze")
    logger.info(f"Bluesky ingestion starting | limit={limit} | target={path}")

    if not settings.bluesky_handle or not settings.bluesky_app_password:
        logger.warning("BLUESKY_HANDLE or BLUESKY_APP_PASSWORD not configured — skipping")
        return

    # Lazy import so atproto is only required when this module is actually used
    from atproto import Client

    client = Client()
    client.login(settings.bluesky_handle, settings.bluesky_app_password)

    per_query = max(1, limit // len(_SEARCH_QUERIES))
    all_records: list[dict[str, Any]] = []
    for query in _SEARCH_QUERIES:
        logger.info(f"Searching Bluesky for: {query!r}")
        try:
            records = _fetch_posts(client, query, per_query)
            all_records.extend(records)
            logger.info(f"  {len(records)} posts fetched")
        except Exception as exc:
            logger.warning(f"Bluesky search failed for {query!r}: {exc}")

    logger.info(f"Total Bluesky posts to ingest: {len(all_records)}")
    if all_records:
        _merge_to_bronze(all_records, path)
    logger.info("Bluesky ingestion complete")
