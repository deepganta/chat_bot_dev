"""Dealer-offer enrichment helpers."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path


log = logging.getLogger(__name__)


def fetch_dealer_offers(
    zip_code: str,
    model: str,
    db_path: Path = Path("data/ford.db"),
    limit: int = 3,
) -> list[dict]:
    """Return nearby active dealer offers for the provided ZIP and model."""

    zip_value = (zip_code or "").strip()
    model_value = (model or "").strip()
    if not zip_value or not model_value or limit <= 0:
        return []

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        query = """
            SELECT
                d.name AS dealer_name,
                z.city AS city,
                z.state AS state,
                dl.distance_miles AS distance_miles,
                o.offer_type AS offer_type,
                o.amount AS amount,
                o.expiry_date AS expiry_date,
                o.model AS model
            FROM dealer_locations AS dl
            JOIN dealers AS d
                ON d.id = dl.dealer_id
            JOIN offers AS o
                ON o.dealer_id = d.id
            JOIN zip_codes AS z
                ON z.zip = dl.zip
            WHERE z.zip = ?
              AND lower(o.model) = lower(?)
              AND o.expiry_date >= date('now')
            ORDER BY dl.distance_miles ASC
            LIMIT ?
        """
        rows = conn.execute(query, (zip_value, model_value, int(limit))).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        log.warning(
            "Dealer enrichment failed for zip=%s model=%s: %s",
            zip_value,
            model_value,
            exc,
        )
        return []
    finally:
        if conn is not None:
            conn.close()


__all__ = ["fetch_dealer_offers"]
