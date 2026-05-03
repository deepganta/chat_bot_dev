"""Response composition utilities for retrieval outputs."""

from __future__ import annotations

from typing import Iterable


def _format_amount(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"{int(number)}"
    return f"{number:.2f}"


def format_response(
    main_answer: str,
    dealer_offers: list[dict],
    suggestions: list[str],
) -> str:
    """Compose final assistant response with optional offers and suggestions."""

    sections: list[str] = [main_answer.strip() or "I'm not sure how to answer that yet."]

    if dealer_offers:
        lines = ["Nearby Dealers with Offers:"]
        for offer in dealer_offers:
            dealer_name = offer.get("dealer_name", "Unknown Dealer")
            city = offer.get("city", "Unknown City")
            state = offer.get("state", "Unknown State")
            distance = float(offer.get("distance_miles", 0.0))
            offer_type = str(offer.get("offer_type", "offer"))
            amount = _format_amount(offer.get("amount", "0"))
            expiry = str(offer.get("expiry_date", "unknown"))
            model = str(offer.get("model", "")).strip()
            model_suffix = f" off {model}" if model else ""
            lines.append(
                f"- {dealer_name} - {city}, {state} ({distance:.1f} mi) — "
                f"{offer_type}: ${amount}{model_suffix} (expires {expiry})"
            )
        sections.append("\n".join(lines))

    cleaned_suggestions = [s.strip() for s in suggestions if s and s.strip()]
    if cleaned_suggestions:
        lines = ["You might also ask:"]
        lines.extend(f"- {suggestion}" for suggestion in cleaned_suggestions)
        sections.append("\n".join(lines))

    return "\n\n".join(section for section in sections if section.strip())


__all__ = ["format_response"]
