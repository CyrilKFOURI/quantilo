"""M01: the fixture the frontend checks its translations against is the engine's current lists."""

from app import engine_texts_dump


def test_the_fixture_is_up_to_date() -> None:
    written = engine_texts_dump.TARGET.read_text(encoding="utf-8")
    assert written == engine_texts_dump.text(), (
        "Run `python -m app.engine_texts_dump` and add the texts it brings to "
        "frontend/src/lib/i18n/catalog.json and fr/catalog.json (reasons: reasons.json and "
        "fr/reasons.json, M02)."
    )
