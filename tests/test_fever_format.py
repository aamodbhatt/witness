from __future__ import annotations

import json

from witness.wiki.fever_format import (
    decode_title,
    detokenize,
    encode_title,
    parse_lines,
    parse_page,
)


def test_title_codec_roundtrip() -> None:
    raw = "Savages_-LRB-2012_film-RRB-"
    assert decode_title(raw) == "Savages (2012 film)"
    assert encode_title("Savages (2012 film)") == raw
    assert decode_title("Star_Wars-COLON-_Episode_IV") == "Star Wars: Episode IV"


def test_detokenize_ptb_text() -> None:
    raw = (
        "Nikolaj Coster-Waldau -LRB- born 27 July 1970 -RRB- is a Danish actor , "
        "known for `` Game of Thrones '' and the Rockets ' 1981 team ."
    )
    assert detokenize(raw) == (
        "Nikolaj Coster-Waldau (born 27 July 1970) is a Danish actor, "
        'known for "Game of Thrones" and the Rockets\' 1981 team.'
    )


def test_parse_lines_keeps_sentence_numbers_and_links() -> None:
    field = (
        "0\tJean is a French saxophonist .\tjazz\tjazz\tParis\tParis\n"
        "1\t\n2\t\n3\tHe played with Django Reinhardt .\tDjango Reinhardt\tDjango Reinhardt"
    )
    lines = parse_lines(field)
    assert [ln.text for ln in lines] == [
        "Jean is a French saxophonist.",
        "",
        "",
        "He played with Django Reinhardt.",
    ]
    assert lines[0].links == ("jazz", "Paris")
    assert lines[0].anchors == (("jazz", "jazz"), ("Paris", "Paris"))
    assert lines[3].links == ("Django Reinhardt",)


def test_parse_page_skips_empty_pages() -> None:
    assert parse_page(json.dumps({"id": "", "text": "", "lines": ""})) is None
    page = parse_page(json.dumps({"id": "A_-LRB-b-RRB-", "text": "x .", "lines": "0\tx ."}))
    assert page is not None
    assert page.title == "A (b)"
    assert page.lines[0].text == "x."
