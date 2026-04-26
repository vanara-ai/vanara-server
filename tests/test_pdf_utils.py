"""Tests for ``app.pdf_utils`` Unicode sanitization.

These functions exist to prevent xhtml2pdf from rendering solid black
boxes when the LLM emits Unicode typography (em-dashes, smart quotes,
bullets, etc.) that reportlab's built-in fonts cannot render.

This is a regression guard: without these, any resume containing typical
LLM-generated text will render with black block glyphs in the final PDF.
"""

from app.pdf_utils import _sanitize_for_pdf, _sanitize_pdf_payload


class TestSanitizeForPdf:
    def test_em_dash_becomes_ascii_dash(self):
        assert _sanitize_for_pdf("Led team \u2014 shipped MVP") == "Led team - shipped MVP"

    def test_en_dash_becomes_ascii_dash(self):
        assert _sanitize_for_pdf("2023\u20132024") == "2023-2024"

    def test_non_breaking_hyphen_becomes_ascii_dash(self):
        # U+2011 was the specific bug we hit in Vanara.ai PDF gen.
        assert _sanitize_for_pdf("full\u2011stack") == "full-stack"

    def test_smart_double_quotes_become_ascii(self):
        assert _sanitize_for_pdf("\u201chello\u201d") == '"hello"'

    def test_smart_single_quotes_become_ascii(self):
        assert _sanitize_for_pdf("\u2018world\u2019") == "\u0027world\u0027"

    def test_bullet_becomes_asterisk(self):
        assert _sanitize_for_pdf("\u2022 item one") == "* item one"

    def test_ellipsis_becomes_three_dots(self):
        assert _sanitize_for_pdf("wait\u2026") == "wait..."

    def test_arrow_becomes_ascii(self):
        assert _sanitize_for_pdf("A \u2192 B") == "A -> B"

    def test_nbsp_becomes_space(self):
        assert _sanitize_for_pdf("a\u00a0b") == "a b"

    def test_zero_width_space_dropped(self):
        assert _sanitize_for_pdf("a\u200bb") == "ab"

    def test_bom_dropped(self):
        assert _sanitize_for_pdf("\ufefftext") == "text"

    def test_accented_latin_transliterated(self):
        # NFKD normalization strips accents: Jose -> Jose, cafe -> cafe.
        assert _sanitize_for_pdf("Jos\u00e9") == "Jose"
        assert _sanitize_for_pdf("caf\u00e9") == "cafe"

    def test_del_control_char_dropped(self):
        # U+007F (DEL) renders as a black block in reportlab.
        assert _sanitize_for_pdf("a\x7fb") == "ab"

    def test_other_c0_controls_dropped(self):
        # Drop \x01 but preserve tab, newline, carriage return.
        assert _sanitize_for_pdf("a\x01b") == "ab"
        assert _sanitize_for_pdf("a\tb") == "a\tb"
        assert _sanitize_for_pdf("a\nb") == "a\nb"

    def test_plain_ascii_untouched(self):
        s = "The quick brown fox jumps over 123 lazy dogs."
        assert _sanitize_for_pdf(s) == s

    def test_non_string_passthrough(self):
        # Non-string input returned as-is (defensive).
        assert _sanitize_for_pdf(42) == 42
        assert _sanitize_for_pdf(None) is None

    def test_empty_string(self):
        assert _sanitize_for_pdf("") == ""


class TestSanitizePdfPayload:
    def test_dict_recursion(self):
        payload = {"name": "Jos\u00e9", "title": "SWE \u2014 Lead"}
        result = _sanitize_pdf_payload(payload)
        assert result == {"name": "Jose", "title": "SWE - Lead"}

    def test_nested_list_in_dict(self):
        payload = {"bullets": ["\u2022 built X", "\u2022 shipped Y"]}
        result = _sanitize_pdf_payload(payload)
        assert result == {"bullets": ["* built X", "* shipped Y"]}

    def test_deeply_nested(self):
        payload = {
            "experience": [
                {"company": "ACME \u2014 Inc.", "bullets": ["\u2022 did thing"]},
            ],
        }
        result = _sanitize_pdf_payload(payload)
        assert result["experience"][0]["company"] == "ACME - Inc."
        assert result["experience"][0]["bullets"] == ["* did thing"]

    def test_tuple_preserved_as_tuple(self):
        payload = ("a\u2014b", "c\u2014d")
        result = _sanitize_pdf_payload(payload)
        assert result == ("a-b", "c-d")
        assert isinstance(result, tuple)

    def test_non_string_leaf_untouched(self):
        payload = {"score": 42, "passed": True, "meta": None}
        assert _sanitize_pdf_payload(payload) == {"score": 42, "passed": True, "meta": None}

    def test_empty_structures(self):
        assert _sanitize_pdf_payload({}) == {}
        assert _sanitize_pdf_payload([]) == []
        assert _sanitize_pdf_payload("") == ""
