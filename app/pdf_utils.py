import contextlib
import logging

import PyPDF2
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


def extract_pdf_hyperlinks(pdf_path: str):
    links = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    annot_obj = annot.get_object()
                    if annot_obj.get("/Subtype") == "/Link" and "/A" in annot_obj:
                        action = annot_obj["/A"]
                        uri = action.get("/URI")
                        if hasattr(uri, "get_object"):
                            uri = uri.get_object()
                        if uri:
                            rect = annot_obj.get("/Rect")
                            links.append({"page": page_num + 1, "uri": str(uri), "rect": rect})
    return links


def extract_pdf_text_with_pypdf2(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    # Extract hyperlinks and append them to the text
    links = extract_pdf_hyperlinks(pdf_path)
    if links:
        text += "\n\nHyperlinks found in the document:\n"
        for link in links:
            text += f"- Page {link['page']}: {link['uri']}\n"
    return text


def _extraction_looks_broken(text: str) -> bool:
    """Detect when PyPDF2 fails to extract spaces (certain font encodings).

    Normal resumes have ~12% spaces. Broken extraction drops below 1%.
    Threshold of 3% gives a wide safety margin.
    """
    if not text or len(text.strip()) < 100:
        return True
    space_pct = text.count(" ") / len(text) * 100
    return space_pct < 3.0


def _extract_pdf_text_with_pymupdf(pdf_path: str) -> str:
    """Fallback extractor using pymupdf (handles font encodings PyPDF2 can't)."""
    import fitz

    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF, with automatic fallback.

    Uses PyPDF2 as primary extractor. If the output looks broken (e.g. no
    spaces due to font encoding issues), falls back to pymupdf.
    """
    text = extract_pdf_text_with_pypdf2(pdf_path)
    if _extraction_looks_broken(text):
        logger.warning(
            "PyPDF2 output looks broken (space_pct=%.1f%%), falling back to pymupdf",
            text.count(" ") / max(len(text), 1) * 100,
        )
        text = _extract_pdf_text_with_pymupdf(pdf_path)
    return text


# Character-sanitization for xhtml2pdf output.
#
# xhtml2pdf renders via reportlab's built-in fonts (Helvetica, Times-Roman,
# Courier, Symbol), which only cover basic ASCII. When the LLM emits Unicode
# typography (em-dashes, smart quotes, bullets, arrows, etc.) reportlab has
# no glyph for them and renders solid black boxes in the PDF.
#
# This sanitizer normalizes common typography to ASCII equivalents and
# transliterates accented Latin characters. It is applied recursively to
# every string in the resume dict right before rendering.

_PDF_SAFE_REPLACEMENTS = {
    # Dashes / hyphens
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2043": "-",  # hyphen bullet
    "\u207b": "-",  # superscript minus
    "\u208b": "-",  # subscript minus
    "\u2212": "-",  # minus sign
    "\ufe58": "-",  # small em dash
    "\ufe63": "-",  # small hyphen-minus
    "\uff0d": "-",  # fullwidth hyphen-minus
    # Quotes
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote (also apostrophe)
    "\u201a": ",",  # single low-9 quote
    "\u201e": '"',  # double low-9 quote
    "\u00ab": '"',  # left guillemet
    "\u00bb": '"',  # right guillemet
    "\u2032": "'",  # prime
    "\u2033": '"',  # double prime
    # Bullets / separators
    "\u2022": "*",  # bullet
    "\u25cf": "*",  # black circle
    "\u25e6": "o",  # white bullet
    "\u2023": ">",  # triangular bullet
    "\u00b7": ".",  # middle dot
    # Ellipsis
    "\u2026": "...",
    # Arrows
    "\u2192": "->",
    "\u2190": "<-",
    "\u21d2": "=>",
    "\u21d0": "<=",
    # Math / misc
    "\u00d7": "x",  # multiplication sign
    "\u00f7": "/",  # division sign
    "\u00b1": "+/-",  # plus-minus
    "\u2260": "!=",  # not equal
    "\u2264": "<=",
    "\u2265": ">=",
    "\u221e": "inf",
    # Spaces / zero-width
    "\u00a0": " ",  # non-breaking space
    "\u2009": " ",  # thin space
    "\u200b": "",  # zero-width space
    "\u200c": "",  # zero-width non-joiner
    "\u200d": "",  # zero-width joiner
    "\ufeff": "",  # BOM
    # Checkmarks / crosses
    "\u2713": "v",
    "\u2714": "v",
    "\u2717": "x",
    "\u2718": "x",
    # Currency
    "\u20ac": "EUR",
    "\u00a3": "GBP",
    "\u00a5": "JPY",
    # Common typographic ligatures
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def _sanitize_for_pdf(s: str) -> str:
    """Sanitize a single string for xhtml2pdf rendering.

    1. Apply targeted character replacements (dashes, quotes, bullets, ...).
    2. NFKD-normalize + drop any remaining non-ASCII characters.

    Accented Latin chars (e.g. Jose -> Jose, cafe -> cafe) are preserved as
    their base letter via NFKD decomposition.
    """
    if not isinstance(s, str):
        return s
    # Targeted replacements first
    for src, dst in _PDF_SAFE_REPLACEMENTS.items():
        if src in s:
            s = s.replace(src, dst)
    # Transliterate remaining accented chars via NFKD + ASCII strip
    import unicodedata

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    # Drop C0/C1 control chars (except TAB, LF, CR) — xhtml2pdf can emit
    # stray DEL (\x7f) or other controls that render as black blocks.
    s = "".join(ch for ch in s if ch in "\t\n\r" or 0x20 <= ord(ch) < 0x7F)
    return s


def _sanitize_pdf_payload(obj):
    """Recursively sanitize every string inside a dict / list / tuple."""
    if isinstance(obj, str):
        return _sanitize_for_pdf(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_pdf_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize_pdf_payload(v) for v in obj)
    return obj


def render_resume_to_html(resume_dict, template_dir, template_name, output_html):
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    resume_html = template.render(resume=resume_dict)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(resume_html)
    return output_html


def html_to_pdf(input_html, output_pdf, base_path=None):
    """Render HTML file -> PDF.

    ``base_path`` should point at the template directory so that relative
    URLs in the HTML (e.g. ``fonts/Vera.ttf``) resolve correctly. xhtml2pdf
    uses this as the starting directory for ``@font-face`` url() lookups.
    """
    import os

    from xhtml2pdf import pisa

    with open(input_html, encoding="utf-8") as html_file, open(output_pdf, "wb") as pdf_file:
        # xhtml2pdf calls getDirName on the ``path`` arg to find the base
        # directory for relative URLs. That means we must pass a *file* inside
        # the directory (not the directory itself) or relative URLs resolve
        # one level up. Use a sentinel filename inside base_path.
        path = None
        if base_path:
            path = os.path.join(base_path, "_render.html")
        pisa.CreatePDF(html_file.read(), dest=pdf_file, path=path)


def generate_pdf_from_resume(resume, original_filename: str, template_name: str = "resume_template_7.html") -> str:
    """Generate PDF from resume JSON data."""
    import os
    import tempfile

    from .constants import TEMPLATE_DIR

    # Create temp files
    # delete=False is intentional — files are cleaned up manually after PDF generation.
    html_file = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)  # noqa: SIM115
    pdf_file = tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False)  # noqa: SIM115

    try:
        # Render HTML
        resume_dict = resume.model_dump()
        # Sanitize Unicode for xhtml2pdf (Helvetica can't render em-dashes,
        # smart quotes, bullets, etc. -> those become solid black blocks).
        resume_dict = _sanitize_pdf_payload(resume_dict)
        render_resume_to_html(resume_dict, TEMPLATE_DIR, template_name, html_file.name)
        html_file.close()
        # Convert to PDF (base_path lets xhtml2pdf find templates/fonts/*.ttf)
        html_to_pdf(html_file.name, pdf_file.name, base_path=TEMPLATE_DIR)

        return pdf_file.name
    finally:
        # Cleanup HTML file
        with contextlib.suppress(FileNotFoundError):
            os.unlink(html_file.name)
