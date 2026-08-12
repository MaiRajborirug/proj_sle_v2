"""Rebuild the SLE brochure with the screening-app QR on its cover panel.

Two phases against `media/SLE info brochure.pdf`, which is never modified:

  1. Shift the cover title card up, to make room for the QR without crowding it.
  2. Draw the QR card into the space that opens up below it.

Phase 1 edits page 1's content stream in place. That is specific to this Google Slides
export — the coordinate literals it keys on would not survive a re-export — so it asserts
its match counts and refuses to write a brochure whose title did not actually move.

Run:

    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project \
      --with pypdf --with reportlab --with qrcode --with uharfbuzz --with fonttools \
      python scripts/stamp_brochure_qr.py \
      "media/SLE info brochure.pdf" "media/SLE info brochure_qr.pdf"

These packages are build-time only and deliberately absent from requirements.txt, which
Streamlit Cloud installs on every deploy.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import qrcode
import uharfbuzz as hb
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont as FTFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONTS = Path(__file__).parent / "fonts"

URL = "https://mdkmitl-osm-sle.streamlit.app"
LABEL = "ลองตรวจคัดกรองที่นี่"

PAGE_W, PAGE_H = 960.0, 540.0
NAVY = HexColor("#12358c")
GREY = HexColor("#555555")

# Page 1 draws inside `1 0 0 -1 0 540 cm`, so its Y axis runs downward, then
# `0.001968504 0 0 0.001968504 0 0 cm`. Text groups nest a further `381 0 0 381 0 0 cm`,
# which leaves 0.75 pt per unit inside them. Moving content *up* the page therefore means
# *subtracting* in both spaces.
PATH_UNITS_PER_PT = 1 / 0.001968504
TEXT_UNITS_PER_PT = 1 / 0.75

# The cover column runs from the logo caption (y≈384) down to the shield icon (y≈89). The
# title card is 81.7 pt tall and the QR card 126 pt, so ~87 pt of slack is split into three
# gaps of roughly 36 / 29 / 24 — a little airier under the logo, which reads correctly.
TITLE_RISE = 40.0
QR_CARD_TOP, QR_CARD_BOT = 239.0, 113.0
LABEL_BASELINE, QR_BOTTOM, URL_BASELINE = 222.0, 127.0, 118.0

BOX_CX, BOX_W = 795.4, 150.0
QR_SIZE = 84.0

# The white title card, drawn twice — once filled, once stroked in the same white.
TITLE_CARD = re.compile(
    r"338782\.0\s+117929\.5\s+m\s+470410\.0\s+117929\.5\s+l\s+"
    r"470410\.0\s+159437\.5\s+l\s+338782\.0\s+159437\.5\s+l"
)
# The two text groups inside that card: the Thai title, and the English + (โรคพุ่มพวง) pair.
TITLE_TEXT = re.compile(r"(q\s+1\.0\s+0\s+0\s+1\.0\s+882\.6982\s+)(\d+\.\d+)(\s+cm)")


class _OutlinePen(BasePen):
    """Trace a glyph outline onto a reportlab path, scaled and translated."""

    def __init__(self, glyph_set, path, scale, dx, dy):
        super().__init__(glyph_set)
        self._path, self._s, self._dx, self._dy = path, scale, dx, dy

    def _pt(self, p):
        return p[0] * self._s + self._dx, p[1] * self._s + self._dy

    def _moveTo(self, p):
        self._path.moveTo(*self._pt(p))

    def _lineTo(self, p):
        self._path.lineTo(*self._pt(p))

    def _curveToOne(self, p1, p2, p3):
        self._path.curveTo(*self._pt(p1), *self._pt(p2), *self._pt(p3))

    def _closePath(self):
        self._path.close()


def draw_shaped(c, text, font_path, size, cx, baseline):
    """Draw centred text as vector outlines, shaped by HarfBuzz.

    reportlab positions glyphs by advance width alone, which drops Thai tone marks onto the
    vowel below them — "ที่นี่" loses both mai eks. HarfBuzz applies the font's GPOS mark
    attachment, so the marks stack where they belong.
    """
    face = hb.Face(hb.Blob.from_file_path(str(font_path)))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb.Font(face), buf)

    ft = FTFont(font_path)
    glyph_set = ft.getGlyphSet()
    names = ft.getGlyphOrder()
    scale = size / face.upem

    width = sum(pos.x_advance for pos in buf.glyph_positions) * scale
    pen_x, pen_y = cx - width / 2, baseline
    path = c.beginPath()
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        glyph_set[names[info.codepoint]].draw(_OutlinePen(
            glyph_set, path, scale,
            pen_x + pos.x_offset * scale, pen_y + pos.y_offset * scale))
        pen_x += pos.x_advance * scale
        pen_y += pos.y_advance * scale
    c.drawPath(path, stroke=0, fill=1)


def raise_title(page, rise_pt: float) -> None:
    """Move the cover title card and its text up the page by `rise_pt`.

    Args:
        page: Page 1 of the brochure. Its content stream is replaced in place.
        rise_pt: How far up the page to move the title, in points.

    Raises:
        SystemExit: If the content stream does not look like the export this was written
            for, rather than silently returning an unmoved title.
    """
    stream = page.get_contents().get_data().decode("latin-1")

    path_shift = rise_pt * PATH_UNITS_PER_PT
    top, bottom = 117929.5 - path_shift, 159437.5 - path_shift
    moved_card = (f"338782.0 {top} m 470410.0 {top} l "
                  f"470410.0 {bottom} l 338782.0 {bottom} l")
    stream, card_hits = TITLE_CARD.subn(moved_card, stream)

    text_shift = rise_pt * TEXT_UNITS_PER_PT
    stream, text_hits = TITLE_TEXT.subn(
        lambda m: f"{m.group(1)}{float(m.group(2)) - text_shift}{m.group(3)}", stream)

    if (card_hits, text_hits) != (2, 2):
        sys.exit(f"expected 2 card paths and 2 text groups, found {card_hits} and "
                 f"{text_hits} — this is not the export the shift was measured against")

    replacement = DecodedStreamObject()
    replacement.set_data(stream.encode("latin-1"))
    page.replace_contents(replacement)


def qr_overlay() -> bytes:
    """Render the QR card as a one-page PDF to merge onto the cover."""
    pdfmetrics.registerFont(TTFont("Sarabun", str(FONTS / "Sarabun-Regular.ttf")))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(PAGE_W, PAGE_H))

    # White card, echoing the title and caption cards already on the panel.
    c.setFillColor(white)
    c.setStrokeColor(Color(0.78, 0.85, 0.94))
    c.setLineWidth(0.8)
    c.roundRect(BOX_CX - BOX_W / 2, QR_CARD_BOT, BOX_W, QR_CARD_TOP - QR_CARD_BOT,
                6, stroke=1, fill=1)

    c.setFillColor(NAVY)
    draw_shaped(c, LABEL, FONTS / "Sarabun-Bold.ttf", 12, BOX_CX, LABEL_BASELINE)

    # Vector modules rather than a raster image, so the code stays sharp at any print size.
    # Error correction H survives a brochure that has been folded, creased and sun-bleached.
    code = qrcode.QRCode(border=0, error_correction=qrcode.constants.ERROR_CORRECT_H)
    code.add_data(URL)
    code.make(fit=True)
    modules = code.get_matrix()
    step = QR_SIZE / len(modules)
    x0 = BOX_CX - QR_SIZE / 2
    c.setFillColor(black)
    for row, cells in enumerate(modules):
        for col, on in enumerate(cells):
            if on:
                c.rect(x0 + col * step, QR_BOTTOM + QR_SIZE - (row + 1) * step,
                       step, step, stroke=0, fill=1)

    # Printed as well as encoded: booth visitors who cannot get a camera to focus, or who
    # have no scanner app, still have a way in.
    c.setFillColor(GREY)
    c.setFont("Sarabun", 7)
    c.drawCentredString(BOX_CX, URL_BASELINE, URL.replace("https://", ""))
    c.save()
    return buf.getvalue()


def main() -> None:
    """Entry point."""
    source, destination = sys.argv[1], sys.argv[2]
    # Cloned rather than page-by-page: replace_contents is only reliable on a page the
    # writer already owns.
    writer = PdfWriter(clone_from=source)
    cover = writer.pages[0]
    raise_title(cover, TITLE_RISE)
    cover.merge_page(PdfReader(io.BytesIO(qr_overlay())).pages[0])
    writer.write(destination)
    print(f"wrote {destination}")


main()
