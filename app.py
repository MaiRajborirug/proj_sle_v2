"""SLE screening app v2 — Thai-only, mobile-first, for CSR outreach booths.

Three modes, selected by the `m` URL parameter so one deployment serves two printed QR
codes plus a safe demo link:

    /          public  — self-service, records, triage band only
    /?m=s      staff   — assisted by อสม./nurse, records, band + score breakdown on tap
    /?m=t      test    — records nothing, shows a prominent test banner

Scoring uses the published EULAR/ACR 2019 weights restricted to the seven observable
criteria, not v1's machine-learning model — see the note at the top of core.py for why
that model cannot be used on a public population.
"""

from __future__ import annotations

import base64
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

import core
import storage
from throttle import Throttle

MODES = {"s": "staff", "t": "test"}

# The address the footer QR code encodes — the public entry point, deliberately without
# ?m=s, since the QR exists so a bystander can open the app on their own phone. Override
# with PUBLIC_URL when the app moves to its own domain.
DEFAULT_PUBLIC_URL = "https://mdkmitl-osm-sle.streamlit.app"

SPONSOR_LOGO_ROWS = [
    ["images/mdkimtl.jpeg", "images/osm.jpg"],
    ["images/kmitllogo.jpeg", "images/sriracha.png"],
]

SEX_OPTIONS = ["หญิง", "ชาย", "ไม่ระบุ"]
AGE_BANDS = ["ต่ำกว่า 20 ปี", "20–29 ปี", "30–39 ปี", "40–49 ปี", "50–59 ปี", "60 ปีขึ้นไป"]

# Both fields default to no selection, so an unanswered question is recorded as blank
# rather than as a value nobody chose.
DEMOGRAPHIC_KEYS = ("d_sex", "d_age")

BANDS = {
    core.GREEN: {
        "emoji": "🟢",
        "bg": "#e8f6ec",
        "fg": "#1b6b34",
        "title": "ไม่พบสัญญาณเตือน",
        "advice": "จากข้อมูลที่ให้มา ยังไม่พบสัญญาณที่บ่งชี้โรคพุ่มพวง "
                  "หากมีอาการผิดปกติเพิ่มเติมในภายหลัง ควรปรึกษาแพทย์",
    },
    core.YELLOW: {
        "emoji": "🟡",
        "bg": "#fdf4e3",
        "fg": "#8a5a00",
        "title": "ควรสังเกตอาการ",
        "advice": "พบสัญญาณบางอย่างที่ควรติดตาม แนะนำให้สังเกตอาการต่อเนื่อง "
                  "และปรึกษาแพทย์หากอาการไม่ดีขึ้นหรือมีอาการเพิ่มเติม",
    },
    core.RED: {
        "emoji": "🔴",
        "bg": "#fdeaea",
        "fg": "#a32020",
        "title": "ควรปรึกษาแพทย์",
        "advice": "พบสัญญาณหลายอย่างที่เกี่ยวข้องกับโรคพุ่มพวง "
                  "แนะนำให้พบแพทย์เพื่อตรวจเพิ่มเติม",
    },
}

CSS = """
<style>
  /* @import must come first in the stylesheet. If Google Fonts is unreachable — a booth
     on a poor connection — the browser falls back to the system sans-serif. */
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600;700;800&display=swap');
  html, body, [class*="css"], .stApp, button, input, select, textarea, div, span, p,
  h1, h2, h3, h4, label {
      font-family: 'Noto Sans Thai', sans-serif !important;
  }
  section.main > div { max-width: 46rem; padding-top: 1rem; }
  /* Sized up for older users at a booth. Everything below is in rem, so it scales
     from this one value. */
  html, body, [class*="css"] { font-size: 21px; }
  [data-testid="stCheckbox"] label p { font-size: 1.35rem !important; font-weight: 600; }
  [data-testid="stCaptionContainer"] p { font-size: 1.02rem !important; }
  [data-testid="stWidgetLabel"] p { font-size: 1.15rem !important; font-weight: 600; }
  [data-testid="stCheckbox"] { padding: 0.35rem 0; }
  div.stButton > button {
      font-size: 1.25rem; font-weight: 700;
      padding: 0.9rem 1rem; width: 100%; border-radius: 0.75rem;
  }
  .band-card { padding: 1.75rem 1.5rem; border-radius: 1rem; text-align: center; }
  .band-card .t { font-size: 2.1rem; font-weight: 800; margin: 0.4rem 0 0.8rem; }
  .band-card .a { font-size: 1.15rem; line-height: 1.7; }
  /* Equal logo height with width:auto keeps each one's own aspect ratio. */
  .sponsors {
      display: flex; flex-direction: column; align-items: center;
      gap: 1.1rem; margin-top: 0.6rem;
  }
  .sponsor-row {
      display: flex; justify-content: center; align-items: center;
      gap: 2.25rem; flex-wrap: wrap;
  }
  .sponsors img { height: 88px; width: auto; }
  .qr-block { text-align: center; margin: 0.5rem 0 1.25rem; }
  .qr-block img { width: 132px; height: 132px; image-rendering: pixelated; }
  .qr-block .cap { font-size: 0.95rem; color: #555; margin-top: 0.3rem; }
  .test-banner {
      background: #ffe58f; color: #613400; font-weight: 800; font-size: 1.15rem;
      padding: 0.85rem 1rem; border-radius: 0.6rem; text-align: center;
      margin-bottom: 1rem; border: 2px solid #d4a017;
  }
</style>
"""


def load_secrets_into_env() -> None:
    """Copy Streamlit-managed secrets into the environment.

    Render and other container hosts supply configuration as environment variables;
    Streamlit Community Cloud supplies it through `st.secrets`. `storage.py` reads only
    the environment, so bridge the two here and keep one source of truth downstream.
    Environment variables already set always win.
    """
    for key in ("GOOGLE_SERVICE_ACCOUNT_JSON", "SHEET_ID", "IP_HASH_SALT", "PUBLIC_URL"):
        if key in os.environ:
            continue
        try:
            value = st.secrets[key]
        except Exception:
            continue
        os.environ[key] = str(value)


@st.cache_resource
def get_recorder(enabled: bool) -> storage.SheetRecorder:
    """Build the sheet recorder, shared across sessions."""
    return storage.SheetRecorder(storage.build_header(core.load_criteria()), enabled=enabled)


@st.cache_resource
def get_throttle() -> Throttle:
    """Build the process-wide submission throttle."""
    return Throttle()


@st.cache_data
def get_criteria() -> list[dict]:
    """Load the d9 criteria once per process."""
    return core.load_criteria()


def public_url() -> str:
    """The address the footer QR code points at."""
    return os.environ.get("PUBLIC_URL") or DEFAULT_PUBLIC_URL


def current_mode() -> str:
    """Resolve the app mode from the `m` URL parameter."""
    return MODES.get(st.query_params.get("m", ""), "public")


def client_ip() -> str:
    """Best-effort client address for throttling.

    Returns:
        The client IP, or an empty string when the host does not expose one (which
        disables throttling rather than blocking everyone).
    """
    try:
        headers = st.context.headers
    except Exception:
        return ""
    forwarded = headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else ""


def init_session() -> None:
    """Seed per-visitor session state on first run."""
    st.session_state.setdefault("session_uuid", str(uuid.uuid4()))
    st.session_state.setdefault("seq", 0)
    st.session_state.setdefault("stage", "form")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("in_flight", False)


def start_new_person() -> None:
    """Reset to a fresh, unrelated screening — new session ID and cleared answers.

    Clears the demographics too: carrying the previous visitor's sex and age band into the
    next person's row would silently corrupt the dataset.
    """
    for c in get_criteria():
        st.session_state.pop(f"c_{c['key']}", None)
    for key in DEMOGRAPHIC_KEYS:
        st.session_state.pop(key, None)
    st.session_state["session_uuid"] = str(uuid.uuid4())
    st.session_state["seq"] = 0
    st.session_state["result"] = None
    st.session_state["stage"] = "form"


def build_record(values: dict, score: int, band: str, mode: str) -> dict:
    """Assemble one sheet row from the answers and the score.

    Unanswered demographics are written as an empty string, distinguishing "not asked"
    from the explicit "ไม่ระบุ" answer.
    """
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_uuid": st.session_state["session_uuid"],
        "submission_seq": st.session_state["seq"],
        "mode": mode,
        "app_version": core.APP_VERSION,
        "sex": st.session_state.get("d_sex") or "",
        "age_band": st.session_state.get("d_age") or "",
        "n_criteria": sum(bool(v) for v in values.values()),
        "eular_score": score,
        "band": band,
    }
    for c in get_criteria():
        record[c["column"]] = int(bool(values.get(c["key"], False)))
    return record


def render_header(mode: str) -> None:
    """Render the test-mode banner and the title."""
    if mode == "test":
        st.markdown(
            '<div class="test-banner">โหมดทดสอบ — ข้อมูลจะไม่ถูกบันทึก</div>',
            unsafe_allow_html=True,
        )
    st.markdown("# ชุดคัดกรองโรคพุ่มพวง")
    st.caption("แบบคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยโรค")


@st.cache_data
def qr_block_html(url: str) -> str:
    """Render a QR code for `url` as an inline SVG data URI.

    Generated at runtime rather than committed as a PNG so the public URL lives in exactly
    one place — a stale QR code sends people to a dead address with no visible symptom.

    Args:
        url: The address the QR code should encode.
    """
    import io

    import qrcode
    import qrcode.image.svg

    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return (
        '<div class="qr-block">'
        f'<img src="data:image/svg+xml;base64,{encoded}" alt="QR code">'
        '<div class="cap">สแกนเพื่อเปิดแบบคัดกรองนี้ในมือถือของคุณ</div>'
        "</div>"
    )


@st.cache_data
def sponsor_strip_html() -> str:
    """Build the sponsor logo rows as one inline HTML block.

    Streamlit's own image widget cannot centre a row of logos or normalise their heights,
    so the images are embedded as data URIs and laid out with flexbox instead.
    """
    rows = []
    for row in SPONSOR_LOGO_ROWS:
        tags = []
        for src in row:
            mime = "image/png" if src.endswith(".png") else "image/jpeg"
            encoded = base64.b64encode(Path(src).read_bytes()).decode()
            tags.append(f'<img src="data:{mime};base64,{encoded}" alt="">')
        rows.append(f'<div class="sponsor-row">{"".join(tags)}</div>')
    return f'<div class="sponsors">{"".join(rows)}</div>'


def render_footer() -> None:
    """Render the QR code, disclaimer and sponsor logos at the foot of the page."""
    st.divider()
    st.markdown(qr_block_html(public_url()), unsafe_allow_html=True)
    st.caption(
        "⚠️ เครื่องมือนี้เป็นการร่วมมือระหว่าง คณะแพทย์ศาสตร์ "
        "สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง และ อสม.คลองเขื่อน "
        "เป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยโรค และไม่ทดแทนการตรวจโดยแพทย์"
    )
    st.markdown(sponsor_strip_html(), unsafe_allow_html=True)


def render_criterion(c: dict) -> bool:
    """Render one criterion card and return whether it was ticked.

    Criteria with no images (currently ไข้) use the full card width; criteria with
    several images stack them in the image column.
    """
    with st.container(border=True):
        images = c["images"]
        if images:
            img_col, text_col = st.columns([1, 2], vertical_alignment="center")
            with img_col:
                for src in images:
                    st.image(src, use_container_width=True)
        else:
            text_col = st.container()
        with text_col:
            ticked = st.checkbox(c["name_th"], key=f"c_{c['key']}")
            st.caption(c["description_th"])
            if c["requires_test"]:
                st.caption(f"🔬 ต้องตรวจ: {c['requires_test']}")
    return ticked


def render_demographics() -> None:
    """Render the optional sex and age-band questions.

    Values are read back off session state in `build_record`, so nothing is returned.
    """
    st.markdown("### ข้อมูลทั่วไป")
    with st.container(border=True):
        st.radio("เพศ", SEX_OPTIONS, key="d_sex", index=None, horizontal=True)
        st.selectbox("ช่วงอายุ", AGE_BANDS, key="d_age", index=None,
                     placeholder="เลือกช่วงอายุ")
        st.caption(
            "ไม่บังคับ · ข้อมูลนี้บันทึกแบบไม่ระบุตัวตน "
            "เพื่อใช้สรุปผลโครงการและการวิจัยเท่านั้น"
        )


def render_form(mode: str) -> dict:
    """Render the demographics and the seven criterion cards; return the checkbox state."""
    render_demographics()
    st.markdown("### เลือกอาการที่พบ")
    return {c["key"]: render_criterion(c) for c in get_criteria()}


def submit(values: dict, mode: str) -> None:
    """Score the answers, record the submission and move to the result screen."""
    if st.session_state["in_flight"]:
        return
    st.session_state["in_flight"] = True
    try:
        if not get_throttle().allow(client_ip()):
            st.session_state["result"] = {"throttled": True}
            st.session_state["stage"] = "result"
            return
        score, breakdown = core.compute_score(values, get_criteria())
        band = core.compute_band(score)
        st.session_state["seq"] += 1
        record = build_record(values, score, band, mode)
        saved = get_recorder(mode != "test").append(record)
        st.session_state["result"] = {
            "throttled": False, "score": score, "breakdown": breakdown,
            "band": band, "saved": saved,
        }
        st.session_state["stage"] = "result"
    finally:
        st.session_state["in_flight"] = False


def render_result(mode: str) -> None:
    """Render the triage card and the redo / next-person actions."""
    result = st.session_state["result"]
    if result.get("throttled"):
        st.warning("มีการส่งข้อมูลถี่เกินไป กรุณารอสักครู่แล้วลองใหม่อีกครั้ง")
    else:
        band = BANDS[result["band"]]
        st.markdown(
            f'<div class="band-card" style="background:{band["bg"]};color:{band["fg"]}">'
            f'<div style="font-size:3rem">{band["emoji"]}</div>'
            f'<div class="t">{band["title"]}</div>'
            f'<div class="a">{band["advice"]}</div></div>',
            unsafe_allow_html=True,
        )
        if mode in ("staff", "test"):
            with st.expander("สำหรับเจ้าหน้าที่"):
                st.write(f"คะแนน EULAR/ACR (เฉพาะ 7 เกณฑ์ที่สังเกตได้): **{result['score']} / 18**")
                domain_th = {c["domain"]: c["domain_th"] for c in get_criteria()}
                for domain, pts in result["breakdown"].items():
                    st.write(f"- {domain_th[domain]}: +{pts}")
                st.caption(
                    "คะแนนนี้ไม่รวมผลตรวจภูมิคุ้มกัน จึงใช้จำแนกโรคตามเกณฑ์ EULAR/ACR 2019 "
                    "อย่างเป็นทางการไม่ได้ ใช้เพื่อการคัดกรองเบื้องต้นเท่านั้น"
                )
                if not result["saved"]:
                    st.error("บันทึกข้อมูลไม่สำเร็จ — ระบบจะพยายามบันทึกใหม่อัตโนมัติ")

    st.write("")
    redo_col, next_col = st.columns(2)
    with redo_col:
        if st.button("แก้ไขคำตอบ", use_container_width=True):
            st.session_state["stage"] = "form"
            st.rerun()
    with next_col:
        if st.button("คัดกรองคนถัดไป", type="primary", use_container_width=True):
            start_new_person()
            st.rerun()


def main() -> None:
    """Entry point."""
    st.set_page_config(page_title="คัดกรองโรคพุ่มพวง | MD KMITL", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    load_secrets_into_env()
    init_session()
    mode = current_mode()
    render_header(mode)

    if st.session_state["stage"] == "form":
        values = render_form(mode)
        st.write("")
        n = sum(values.values())
        st.caption(f"เลือกแล้ว {n} จาก {len(get_criteria())} ข้อ")
        if st.button("ดูผลการคัดกรอง", type="primary", use_container_width=True):
            submit(values, mode)
            st.rerun()
    else:
        render_result(mode)

    render_footer()


main()
