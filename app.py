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

import uuid
from datetime import datetime, timezone

import streamlit as st

import core
import storage
from throttle import Throttle

MODES = {"s": "staff", "t": "test"}

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
  section.main > div { max-width: 46rem; padding-top: 1rem; }
  html, body, [class*="css"] { font-size: 18px; }
  [data-testid="stCheckbox"] label p { font-size: 1.3rem !important; font-weight: 600; }
  [data-testid="stCheckbox"] { padding: 0.35rem 0; }
  div.stButton > button {
      font-size: 1.25rem; font-weight: 700;
      padding: 0.9rem 1rem; width: 100%; border-radius: 0.75rem;
  }
  .band-card { padding: 1.75rem 1.5rem; border-radius: 1rem; text-align: center; }
  .band-card .t { font-size: 2.1rem; font-weight: 800; margin: 0.4rem 0 0.8rem; }
  .band-card .a { font-size: 1.15rem; line-height: 1.7; }
  .test-banner {
      background: #ffe58f; color: #613400; font-weight: 800; font-size: 1.15rem;
      padding: 0.85rem 1rem; border-radius: 0.6rem; text-align: center;
      margin-bottom: 1rem; border: 2px solid #d4a017;
  }
</style>
"""


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
    """Reset to a fresh, unrelated screening — new session ID and cleared answers."""
    for c in get_criteria():
        st.session_state.pop(f"c_{c['key']}", None)
    st.session_state["session_uuid"] = str(uuid.uuid4())
    st.session_state["seq"] = 0
    st.session_state["result"] = None
    st.session_state["stage"] = "form"


def build_record(values: dict, score: int, band: str, mode: str) -> dict:
    """Assemble one sheet row from the answers and the score."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_uuid": st.session_state["session_uuid"],
        "submission_seq": st.session_state["seq"],
        "mode": mode,
        "app_version": core.APP_VERSION,
        "n_criteria": sum(bool(v) for v in values.values()),
        "eular_score": score,
        "band": band,
    }
    for c in get_criteria():
        record[c["column"]] = int(bool(values.get(c["key"], False)))
    return record


def render_header(mode: str) -> None:
    """Render the logo, title and the test-mode banner."""
    if mode == "test":
        st.markdown(
            '<div class="test-banner">โหมดทดสอบ — ข้อมูลจะไม่ถูกบันทึก</div>',
            unsafe_allow_html=True,
        )
    st.image("images/md_kmitl.png", use_container_width=True)
    st.markdown("## ชุดคัดกรองโรคพุ่มพวง")
    st.caption("แบบคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยโรค · อ้างอิงเกณฑ์ EULAR/ACR 2019")


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


def render_form(mode: str) -> dict:
    """Render the seven criterion cards and return the checkbox state."""
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

    st.divider()
    st.caption(
        "⚠️ เครื่องมือนี้เป็นการคัดกรองเบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยโรค "
        "และไม่ทดแทนการตรวจโดยแพทย์"
    )


main()
