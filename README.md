# proj_sle v2 — ชุดคัดกรองโรคพุ่มพวง (CSR outreach)

Thai-only, mobile-first SLE screening tool for a CSR outreach booth. Filled in mainly by
อสม./nurses interviewing visitors, with public self-service also supported. Every
submission is recorded for the CSR project record and a future research paper.

**v1 is unaffected.** It remains the clinician-facing tool at
<https://projsle-en4e9d6kiwyvx2cpq4xpxs.streamlit.app/> and is the only place the `d5` and
`full` EULAR levels exist. This repo was forked from
[`MaiRajborirug/proj_sle`](https://github.com/MaiRajborirug/proj_sle) at commit
**`fee7595`**, tagged there as `v1.2-frozen`.

## Modes

One deployment, two printed QR codes, plus a safe demo link.

| URL | Mode | Records? | Result shown |
| --- | --- | --- | --- |
| `/` | `public` | yes | Triage band only |
| `/?m=s` | `staff` | yes | Band + score breakdown under "สำหรับเจ้าหน้าที่" |
| `/?m=t` | `test` | **no** | Band + a yellow **โหมดทดสอบ** banner |

Print the plain URL on the public poster and `?m=s` on staff lanyards. `?m=t` is for
demos, training and screenshots — it never writes a row.

## Scoring: why there is no ML model here

v1 ships `model_d9.joblib`, a `CalibratedClassifierCV` over an SVC trained on
`SLE_NotSLE.csv` — a **202/200 case-control cohort of hospital patients**. Evaluated across
all 128 possible d9 inputs:

- **0.921 probability with no symptoms ticked.**
- **117 of 128 inputs** land in the highest risk band.
- **Non-monotonic**: an isolated fever *lowers* the output to 0.185; oral ulcer alone gives
  0.072; joint involvement alone 0.089.

That behaviour is coherent for the training cohort, where the comparison group was *other
sick people* — so isolated fever really does argue against SLE there. It is invalid at a
booth, where the comparison group is healthy people. No threshold choice fixes a model
whose no-symptom case already scores 0.921.

v2 therefore scores with the published **EULAR/ACR 2019** weights restricted to the seven
observable criteria, using the standard domain-maximum rule:

| Criterion | Domain | Weight |
| --- | --- | --- |
| ไข้ (Fever) | Constitutional | 2 |
| ผื่นลูปัสเฉียบพลัน (ACL) | Mucocutaneous | 6 |
| ผื่นลูปัสกึ่งเฉียบพลัน (SCL/DL) | Mucocutaneous | 4 |
| แผลในปาก (Oral ulcer) | Mucocutaneous | 2 |
| ผมร่วง (Alopecia) | Mucocutaneous | 2 |
| ข้ออักเสบ (Joint involvement) | Musculoskeletal | 6 |
| โปรตีนรั่วในปัสสาวะ (Proteinuria) | Renal | 4 |

Range 0–18. Monotonic by construction, and the test suite asserts that exhaustively over
all 128 inputs. Because immunological markers are excluded, this **cannot** be used for
formal EULAR/ACR 2019 classification — it is a screening triage only.

### Band cut-offs — PROVISIONAL

`core.BAND_YELLOW = 6`, `core.BAND_RED = 10`. RED matches the published classification
threshold of 10, reached on observable findings alone; YELLOW is the weight of a single
major finding. **These need clinical sign-off**, and should be revisited after the first
event against the real score distribution and the referral clinic's capacity.

## Recorded data

No contact details and nothing identifying an individual. **Sex and a 10-year age band are
collected** — both optional, neither pre-selected. One row per submission:

```
timestamp_utc, session_uuid, submission_seq, mode, app_version, sex, age_band,
Fever, ACL, SCL_or_DL, Oral_Ulcer, Alopecia, Joint_involvement, Proteinuria,
n_criteria, eular_score, band
```

`sex` is `หญิง` / `ชาย` / `ไม่ระบุ` / empty, and `age_band` is one of six 10-year bands or
empty. **Empty means the question was not answered**, which is distinct from the explicit
`ไม่ระบุ` answer — keep them apart in analysis.

> **These two fields are personal data under PDPA.** The app carries a short on-screen
> notice, but the wording has not been reviewed by anyone qualified, and collecting them
> may change what the ethics committee requires — a plain exemption determination may no
> longer be the right application. Settle this before the first event.

Redo appends a new row with the same `session_uuid` and an incremented `submission_seq`.
**At analysis time take the highest `submission_seq` per `session_uuid`**; earlier rows are
kept so correction behaviour can be studied.

Client IPs are hashed with a per-process salt purely for rate limiting and are **never**
written anywhere. Nothing else identifying is collected.

## Images

Each criterion carries an `images` **list** in `criteria_d9.json` — zero, one or several.
ไข้ has none and renders full width; ผื่นลูปัสกึ่งเฉียบพลัน has two, stacked.

| Criterion | Image | Status |
| --- | --- | --- |
| ไข้ | *(none)* | — |
| ผื่นลูปัสเฉียบพลัน | `acute1.jpg` | supplied by project team |
| ผื่นลูปัสกึ่งเฉียบพลัน | `subacute1.png`, `subacute2.jpg` | supplied by project team |
| แผลในปาก | `oral_ulcer.png` | **v1 textbook placeholder** |
| ผมร่วง | `alopecia.png` | **v1 textbook placeholder** |
| ข้ออักเสบ | `joint1.png` | supplied by project team |
| โปรตีนรั่วในปัสสาวะ | `proteinuria.png` | **v1 textbook placeholder** |

Every entry also carries `image_source` and `image_licence`. **All are currently UNKNOWN or
TO BE CONFIRMED** — resolve them before public launch and update the fields as you go, an
ethics committee or reviewer will ask. Note that teaching consent often does **not** cover
use in a publicly accessible app.

`tests/test_core.py` asserts that every referenced path exists, so a stale reference fails
the suite rather than showing a broken card at the booth.

## Develop

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m pytest -q          # 28 tests
.venv/bin/python -m streamlit run app.py
```

`uv` targets an active conda environment in preference to `.venv`, so pass
`--python .venv/bin/python` explicitly if `CONDA_PREFIX` is set.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Thai mobile UI, mode handling, submit/redo flow |
| `core.py` | EULAR subset scoring and triage banding — no Streamlit import |
| `storage.py` | Google Sheets append, retry queue, no-op in test mode |
| `throttle.py` | Salted in-memory IP-hash rate limit |
| `criteria_d9.json` | The 7 criteria: weights, Thai text, image provenance |
| `SETUP.md` | Domain, hosting and Google service-account setup |

Deployment and Google setup: see [SETUP.md](SETUP.md).
