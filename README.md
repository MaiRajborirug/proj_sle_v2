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

## Scoring: why there is no ML classifier here

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

v2 therefore does not use a learned classifier. It uses a **points table**, summed, with
weights that start from the published EULAR/ACR 2019 values and are adjusted by the cohort.

### The weights, and why they are not the published ones

| Criterion | Domain | **Weight used** | EULAR/ACR 2019 |
| --- | --- | ---: | ---: |
| โปรตีนรั่วในปัสสาวะ (Proteinuria) | Renal | **12** | 4 |
| ผื่นลูปัสกึ่งเฉียบพลัน (SCL/DL) | Mucocutaneous | **8** | 4 |
| ผื่นลูปัสเฉียบพลัน (ACL) | Mucocutaneous | **7** | 6 |
| ผมร่วง (Alopecia) | Mucocutaneous | **5** | 2 |
| ไข้ (Fever) | Constitutional | **2** | 2 |
| แผลในปาก (Oral ulcer) | Mucocutaneous | **1** | 2 |
| ข้ออักเสบ (Joint involvement) | Musculoskeletal | **1** | 6 |

Range 0–36. `criteria_d9.json` carries both: `score` is the weight in use, `eular_score`
the published value.

**The published weights alone performed badly here.** Scored with the domain-maximum rule
on the 402-patient cohort, they reach **ROC AUC 0.647 and specificity 0.205** — they flag
906 people per 1,000 screened. The cause is joint involvement at +6: 170 of the 402
patients present as joint-involvement-only, and 154 of those are controls, so that one
criterion alone pushes most of the comparison group into the referral band. Those weights
were derived against a broader comparison group than this cohort's; they are not wrong,
they are being asked the wrong question.

**Fitting freely on the cohort is worse.** Unconstrained, oral ulcer and joint involvement
both take *negative* weight — coherent for a sample whose controls are rheumatology
patients, and dangerous anywhere else, since it means reporting one more symptom can lower
your score.

**So the fit is shrunk toward the published values** (`exp/fit_points_model.py`):

```
minimise   logloss(Xw + b)  +  λ‖w − κ·w_eular‖²      subject to  w ≥ ε
```

λ = 2, chosen by cross-validating the whole procedure. Every weight stays strictly
positive, so the score is monotonic by construction and the test suite asserts that
exhaustively over all 128 inputs. Cross-validated at the referral cut-off: **AUC 0.905,
sensitivity 0.802, specificity 0.966** — against 0.647 / 0.862 / 0.205 for the published
weights. Full comparison of 16 model families in [`exp/REPORT.md`](exp/REPORT.md).

**Summed, not domain-maximum.** The domain-maximum rule exists to stop one organ system
dominating a *classification* score. For screening it discards information — two
mucocutaneous findings are stronger evidence than either alone.

Because immunological markers are excluded, this **cannot** be used for formal EULAR/ACR
2019 classification — it is a screening triage only.

> ⚠️ **The urine dipstick is not optional.** Proteinuria carries 12 of the 36 points, and
> 59% of the SLE patients in the cohort have it. Without a dipstick at the booth the table
> degrades to AUC 0.651 and sensitivity 0.545 — no better than the published weights it
> replaced. Do not run this form without one.

### Band cut-offs — PROVISIONAL

Four bands, placed where the likelihood ratio changes rather than at even intervals:

| Band | Score | LR | Meaning |
| --- | --- | ---: | --- |
| 🟢 GREEN | 0–2 | 0.14 | Risk genuinely reduced |
| 🟡 YELLOW | 3–7 | **0.93** | Findings present, risk **unchanged** — no referral |
| 🟠 ORANGE | 8–12 | 10.1 | Refer, non-urgently |
| 🔴 RED | 13+ | ≥35 | Refer promptly; 0 of 202 controls reached this band |

`core.BAND_YELLOW = 3`, `core.BAND_ORANGE = 8`, `core.BAND_RED = 13`.

YELLOW exists because scores 3–7 carry a likelihood ratio of 0.93 — statistically
indistinguishable from 1. Those visitors must not be told they are clear, since their risk
was not reduced, but referring them would mean referring one control for every case found.
Folding that range into either neighbour would misreport it. **Referral starts at ORANGE.**

Even RED is roughly 3–4% post-test probability at a public booth's base rate, so the copy
says *ควรพบแพทย์โดยเร็ว*, never *คุณน่าจะเป็น SLE*.

**These need clinical sign-off**, and should be revisited after the first event against the
real score distribution and the referral clinic's capacity.

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
