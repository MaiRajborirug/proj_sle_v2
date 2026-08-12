# Setup — domain, hosting, Google Sheet

Everything here is done once. Nothing in this file requires Python knowledge.

---

## 1. Google Sheet + service account (free)

A "service account" is a robot Google account the app logs in as. You give it permission
to one spreadsheet and nothing else.

1. Create the spreadsheet. **Use a KMITL Workspace account as the owner, not a personal
   Gmail** — an ethics committee will ask where study data lives, and "in a staff member's
   private Drive" is a bad answer. Copy the **sheet ID** from the URL:
   `docs.google.com/spreadsheets/d/`**`THIS_PART`**`/edit`
2. Go to <https://console.cloud.google.com/> → create a project (any name).
3. **APIs & Services → Library** → search "Google Sheets API" → **Enable**.
4. **APIs & Services → Credentials → Create credentials → Service account.** Any name.
   Skip the optional role/access steps.
5. Open the new service account → **Keys → Add key → Create new key → JSON**. A `.json`
   file downloads. **This is a credential — never commit it to git.**
6. Open that JSON and copy the `client_email` value (looks like
   `something@project.iam.gserviceaccount.com`).
7. Back in your spreadsheet: **Share** → paste that email → give it **Editor** → Send.
   This is the step everyone forgets; without it the app gets a permission error.

The app writes the header row itself on first use.

---

## 2. Domain

1. Buy the domain at any registrar (Namecheap, Cloudflare, GoDaddy). Roughly ฿500/yr for
   a `.com`.
2. After step 3 the host gives you a target hostname. Add one **CNAME** record at the
   registrar pointing your domain at it. The host issues the HTTPS certificate
   automatically, usually within a few minutes.

**Optional, and worth doing later:** ask KMITL IT for an institutional subdomain such as
`sle.md.kmitl.ac.th`. That is one more CNAME pointing at the same host, it costs nothing,
and it is a far more credible URL on a poster and in a paper.

---

## 3. Hosting

Any always-on host works. Render is the least fiddly.

> **Do not use Streamlit Community Cloud for v2.** It sleeps after ~12h idle, and with
> bursty booth traffic that means the first person to scan your QR code gets a "wake this
> app up" screen. Its filesystem is also wiped on restart.

1. Push this repo to GitHub.
2. Render → **New → Web Service** → connect the repo.
   - Build command: `pip install -r requirements.txt`
   - Start command:
     `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Instance: **1–2 GB RAM.** The 512 MB tier is too small for ~30–90 concurrent
     Streamlit sessions. Roughly ฿700–900/mo.
3. Add the environment variables below.
4. **Settings → Custom Domain** → add your domain → copy the CNAME target into your
   registrar (step 2).

### Environment variables

| Variable | Value |
| --- | --- |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | The **entire contents** of the downloaded JSON key, pasted as one value |
| `SHEET_ID` | The sheet ID from step 1 |
| `IP_HASH_SALT` | Any long random string. Optional — a random one is generated per restart if unset |

---

## 4. QR codes

Generate two, from the same deployment:

- **Public poster** → `https://your-domain/`
- **Staff lanyards** → `https://your-domain/?m=s`

Use `https://your-domain/?m=t` for demos and training. It shows a yellow
**โหมดทดสอบ** banner and writes nothing.

---

## 5. Before the first real event

- [ ] **Ethics.** Apply to the KMITL faculty committee for an **exemption determination**.
      The data is anonymous so this is usually low-friction, but journals ask for it and
      "not applicable" is a weak answer. Mention the transient IP hashing used for rate
      limiting.
- [ ] **Image consent.** Confirm the patient photographs' consent covers use in a publicly
      accessible app. Teaching consent often does not. Update `image_source` and
      `image_licence` in `criteria_d9.json` as each image clears.
- [ ] **Band cut-offs.** Get clinical sign-off on `BAND_YELLOW = 6` / `BAND_RED = 10` in
      `core.py`, and agree how many 🔴 referrals the clinic can absorb per day.
- [ ] **Thai wording.** The criterion names and descriptions are v1's clinician-written
      Thai. Have a clinician check they read correctly for อสม. and the general public —
      terms like "ผื่นลูปัสกึ่งเฉียบพลัน" may need lay phrasing.
- [ ] **Dry run on a real phone** in daylight: tap targets, Thai text at large sizes, and
      whether the photographs are legible outdoors.
- [ ] **Check the sheet** after the dry run: rows present, one per submission, and the
      test-mode link produced none.
