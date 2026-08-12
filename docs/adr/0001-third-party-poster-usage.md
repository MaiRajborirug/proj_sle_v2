# 0001 — Third-party posters on the บทความ page

- **Date**: 2026-08-13
- **Status**: Accepted, provisional — revisit when the permission requests below are answered
- **Decision owner**: project team (clinical lead: napat.po@kmitl.ac.th)

## Context

The บทความ page shows public-health infographics about SLE so booth visitors have
something to read. Four candidate posters were collected. None of them were produced by
this project, and the licence position of each was checked on 2026-08-13:

| Poster | Owner | Licence found |
|---|---|---|
| `poster_doccouncil.png` | กรมการแพทย์ กระทรวงสาธารณสุข / Thai PBS | none stated — all rights reserved by default |
| `poster_payathai.jpg` | โรงพยาบาลพญาไท | none stated — all rights reserved by default |
| `poster_chula9.png` | โรงพยาบาลจุฬารัตน์ 9 แอร์พอร์ต (CHG) | none stated — all rights reserved by default |
| `poster_thaihealth.png` | สสส. / Creative Citizen | **reuse expressly prohibited** without prior written consent |

Two other sources were checked while looking for a licence:

- **doctor.or.th** (มูลนิธิหมอชาวบ้าน) publishes its health media under **CC BY-NC-SA 3.0**.
  Reusable with attribution for non-commercial purposes. No SLE-specific item has been
  located there yet.
- **kalong.go.th** carries an SLE article with no attribution of its own. It is a
  downstream re-publisher, not a licence source, and must not be cited as one.

Attribution is not permission. Thai copyright law §33 covers quoting *part* of a work with
acknowledgement; reproducing a whole infographic falls back on §32, which is a judgement
call rather than a clear exemption. Crediting the owner is necessary but not sufficient.

## Decision

1. **`poster_thaihealth.png` is removed.** Creative Citizen prohibits reuse in writing, so
   showing it would be a knowing infringement rather than an unclear one.
2. **The remaining three are shown with attribution and a link to the original**, plus a
   note at the head of the page stating that the images belong to their owners and are
   shown for non-commercial education.
3. **Permission is requested in parallel** (plan below). The app ships without waiting.
4. **`articles.json` records the licence status of every poster**, matching how
   `criteria_d9.json` records the provenance of the patient photographs. The status string
   must be updated when a request is answered.

## Permission requests

Ask for: permission to display the poster inside a free, non-commercial SLE screening web
app run by คณะแพทยศาสตร์ สจล. and อสม.คลองเขื่อน as a CSR outreach project, with credit and
a link back to the source; and whether the material may be reproduced in a subsequent
academic publication about the project.

| # | Who | Channel | Notes |
|---|---|---|---|
| 1 | โรงพยาบาลจุฬารัตน์ 9 แอร์พอร์ต | Facebook page `ch9airport` (the post the image came from), or the hospital's PR contact | Most likely to answer quickly; the post is public outreach material |
| 2 | โรงพยาบาลพญาไท | Contact form / PR at phyathai.com, referencing article 3817 | |
| 3 | Thai PBS + กรมการแพทย์ | Thai PBS audience contact; กรมการแพทย์ PR | Two owners on one image — needs the original URL located first, which is still outstanding |
| 4 | มูลนิธิหมอชาวบ้าน (doctor.or.th) | Not needed — CC BY-NC-SA 3.0 already permits this use | Only needs correct attribution and the same licence on the page |

Outcomes:

- **Yes** → update the `licence` field in `articles.json` with the date and who granted it,
  and keep the file of the reply with the project records.
- **No, or no reply by the event date** → drop that poster and replace the slot with
  content written by the project team. Medical facts are not copyrightable; only the
  specific wording and design are. Project-written content also removes the oddity of
  showing four other hospitals' branding on a KMITL/อสม. booth.

## Consequences

- The page ships now, which is why this ADR is provisional rather than a clean decision.
- Until permission arrives, a takedown request is a live possibility. The blast radius is
  one JSON entry and one image file, so complying takes minutes.
- **The repository must stay private.** Making it public would republish these posters via
  GitHub as well, and they are already in git history from commit `fa00da5` — history
  would need rewriting before any move to a public repo, unless permission has arrived by
  then.
- The ethics/IRB application and any resulting paper should state that third-party
  educational images were used with permission, and name which. That is the reason this
  file is committed rather than kept locally.
