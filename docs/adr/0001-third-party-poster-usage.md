# 0001 — Content for the บทความ page

- **Date**: 2026-08-13
- **Status**: Accepted
- **Decision owner**: project team (clinical lead: napat.po@kmitl.ac.th)

## Context

The บทความ page exists so booth visitors have something to read about SLE. It was first
built from four public-health infographics collected by the project team. None of them
were produced by this project, and the licence position of each was checked on 2026-08-13:

| Poster | Owner | Licence found |
|---|---|---|
| `poster_doccouncil.png` | กรมการแพทย์ กระทรวงสาธารณสุข / Thai PBS | none stated — all rights reserved by default |
| `poster_payathai.jpg` | โรงพยาบาลพญาไท | none stated — all rights reserved by default |
| `poster_chula9.png` | โรงพยาบาลจุฬารัตน์ 9 แอร์พอร์ต (CHG) | none stated — all rights reserved by default |
| `poster_thaihealth.png` | สสส. / Creative Citizen | **reuse expressly prohibited** without prior written consent |

Two further sources were checked while looking for a licence:

- **doctor.or.th** (มูลนิธิหมอชาวบ้าน) publishes its health media under **CC BY-NC-SA 3.0**.
  Reusable with attribution for non-commercial purposes. No SLE-specific item was located.
- **kalong.go.th** carries an SLE article with no attribution of its own. It is a
  downstream re-publisher, not a licence source, and must not be cited as one.

Attribution is not permission. Thai copyright law §33 covers quoting *part* of a work with
acknowledgement; reproducing a whole infographic falls back on §32, which is a judgement
call rather than a clear exemption. An intermediate revision earlier the same day dropped
the prohibited poster and showed the other three with credit, links and a non-commercial
notice. That is a defensible position, but it is still a position that would have to be
argued if challenged.

## Decision

**The บทความ page carries content written by the project team. No third-party image is
reproduced.** All four posters are removed from the app and the repository.

Medical facts are not copyrightable; only the specific wording and design are. Writing the
content removes the question entirely rather than answering it well. The four sources
above are cited as further reading, with links — citing and linking are always permitted.

Content lives in `article_th.md` as Markdown so a clinician can review and edit it without
touching code. `test_article_embeds_no_images` enforces this decision mechanically, so
adding an image means deliberately changing a test rather than quietly editing a data file.

Secondary benefits, which are not the reason but do matter here: text scales with the app's
21px base font and pinch-zoom, where a poster's baked-in small print does not; the page
drops from ~2.2 MB to a few KB on a booth's mobile connection; and the booth stops showing
four other hospitals' branding above its own sponsors' logos.

## If posters are wanted later

Permission has not been requested. If the team decides the visuals are worth it, ask for:
permission to display the poster inside a free, non-commercial SLE screening web app run by
คณะแพทยศาสตร์ สจล. and อสม.คลองเขื่อน as a CSR outreach project, with credit and a link
back; and whether it may be reproduced in a subsequent academic publication.

| Who | Channel | Notes |
|---|---|---|
| โรงพยาบาลจุฬารัตน์ 9 แอร์พอร์ต | Facebook page `ch9airport` (the post the image came from), or the hospital's PR contact | Most likely to answer quickly; the post is public outreach material |
| โรงพยาบาลพญาไท | Contact form / PR at phyathai.com, referencing article 3817 | |
| Thai PBS + กรมการแพทย์ | Thai PBS audience contact; กรมการแพทย์ PR | Two owners on one image, and the original URL was never located |
| มูลนิธิหมอชาวบ้าน (doctor.or.th) | No request needed — CC BY-NC-SA 3.0 already permits this use | Needs correct attribution and the same licence on the page |

On a written yes, restore the image and record the grant and its date here.

## Consequences

- **`article_th.md` states medical guidance in the project's name and needs clinical
  sign-off before the first event.** Nobody outside the project can be pointed at if a
  statement is wrong. This is the cost of the decision and it is not optional.
- The four posters remain in git history (commits `fa00da5` and `53d79c5`). The repository
  is private, so they are not published; **they must be purged from history before the
  repository is ever made public.**
- The ethics/IRB application and any resulting paper can state that all patient-facing
  educational content is original to the project. That is the reason this file is
  committed rather than kept locally.
