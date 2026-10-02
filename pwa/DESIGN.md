# PWA design map

Visual source of truth: the Claude Design export in `docs/design/Design system and mobile flows/`
(git-ignored, read-only). Frames are rebuilt as React components using only `src/styles/tokens.css`.

| Frame | Design file | Route / component | Notes |
|---|---|---|---|
| A1-A11 | `A Design System.dc.html` | `src/styles/tokens.css`, `src/components/*` | Tokens, type scale, buttons, verdict card, chips, hold-to-talk, block-print strip, spacing. |
| B1 First open | `B Mom Android.dc.html` | `/` -> `screens/Install.tsx` | Shown once in the browser (not standalone, install not yet seen). Button triggers the browser's install prompt. |
| B2 Home | `B Mom Android.dc.html` | `/home` (and `/` once installed) -> `screens/Home.tsx` | Block-print strip, wordmark, gear, two actions, recently checked (verdict + time only). |
| B3 Share sheet | `B Mom Android.dc.html` | none (Android system UI) | Provided by the manifest `share_target`; POST `/share` is handled by the hub, which answers 303 to `/v/{id}`. |
| B4 Check | `B Mom Android.dc.html` | `/check` -> `screens/Check.tsx` | Paste text or choose a screenshot; Check disabled until one is present. |
| B5 Checking | `B Mom Android.dc.html` | shown by `Check`, `Talk` and `/v/:id` while waiting -> `screens/Checking.tsx` | One yellow glide bar; stage list is timed (the hub sends no stage events). |
| B6 Verdict SCAM | `B Mom Android.dc.html` + `Verdict.dc.html` | `/v/:id` -> `screens/Verdict.tsx` | `verdict = SCAM`. |
| B7 Verdict BE CAREFUL | same | `/v/:id` | `verdict = SUSPICIOUS`. |
| B8 Verdict LOOKS NORMAL | same | `/v/:id` | `verdict = SAFE`; primary action is OK (back to Home). |
| B9 Couldn't check | same | `/v/:id`, `/v/offline` | `verdict = UNKNOWN`, or the hub didn't answer within 20 s (`/v/offline`). Never a guess. |
| B10a/b/c Describe a call | `B Mom Android.dc.html` | `/talk` -> `screens/Talk.tsx` | Idle, recording (held), cancelled (slid left past 80 px). Release sends, then B5. |
| B11 Settings | `B Mom Android.dc.html` | `/settings` -> `screens/Settings.tsx` | One row, Explanation language; PATCH `/api/parents/{id}`. |
| C1-C3, C5 | `C Dad iPhone.dc.html` | none | iOS Shortcuts and native alerts, not the PWA. |
| C4a/b Safari verdict | `C Dad iPhone.dc.html` + `Verdict.dc.html` (variant `ios`) | `/v/:id` in iOS Safari at 390x844 | Same component; iOS UA drops the back arrow (Safari's own Done bar is the chrome). Auto-play is blocked there, so Listen again takes the focus ring. |
| D1-D* Son status | `D Son.dc.html` | not in the PWA | Served by the hub status page. |
| E Outside layer | `E Outside Layer.dc.html` | not in the PWA | Marketing only. |

Data: `src/i18n/en.json` holds all copy keyed by frame (`B1.*`, `B2.*`, `B4.*`, `B5.*`, `verdict.*`, `flags.*`,
`B10.*`, `B11.*`). `hi.json` mirrors the keys with `TODO` values; `t()` falls back to English for those.

Not in the designs (nearest designed pattern used): Home empty state, "Screenshot added" row on B4, mic-permission
and too-short notices on B10, install hint when the browser has no install prompt, settings save error.
