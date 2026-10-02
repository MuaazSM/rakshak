# Rakshak brand files

Traced from the chosen र concept and rebuilt as clean vectors: one ink headline bar (the doorstep line) and one haldi thread hanging from it, ending in a raksha-sutra knot. Bar and thread share the same weight. See PRD §18.4.

Colors: ink `#1C1917` · haldi `#F5C518` · white `#FFFFFF` · crimson `#9F1239` (badge only).

| File | Use |
|---|---|
| `rakshak-icon-ink.svg` / `-1024.png` | **Main app icon** (full-bleed square; iOS and Android apply their own corner mask) |
| `rakshak-icon-ink-rounded.svg` / `-1024.png` | Rounded-tile version for docs, slides, the DEV post |
| `rakshak-icon-small.svg` | Thicker strokes for ≤ 48 px (favicons, notification icons) |
| `rakshak-icon-small-rounded.svg` | Same, rounded preview |
| `rakshak-icon-maskable.svg` / `pwa-icon-maskable-512.png` | Android adaptive/maskable PWA icon (mark inside the safe zone) |
| `pwa-icon-192.png`, `pwa-icon-512.png` | PWA manifest icons |
| `apple-touch-icon-180.png` | iOS home screen / Shortcut icon reference |
| `favicon-16/32/48.png` | Browser favicons (from the small version) |
| `rakshak-mark-light.svg` / `-1024.png` | Mark on light backgrounds (README, docs), no tile |
| `rakshak-mark-mono-ink.svg`, `rakshak-mark-mono-white.svg` | One-color versions |
| `rakshak-badge-crimson.svg` / `-1600.png` | Outside layer only: jharokha badge with block-print band |
| `preview.png` | All variants and 180 / 60 / 24 px checks |

PWA manifest snippet:
```json
"icons": [
  {"src": "/brand/pwa-icon-192.png", "sizes": "192x192", "type": "image/png"},
  {"src": "/brand/pwa-icon-512.png", "sizes": "512x512", "type": "image/png"},
  {"src": "/brand/pwa-icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}
]
```

Rules: don't recolor, stretch or add a second horizontal bar (it would read as ₹). Wordmark is set in a real font (Yatra One + Teko), not drawn.
