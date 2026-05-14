# Landing Page

Static front-end landing page for Agentic Video Editor. Drop it into any GitHub Pages repo.

## Setup

1. **Copy your 3 demo videos** into `landing/videos/`:

```
landing/
  index.html
  videos/
    reel_01_sitas-challenge-ram-couldnt-ignore.mp4   ← from outputs/2026-04-19_03-01-56/
    reel_01_7-lakh-crores-of-gold-looted.mp4          ← from outputs/
    reel_03_id-trade-my-kingdom-for-her.mp4            ← from outputs/2026-04-19_00-48-25/
```

2. **Update the GitHub URL** — search for `loukiknaik/agentic-video-editor` in `index.html`
   and replace with your actual repo path.

3. **Deploy** — push the `landing/` folder contents as the root of a GitHub Pages branch
   (e.g. `gh-pages`), or place `index.html` + `videos/` at the root of a portfolio repo.

## Notes

- Videos autoplay muted and loop. No JavaScript libraries required.
- Fonts are loaded from Google Fonts (requires internet).
- Page is fully responsive down to 375px wide.
