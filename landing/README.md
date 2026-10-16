# PodClipper website

React + Vite site at [podclipper.loukik.dev](https://podclipper.loukik.dev).

## Develop

From this directory:

```bash
npm ci
npm run dev
```

`src/App.jsx` contains the page and installation commands. `src/App.css` keeps
the blue theme, Fraunces headings, and Instrument Sans body text. Demo videos
live in `public/videos/`; public assets are copied into the build unchanged.

## Deploy

```bash
npm run build
```

Push landing changes to `main`. `.github/workflows/deploy-landing.yml` builds
the site, sets its custom domain, and publishes `dist/` to the `gh-pages` branch.
The workflow also supports a manual dispatch.

Keep Homebrew/pip requirements and the shared `podclipper init` flow aligned
with the root README and the published package. The page links to the PyPI
package, Homebrew tap, configuration docs, releases, issues, and AGPL license.
