# PodClipper website

React + Vite site at [podclipper.loukik.dev](https://podclipper.loukik.dev).

The `#/try` workspace uses Google-only sign-in against the separate laptop API.
Set `VITE_API_URL` in `.env.local` for development; production defaults to
`https://podclipper-api.loukik.dev`. The Google client ID comes from the API's
public `/config` endpoint, keeping frontend and backend settings aligned.
Jobs live at `#/jobs/JOB_ID`, which works on GitHub Pages without server rewrites.
The configured administrator also gets `#/admin` and `#/admin/jobs/JOB_ID` for a
read-only view of all accounts' jobs and reels. Authorization is enforced by the
API using `WEB_ADMIN_GOOGLE_SUB`; no admin identity is bundled into the frontend.
The public privacy page is `/privacy/`. See `../backend/README.md` for deployment.

## Develop

From this directory:

```bash
npm ci
npm run dev
```

Use Node.js 24. `npm test` checks admin list/detail navigation,
direct job links, and browser Back/Forward without refreshing the page.

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
