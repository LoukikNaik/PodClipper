# PodClipper website

React + Vite site at [podclipper.loukik.dev](https://podclipper.loukik.dev).

The `/try` workspace uses Google-only sign-in against the separate laptop API.
Set `VITE_API_URL` in `.env.local` for development; production defaults to
`https://podclipper-api.loukik.dev`. The Google client ID comes from the API's
public `/config` endpoint, keeping frontend and backend settings aligned.
Jobs live at `/jobs/JOB_ID`, using browser history navigation. The build copies the app into `404.html` so direct links and refreshes work on GitHub Pages, matching the portfolio. Assets use root-relative paths. Old hash links are automatically replaced with clean URLs.
Users paste a YouTube link or choose a local video in the source panel. Podcast cards
show processing status and an authenticated reel preview when one is ready. The
workspace uses the existing blue palette and adapts to mobile screens.
Discovery and the duration-based reel cap are automatic;
the UI shows actual ready counts rather than a requested quota. Lists request
compact summaries; open podcasts fetch their own details. Busy views poll every
five seconds, idle libraries every thirty seconds, and hidden tabs skip polling.
Upload completion queues background validation while retaining the resume key.
The configured administrator also gets `/admin` and `/admin/jobs/JOB_ID` for a
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

### Themes and mobile checks

The landing page and workspace share a light/dark toggle. Light mode is
the default; explicit choices are saved locally and applied before React renders.
Private browsing or unavailable local storage does not block the toggle.

The mobile landing header uses an expandable navigation menu. Forms use 16px
input text to avoid focus zoom on iOS, primary controls have 44px minimum touch
targets, and both themes support reduced motion. `npm test` covers theme choice,
workspace routing, and resumable upload behavior. Visual checks should include
320px, 390px, 768px, and desktop widths, plus logged-out sign-in and reel details.
