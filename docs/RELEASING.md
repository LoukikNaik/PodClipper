# Publishing PodClipper to PyPI

The workflow at `.github/workflows/publish-pypi.yml` publishes each new
`vX.Y.Z` tag to PyPI. For example, tag `v0.1.1` publishes package version
`0.1.1`. Version numbers are generated at build time by setuptools-scm;
you do not edit `pyproject.toml` or `src/podclipper/__init__.py` for a release.

## One-time setup

1. Create an account at [PyPI](https://pypi.org/account/register/), verify your
   email, and complete PyPI's required account security setup.
2. In GitHub, open **PodClipper → Settings → Environments** and create an
   environment named **`pypi`**. If you add a deployment branch/tag rule, allow
   the release tags (`v*`); a main-only rule will block tag publishing.
3. Open [PyPI account Publishing](https://pypi.org/manage/account/publishing/)
   and add a pending **GitHub** publisher with these exact values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `podclipper` |
   | Owner | `LoukikNaik` |
   | Repository | `PodClipper` |
   | Workflow filename | `publish-pypi.yml` |
   | Environment | `pypi` |

No PyPI API key or GitHub Actions secret is needed. The first successful tag
upload creates the project. A pending publisher does not reserve the package
name; if it is taken before your first upload, choose another distribution name
and update the package metadata, workflow project URL, release validator, and
these instructions. For an existing project you own, add the trusted publisher
from that project's **Manage → Publishing** page instead.

Official references: [creating a project with Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [publishing from GitHub Actions](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## First release

Commit and push the release setup to `main` first. Then, from a clean checkout:

```bash
git switch main
git pull --ff-only origin main
git status --short
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

Only tag when `git status --short` is empty and the desired fixes are committed.
Open **GitHub → Actions → Publish to PyPI** to follow the build and upload.
The build validates the README, license, version, required resources, and archive
contents before uploading. The wheel is built from the source distribution.
Source archives exclude videos, local configs, transcripts, and model weights.

You can also run **Actions → Publish to PyPI → Run workflow** on `main` to
build and validate distributions without publishing. Manual runs never upload,
even when a tag is selected.

## Releasing fixes

Commit and push your fixes, then repeat with a **new** version tag:

```bash
git switch main
git pull --ff-only origin main
git status --short
git tag -a v0.1.1 -m "Release 0.1.1"
git push origin v0.1.1
```

Use a new patch version for fixes (`0.1.1`, `0.1.2`, ...). This workflow accepts
stable `vX.Y.Z` tags only. Do not move or overwrite an existing release tag.
PyPI does not let you replace an uploaded distribution file with new contents.
If a build or authentication step fails before uploading, correct the setup and
rerun the failed workflow. If files were already uploaded, inspect PyPI first;
release corrected code under a new version rather than replacing old files.

## Installing after publication

Use Python 3.10–3.12. Install FFmpeg separately; pip cannot install that system
tool. For example, on macOS:

```bash
brew install ffmpeg pipx python@3.12
pipx ensurepath
pipx install --python python3.12 podclipper
podclipper init
# Fill in LLM_API_KEY and, optionally, LLM_ENDPOINT / LLM_MODEL in .env.
podclipper podcast.mp4
```

If `pipx ensurepath` asks you to reopen your terminal, do that before running
`podclipper`. Existing users upgrade with `pipx upgrade podclipper`.
Inside an activated Python environment, `pip install podclipper` works too.

## Homebrew releases

The public tap is [LoukikNaik/homebrew-tap](https://github.com/LoukikNaik/homebrew-tap).
It supports Apple Silicon Macs running macOS 14 or newer:

```bash
brew install loukiknaik/tap/podclipper
```

Homebrew installs Python 3.12 and FFmpeg and keeps PodClipper's Python packages
in an isolated environment. Users upgrade with `brew update` followed by
`brew upgrade podclipper`.

After a PyPI release succeeds, the tap's **Update PodClipper** workflow checks
PyPI every six hours, resolves compatible wheels, and commits the updated formula
and dependency hashes. No extra repository token or PyPI credential is needed.
For an immediate update, open the tap's **Actions → Update PodClipper → Run
workflow** and enter the published version, such as `0.1.1`, without `v`.
Check that workflow finishes successfully before announcing Homebrew availability.

The tap uses pinned upstream binary wheels for native ML dependencies. It is
maintained separately from Homebrew's core repository; users install with the
full `loukiknaik/tap/podclipper` name.

## License

New releases use **AGPL-3.0-only**, including the full `LICENSE` and attribution
in `NOTICE`. Ultralytics YOLO remains the person detector. This change governs
new releases; it does not revoke any permissions already granted for older
versions. Dependencies and downloaded assets keep their own licenses.
