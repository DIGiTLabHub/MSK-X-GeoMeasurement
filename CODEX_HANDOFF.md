# MSK-X-GeoMeasurement handoff

Updated: 2026-11-02

## Project and source of truth

- Repository: `DIGiTLabHub/MSK-X-GeoMeasurement`
- Production site: `https://digitlabhub.github.io/MSK-X-GeoMeasurement/`
- GitHub Pages source: `main` branch, `docs/` folder. Keep `docs/.nojekyll`.
- Website requirements: `MSK-X-GeoMeasurement_Website_Spec.md` in the repository root. This file was supplied by the user and is currently untracked.
- The repository holds two separate research works in `paper-1/` and `paper-2/`. Their public project page paths, `docs/paper-1/` and `docs/paper-2/`, should remain stable.

## Completed

Commit `a5eb5f5` (`Redesign research website and paper pages`) was pushed to `origin/main` on 2026-11-02. It contains:

- `docs/index.html`: research portal with navigation, hero, exactly two publication cards, project resources, and footer.
- `docs/paper-1/index.html` and `docs/paper-2/index.html`: matching paper pages with publication, overview, workflow, and code sections.
- `docs/assets/css/site.css`: shared responsive styling, visible keyboard focus, and reduced-motion support.

The pages use relative internal links for the GitHub Pages project subpath. Paper titles, metadata, figures, DOI, and PDF links remain clearly marked as forthcoming; none were invented. No research folders, root README, `.nojekyll`, or Pages settings were changed.

## Verification and limits

- `git diff --check` passed before commit.
- A local HTML parser check found one H1 per page, exactly two homepage paper cards, and no missing relative links or anchors among the three pages.
- Source inspection found no root-absolute paths, remote scripts, CDN references, or fabricated DOI links.
- Visual browser QA is **not complete**. Ego Light opened a signed-out screen, and the localhost preview did not load through this environment. Responsive layout and live GitHub Pages rendering therefore need a real browser check.
- Ego Light reported that an update was available; no upgrade was installed.

## Current working tree

At handoff, `main` matches `origin/main`. The only pre-existing untracked file is `MSK-X-GeoMeasurement_Website_Spec.md`. This handoff file is also untracked and has not been committed or pushed.

## When work resumes

1. Inspect `git status --short --branch` and read the spec and this handoff before editing.
2. Open the production site or a local `docs/` preview in Ego Light when access works. Check homepage and both paper pages at approximately 320, 375, 768, 1024, and 1440 px, including link navigation, overflow, focus, and contrast.
3. Replace placeholders only with user-supplied or verified paper titles, authors, venue, year, DOI/PDF links, methods, figures, and code instructions. Preserve evidence boundaries and the separate paper URLs.
4. Follow the user's next specific request. Do not commit or push new changes without authorization.
