# MSK-X-GeoMeasurement — GitHub Pages Website Implementation Specification

**Audience:** Codex working locally inside the existing repository  
**Repository:** `https://github.com/DIGiTLabHub/MSK-X-GeoMeasurement`  
**Production site:** `https://digitlabhub.github.io/MSK-X-GeoMeasurement/`  
**Status:** Implementation specification, not a claim that changes have been deployed.

## 1. Objective

Upgrade the existing basic GitHub Pages site into a polished, accessible, responsive academic project portal for **MSK-X-GeoMeasurement**, a research initiative on AI-enabled geometric measurement from musculoskeletal X-ray images. It serves as the umbrella site for **two separate research papers/works**, each with a dedicated publication webpage and a separate conventional code/README/examples area.

**Critical constraint:** This is an in-place improvement of a *working* repository. Do not recreate the repository, change its owner/name, alter the existing GitHub Pages publishing source (`main` branch, `/docs` folder), rename `paper-1` or `paper-2`, or modify any scientific code. Do not commit or push unless explicitly asked by the user.

## 2. Existing repository facts (verified 2026-11-02)

At root: `README.md`, `.gitignore`, `docs/`, `paper-1/`, `paper-2/`, `shared/`. The current `docs/index.html` is a standalone basic HTML page containing an overview and two links to `paper-1/` and `paper-2/`. The public URL already works per the project owner. Before writing code, **inspect the actual local working tree and preserve any changes not reflected in this spec**.

Expected starting layout (adjust to match local tree):

```text
MSK-X-GeoMeasurement/
├── README.md
├── docs/
│   ├── index.html
│   ├── .nojekyll
│   ├── paper-1/index.html
│   ├── paper-2/index.html
│   └── assets/
├── paper-1/
│   ├── README.md
│   ├── src/
│   ├── examples/
│   ├── configs/
│   └── requirements.txt
├── paper-2/
│   ├── README.md
│   ├── src/
│   ├── examples/
│   ├── configs/
│   └── requirements.txt
└── shared/
```

## 3. Technical approach

- Use **semantic HTML5 and modern vanilla CSS**, with **zero required build steps and zero runtime dependencies**. Vanilla JavaScript is optional only if a feature genuinely requires it.
- Keep pages directly publishable from `docs/` with GitHub Pages; retain `.nojekyll` if present.
- Use a common stylesheet at `docs/assets/css/site.css`. Optional shared JavaScript at `docs/assets/js/site.js` only if necessary.
- All internal website paths must be **relative** and work under the GitHub Pages project subpath `/MSK-X-GeoMeasurement/`. Examples: from the homepage use `paper-1/`, `paper-2/`, and `assets/css/site.css`; from paper pages use `../`, `../assets/css/site.css`, and `../assets/img/...`. Do not use root-absolute paths like `/paper-1/`.
- GitHub *source* links may be absolute: `https://github.com/DIGiTLabHub/MSK-X-GeoMeasurement/tree/main/paper-1` (similarly for paper-2).
- Do not add React, Jekyll, Astro, Tailwind CDN, remote analytics, trackers, web fonts, or third-party JavaScript packages unless explicitly requested.
- Keep source readable, modular, and easy to edit manually.

## 4. Visual design

**Direction:** A clean, credible university research website, closer to a modern research-group publication portal than a software marketing landing page.

- White or near-white primary background; deep navy headings; blue links and restrained blue accents; muted gray supporting text.
- Suggested design tokens (modify if required for contrast): navy `#17365D`, blue `#1769AA`, body `#243447`, muted `#5F6B7A`, soft background `#F5F8FC`, border `#DDE4EC`.
- A centered content container, maximum width ~1120 px, with generous but controlled whitespace.
- System font stack for speed, legibility, and no external network requirement.
- Rounded paper cards, subtle borders, modest shadow or hover outline; no exaggerated animation.
- Responsive: two equal columns for paper cards on wide screens, a single column on smaller screens; all content remains readable at 320 px viewport width.
- Respect `prefers-reduced-motion` and visibly distinguish keyboard focus.

### 4.1 Global header

- Left: text-based **MSK-X-GeoMeasurement** wordmark, linking to homepage. Small secondary affiliation label **DIGiTLabHub** (link to organization).
- Right desktop navigation: **Overview**, **Publications**, **Repository**. Overview goes to homepage top, Publications to homepage section anchor, Repository to the GitHub repo.
- On mobile, use a simple wrapping nav or stacked navigation, not a hamburger menu that requires JavaScript.
- Header should be consistent on homepage and both paper pages.

### 4.2 Homepage hero

- Small eyebrow: `DIGiTLabHub / Research Project`.
- H1: `MSK-X-GeoMeasurement`.
- Subtitle: `AI-enabled geometric measurement from musculoskeletal X-ray images`.
- One short paragraph explaining this single portal links research publications, open implementations, and reproducible examples. Avoid unsupported claims about model accuracy, clinical application, or findings.
- Optional small non-interactive topic chips: `Medical Imaging`, `Geometric AI`, `Reproducible Research`.
- Two prominent text links: `Explore Publications` (homepage anchor) and `View GitHub Repository` (external GitHub URL). These must be functional anchors, not inert buttons.
- Do not use synthetic/fabricated medical images as though they were actual research output. Hero may be text-only until approved illustrations/figures exist.

### 4.3 Publications section

- Heading `Publications & Software` with a one-line introduction.
- Exactly two cards, maintaining independently editable metadata:
  - Card 1: `Paper 1` unless a real title is present in existing content.
  - Card 2: `Paper 2` unless a real title is present in existing content.
- Each card includes a research figure area (use a tasteful **CSS placeholder** with a clear `Figure forthcoming` label if no approved figure exists), title, publication status label only if known, brief neutral placeholder description, and links:
  - `Project page` → `paper-1/` or `paper-2/`.
  - `Code & examples` → corresponding GitHub folder.
  - `Paper / DOI` → **only if a real verified link exists**, otherwise omit; do not create dead anchor links.
- Do not assign specific anatomy or imaging modalities to each paper without verified information.
- Cards must be independently editable and remain legible on phones.

### 4.4 Project resources / reproducibility strip

- A short section explaining that each work has its own environment, source code, configuration, and running examples.
- Text links to `paper-1/README.md` and `paper-2/README.md` **on GitHub** (rather than attempting to serve raw README from Pages). Example: `https://github.com/DIGiTLabHub/MSK-X-GeoMeasurement/blob/main/paper-1/README.md`.
- Make no claims that code is complete or executable until verified.

### 4.5 Footer

- Project name, `DIGiTLabHub`, link to repository, and short text noting that pages link to distinct research works.
- No fabricated copyright years, affiliations, paper authors, or licensing terms. If including current year, generate/update deliberately and do not imply a LICENSE file exists.

## 5. Dedicated paper webpages

Implement **`docs/paper-1/index.html`** and **`docs/paper-2/index.html`** using the same header, typography, spacing, and footer.

Each page must contain:

1. Breadcrumb: `Home / Paper 1` or `Home / Paper 2`, with working home link `../`.
2. H1 placeholder title and a concise `Details forthcoming` note if content is not yet available.
3. Publication metadata area designed for later **authors, venue, year, DOI, PDF**, but omit any empty or unverified links.
4. `Overview` section with neutral placeholder copy.
5. `Method / Workflow` placeholder area for approved research workflow graphic and explanation.
6. `Code & Reproducibility` section linking to the paper's GitHub code directory and `README.md`, plus future anchor points for Installation, Data, Running Examples, Evaluation, and Citation.
7. `Back to all publications` link to `../#publications`.

Do not copy guessed technical methods, metrics, medical assertions, or paper titles into these pages. Keep a straightforward template that can be populated once paper material is supplied.

## 6. File deliverables

Edit or add only as needed:

```text
docs/index.html                        # redesigned unified homepage
docs/paper-1/index.html                # coordinated paper 1 project page
docs/paper-2/index.html                # coordinated paper 2 project page
docs/assets/css/site.css              # single shared stylesheet
# optional only if needed:
docs/assets/js/site.js
```

Do not overwrite any real figures already present in `docs/assets/`. Do not reorganize or modify research code folders.

## 7. Accessibility and metadata

- One H1 per page, semantic `header`, `nav`, `main`, `section`, and `footer`; meaningful heading order.
- All links keyboard-operable with clear focus outlines; avoid color-only distinctions.
- WCAG AA-level contrast target for normal text; maintain reasonable tap targets.
- Every real image has accurate descriptive alt text; decorative images have empty alt. A CSS placeholder does not need pretend image alt text.
- Browser `<title>` and `<meta name="description">` tailored to homepage vs paper pages.
- `lang="en"`, UTF-8, viewport meta, and suitable responsive styling.
- Internal links must not yield 404s. External links should be real and functional.

## 8. Stable URLs and future renames

Public webpage paths `docs/paper-1/` and `docs/paper-2/` should be treated as stable publication URLs, even if the *code directories* are eventually renamed to meaningful paper/project titles. To support this:

- Keep paper website links centralized or easy to locate in HTML.
- Keep separate GitHub code links per card and page; update those links if code directories change.
- Never automatically rename public paper URL folders just because code directories are renamed.
- If a public URL must change later, preserve an old-page redirect where feasible and update all links.

## 9. Validation / acceptance criteria

Codex should complete and self-check the following:

- [ ] Main site has academic-quality hero, navigation, exactly two attractive publication cards, project resources, and footer.
- [ ] Both individual paper webpages follow the same visual system and work independently.
- [ ] Existing production homepage URL and repository paths remain unchanged.
- [ ] All internal links work when the site is hosted *under the repository subpath*, not the root of a custom domain.
- [ ] GitHub links target actual paths on `main`.
- [ ] No broken images, guessed paper data, mock scientific findings, or fake DOI/PDF links.
- [ ] Works responsively at approximately 320, 375, 768, 1024, and 1440 px widths; no horizontal scrolling.
- [ ] Clear hover/focus states; semantic document structure; adequate contrast.
- [ ] Page builds locally without `npm install`, Python dependency install, or remote CDN.
- [ ] Existing `.nojekyll`, branch deployment, and any research code are left intact.
- [ ] Final response includes a brief change summary and the list of files changed.

## 10. Local preview, verification, and publishing

From repository root, Codex may advise these user-run commands:

```bash
# Optional isolated branch before working
git switch -c feature/research-website

# Serve docs locally
python3 -m http.server 8000 --directory docs
```

Local URLs for visual checking:

- `http://localhost:8000/`
- `http://localhost:8000/paper-1/`
- `http://localhost:8000/paper-2/`

**Important:** This localhost root preview is sufficient to inspect relative links, but the actual GitHub Pages URL uses a subpath; confirm that code uses no root-absolute references before deployment.

After reviewing and approving Codex changes, user may commit and push through their usual Git workflow. Codex should not do so on its own.

## 11. Exact instruction to Codex

> Implement this specification in my current checked-out `DIGiTLabHub/MSK-X-GeoMeasurement` repository. Inspect the actual files first. Preserve all existing code, `.nojekyll`, published URL structure, and Pages settings. Build the redesigned homepage and the two coordinated paper pages with semantic HTML and one shared CSS stylesheet, without new dependencies. Use accurate placeholder material where paper details or figures are not yet supplied. Test local navigation, links, and responsive behavior as far as your environment allows. Do not commit or push; show me the file changes and any limitations for review.
