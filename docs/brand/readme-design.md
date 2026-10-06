# README design

The full draft is [`README.draft.md`](README.draft.md) (written for the repository root; it is not
installed there yet). This note explains its structure and the GitHub constraints behind it, so later
edits keep the same quality.

## Who reads it, and in what order

A visitor decides in about 10–20 seconds. A mathematician or researcher looks for four things, in
this order: *can I trust it* (status, license, citation), *is it correct* (the figure, the notation,
the stopping tests), *what is in it* (methods, families), *how do I run it* (three lines). The page
answers in that order and puts everything else below the fold.

| # | Section | Job | Length budget |
|---|---|---|---|
| 1 | Wordmark + tagline | Identity; the one-line promise | 2 lines |
| 2 | One quiet badge row | Trust signals researchers scan for first: Python ≥ 3.11, MIT, *Cite* (CITATION.cff); tests and DOI when they exist. Flat, in the brand's neutrals (`#52514e` / `#6b6963`), at most five | 1 line |
| 3 | Link row | Labs, quick start, methods, research, architecture, cite | 1 line |
| 4 | **Hero figure** (animated SVG; wide and stacked; light and dark) + caption | Proof by demonstration: four real methods under one stopping test, their real iteration counts, linear vs superlinear vs quadratic convergence visible as *shapes* | 1 screen |
| 5 | **Open the interactive labs →** | The most impressive part gets one prominent line right under the figure | 1 line |
| 6 | One-liner + one paragraph | "168 numerical methods, each cited to its algorithm and equation, tested against an oracle, and replayable iterate by iterate in the browser." | ≤ 80 words |
| 7 | Quick start | Install + four lines of Python + a compact, fair comparison with its real output | < 20 seconds to read |
| 8 | What's inside | Family table with live counts and named examples; the `Result`/`Step` contract in one paragraph | 1 table |
| 9 | Gallery | Four lab screenshots (placeholders until the labs ship) | 2 × 2 grid |
| 10 | Design principles | Cited · checked · honest about failure · one implementation in two languages · readable | 5 bullets |
| 11 | Research | One row per note in `research/`: title, one-line finding, link (generated) | grows |
| 12 | Repository map, citing, license | Where things live; BibTeX and CITATION.cff | 1 table + 1 block |

What is deliberately **absent**: a badge wall (one quiet row of four facts is the limit), a
feature checklist with emoji, a "Why numopt?" marketing section, a table of contents (GitHub renders
one from the headings), and installation alternatives above the fold.

## Package name

The PyPI name `numopt` already belongs to another project (numopt 0.0.7, an engineering-design
package whose import name is `NumOpt`). A visitor who copies `pip install numopt` would install the
wrong package. Checked on 2026-10-05 through `https://pypi.org/pypi/<name>/json`: `numopt-lab`,
`numoptlab`, `numopt-ref`, `numopt-reference` and `pynumopt` are free.

* **Recommendation:** publish as **`numopt-lab`** (it matches "interactive labs") and keep
  `import numopt`. The other project's module is `NumOpt`; Python imports are case-sensitive, so the
  only clash is a user who installs *both* on a case-insensitive file system (macOS, Windows), where
  the two directories merge. That risk is small; note it in the package's FAQ.
* **Decided (2026-10-06):** `pyproject.toml` names the distribution `numopt-lab`; the import name,
  the CLI and the repository stay `numopt`. Until a PyPI release exists, every install line is the
  name-free `pip install git+https://github.com/ML-Dev-Hub/numopt`.

## The hero

Revised 2026-10-06 after owner review: the first hero (Rosenbrock, near-square, animated, with an
off-view Newton step and a convergence axis down to 10⁻²⁴) read as tall and "vertically weird".

* **One wide figure** (960 × 394, aspect ≈ 2.4:1): Himmelblau's function from 𝐱₀ = (−3.75, 2.5),
  where gradient descent (Armijo backtracking), heavy-ball momentum, BFGS and damped Newton take four
  clearly different routes to the same minimizer 𝐱⋆ ≈ (−2.805, 3.131). Every iterate stays inside
  the panel (`make_hero.py` asserts it).
* **Convergence panel plots ‖∇f(𝐱ₖ)‖₂**, the quantity the stopping test reads, on a log-k axis.
  The dashed stopping line (10⁻⁸) is the floor of the chart, so every curve ends at the line instead
  of plunging; Newton's steep final step is labelled *quadratic*.
* **Static**, transparent background (native on GitHub's #ffffff and #0d1117), light and dark, plus
  a stacked 540 × 876 variant for viewports ≤ 640 px. Real traces from `numopt.run`; the caption
  states the problem, 𝐱₀ and the single stopping test.

| File | Size | Use |
|---|---|---|
| `readme-hero/hero-{light,dark}.svg` | ~177 kB | README, desktop |
| `readme-hero/hero-stacked-{light,dark}.svg` | ~178 kB | README on viewports ≤ 640 px |

Regenerate with `.venv/bin/python docs/brand/scripts/make_hero.py`.

## The compact comparison

The CLI's `numopt compare` output is about 200 characters wide, mixes ‖∇f(x)‖ and ‖∇f‖∞, nests
parentheses and says `max_iter=5000`. The README therefore shows a five-line Python loop that prints
method · iterations · f(x) − f⋆ · stop, with its real output, and states which norm each method's
`gtol` tests. **Request to the package team** (outside `docs/brand/`): add `numopt compare
--compact` with those four columns, and one message format for every method — "‖∇f‖₂ = 9.98×10⁻⁹
≤ 10⁻⁸", "Stopped at the 5,000-iteration budget · ‖∇f‖₂ = 1.17×10⁻³ > 10⁻⁶" — so the README can show
the CLI again.

## GitHub rendering constraints (and how the assets meet them)

* **README SVGs render through `<img>`:** no scripts, no external fonts, no external CSS. Every glyph in
  the hero, wordmark and social card is therefore converted to outlines with fontTools
  (`scripts/typeset.py`), using Inter, Newsreader and the KaTeX fonts (including KaTeX Main Bold for
  the bold vectors) — the figure's math is real Computer Modern, identical on every OS.
* **SMIL and CSS animations inside an SVG do run in `<img>`**, and media queries inside the SVG
  (`prefers-reduced-motion`) are honored. Keep each animated file under ~1 MB so it is not lazily
  replaced; ours are ~350 kB.
* **Theme and width switching** use one `<picture>` with four `<source>` elements, most specific
  first: `(max-width: 640px) and (prefers-color-scheme: dark)`, `(max-width: 640px)`,
  `(prefers-color-scheme: dark)`, then the `<img>` fallback (wide, light). Do not use the
  `#gh-dark-mode-only` fragment trick.
* **Relative paths** (`docs/brand/...`) so forks and branches render their own assets. GitHub's
  rewriting of *relative* `srcset` paths inside `<source>` must be verified on github.com in both the
  repository home view and the file (blob) view before merging (see the checklist below).
* **Alt text states the result** (who converged in how many iterations, under which test), not
  "hero image".
* **Badges** come from shields.io with `style=flat-square`, `labelColor=52514e` and color `6b6963`,
  so they read as quiet metadata, not as a second color system.
* **Social preview:** upload `docs/brand/logo/social-preview.png` (1280 × 640) in Settings → General →
  Social preview; GitHub does not read it from the repository. The card says "150+ methods", because
  it is not rebuilt on every release.

## Before publishing the draft

1. Run `bash docs/brand/scripts/build.sh`. It rebuilds every asset and prints the "What's inside"
   and "Research" tables from the registry and from `research/`. Then run
   `.venv/bin/python docs/brand/scripts/readme_facts.py --write docs/brand/README.draft.md`: it
   replaces both tables and the counts in the one-liner (168) and the `numopt problems` sentence
   (103). `--check` exits with status 1 while any of them is stale.
2. Resolve every `TODO(...)` marker: portal URL, PyPI name (`numopt-lab`), tests and DOI badges,
   screenshots, citation authors.
3. Copy `docs/brand/CITATION.draft.cff` to the repository root as `CITATION.cff` with the
   maintainers' names, so GitHub shows "Cite this repository" (the *Cite* badge links to it).
4. Copy `README.draft.md` to the repository root as `README.md` (its paths already assume the root).
5. Push to a branch and check on github.com before merging: light and dark themes, a phone-width
   viewport (stacked hero), the repository home view and the blob view of `README.md` (relative
   `srcset` paths).
