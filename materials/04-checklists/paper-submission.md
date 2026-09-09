# Paper submission checklist — MCSoC 2026

System: **EDAS**, https://edas.info/N34632. All dates in `../00-project/mcsoc-2026-cfp.md`.

## Ahead of Oct 10
- [ ] EDAS account; register the paper (title, abstract, authors, affiliations, keywords)
      early — you can upload the PDF later and revise it up to the deadline.
- [ ] `main.tex` compiles to **≤ 8 pages including references** with `latexmk -pdf`.
- [ ] Remove the temporary `\nocite{*}`; every reference is actually cited.
- [ ] Verify each `refs.bib` entry marked `note = {verify}` (author lists, venues, years).
- [ ] Figures legible in greyscale and at column width; captions self-contained.
- [ ] Abstract in the PDF matches the abstract entered in EDAS.
- [ ] Run EDAS's PDF check (IEEE Xplore-compatible PDF / embedded fonts). Fix font-embedding
      warnings (`pdflatex` from MiKTeX usually embeds; check with `pdffonts main.pdf`).
- [ ] Author list finalised — order, corresponding author, ORCIDs if used.
- [ ] Supervisor sign-off on the final version.

## Submit (by Oct 10, hard)
- [ ] Upload final PDF to EDAS. Confirm the submission shows as complete.

## On acceptance (notification Oct 31)
- [ ] Address reviewer comments.
- [ ] IEEE copyright form (eCF, usually linked from EDAS).
- [ ] Camera-ready PDF through IEEE PDF eXpress if required; upload by **Nov 8**.
- [ ] **Register at least one author** by Nov 8 (required for the paper to appear).
- [ ] Decide onsite vs online presentation (hybrid conference, Dec 14–17, Shanghai).
