# Paper — MCSoC 2026

- `main.tex` — the manuscript. Section skeleton with per-section notes as comments.
- `refs.bib` — bibliography (mirrors `../BIBLIOGRAPHY.md`; some entries need verification).
- `figures/` — figure sources.

## Build

```
latexmk -pdf main.tex
```

MiKTeX has `IEEEtran.cls`, `IEEEtran.bst`, and all packages `main.tex` uses. **Run one build
while still online** so MiKTeX can fetch anything missing before the offline month.

## Convention

IEEE Conference Proceedings format (goes to IEEE Xplore):

- `\documentclass[conference]{IEEEtran}` — two-column, 10pt, US letter.
- **8 pages including references** (supervisor's limit).
- Abstract: one paragraph, ~150–230 words, no citations, no custom macros.
- `\begin{IEEEkeywords}` after the abstract.
- Numeric citations in order of appearance, `\bibliographystyle{IEEEtran}`.
- Submission via EDAS: https://edas.info/N34632
- Official IEEE template (if a fresh copy is wanted):
  https://www.ieee.org/conferences/publishing/templates.html
