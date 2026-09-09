# IEEE LaTeX (MCSoC paper)

## Class
`\documentclass[conference]{IEEEtran}` — two-column, 10pt, US letter. Do **not** add
`[compsoc]` or `[journal]`.

## Structure (order matters)
```
\title{...}
\author{\IEEEauthorblockN{Name}\IEEEauthorblockA{dept \\ inst \\ email} \and ...}
\maketitle
\begin{abstract} ... \end{abstract}          % one paragraph, ~150–230 words, NO citations
\begin{IEEEkeywords} ... \end{IEEEkeywords}
\section{Introduction} ...
\section*{Acknowledgment} ...
\bibliographystyle{IEEEtran}
\bibliography{refs}
```

## Known offline-build state (this machine, 2026-09-09)
- MiKTeX at `D:\Programs\MikTex`; `latexmk` + `pdflatex` on PATH.
- `IEEEtran.cls`, `IEEEtran.bst`, `cite`, `hyperref`, `booktabs`, `algorithmic`, `subfig`,
  `orcidlink` all present.
- `[MPM]AutoInstall = 1` set — MiKTeX fetches missing packages silently. **Do one build
  online after adding any new package** so it caches before going offline.
- Build: `latexmk -pdf main.tex`. Clean: `latexmk -c` (aux) / `latexmk -C` (aux + pdf).

## Errors seen / likely
| Symptom | Cause | Fix |
|---|---|---|
| `Something's wrong--perhaps a missing \item` at `\end{thebibliography}` | no `\cite` / `\nocite` anywhere, so the bib is empty | add real `\cite{}`s, or keep the temporary `\nocite{*}` in `main.tex` |
| bibliography not updating | stale `.bbl` | `latexmk -C` then rebuild; latexmk normally reruns bibtex automatically |
| first cold build fails, second succeeds | AutoInstall fetched packages mid-run | rerun; harmless once cached |
| `hyperref` option clash / broken links | load order | keep `hyperref` **last** in the preamble |

## Page limit
8 pages **including references**. IEEEtran conference is dense — budget figures early.
