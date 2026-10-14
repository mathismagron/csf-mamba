# Manuscript (IEEE JSTARS) — DistilSCD

Build (from the repository root; LaTeX via `tectonic`):

```bash
python paper/scripts/build_tables.py      # logs/ -> paper/tables/*.csv (canonical numbers)
python paper/scripts/analyze_revision_a.py # logs/revision_a/ -> paper/tables/revision_a_*.csv (controls)
python paper/scripts/make_numbers.py      # tables -> paper/numbers.tex (one macro per number)
python paper/scripts/make_tables_tex.py   # tables -> paper/tables_tex/table*.tex
python paper/scripts/make_supp_tables.py  # tables -> paper/tables_tex/supp_*.tex
python paper/scripts/make_figures.py      # tables + logs -> paper/figures/*.{png,pdf}
python paper/scripts/plot_qualitative.py --in logs/revision_a/qualitative   # Fig. 7
python paper/scripts/check_numbers.py     # provenance check (must report 0 problems)
cd paper && tectonic main.tex && tectonic supplement.tex
```

- `brief.md`: pitch, contributions, figure arc, binding wording rules.
- `venue.md`: JSTARS requirements and name check.
- `notes/`: verified bibliography log, related-work notes, published-number checks, literal-number
  sources, internal review synthesis, e-mail drafts.
- `cover_letter.md`: draft cover letter.
- `scripts/qualitative.sbatch`: inference job for Fig. 7 (to run on Narval).

Note: `logs/` is git-ignored; the tables in `paper/tables/` are the versioned record of the results.
