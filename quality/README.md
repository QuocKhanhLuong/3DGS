# Point-guided frontend quality gate

`quality/checklists.json` contains one executable gate:
`POINT_GUIDED_FRONTEND`. It covers only the locked `T1/T2/FLAIR` frontend:
coarse semantics, deterministic initial points, bounded refinement,
point-centre semantics, a sparse semantic-aware PoU, and static diagnostic B
planes. It does not authorize or produce a T1ce volume.

This gate records evidence for the current executable boundary only: the
implemented Phase 1-5 frontend through static B planes. It neither implements
nor certifies the separately authorized, not-yet-implemented Phase 6 SWT-Haar
anchor `A` or Phase 7 geometry-aware point evidence `f_spec`; their policy
authority is defined by `AGENTS.md` and `CODEGRAPH.json`. Gate C remains
blocked and default-deny, including dynamic trajectory, decoding, losses, and
T1ce synthesis.

Run it with the project environment:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_phase.py --list
PYTHONPATH=src .venv/bin/python scripts/check_phase.py POINT_GUIDED_FRONTEND
PYTHONPATH=src .venv/bin/python scripts/check_phase.py POINT_GUIDED_FRONTEND --run
```

`--run` executes the point-guided CPU suite (including its frontend smoke),
compilation, and `git diff --check`. It refuses to run on a dirty tree unless
you explicitly add `--allow-dirty`; that produces development-only evidence.
Use `--report-dir <path>` to write a JSON and Markdown evidence report.

A passing automated result is software evidence only. The Human Gate remains
`pending` and this runner has no approval command or decision-writing path.
