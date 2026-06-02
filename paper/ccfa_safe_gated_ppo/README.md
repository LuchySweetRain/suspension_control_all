# CCF-A Style Paper Draft

This directory contains a CCF-A conference-style manuscript draft for the current active-suspension project.

Target framing:

- Primary venue style: AAAI/NeurIPS-like two-column AI conference manuscript.
- Main claim: offline-imitation-initialized, projection-aware, safe-teacher-gated PPO for actuator-feasible online active suspension adaptation.
- Evidence source: `results/improvement_gate_fair_evidence_table` and `results/si_rppo_e20_improvement_gate_fair_baselines`.

Files:

- `main.tex`: full paper draft.
- `references.bib`: verified core references used by the current draft.

Build:

```bash
cd paper/ccfa_safe_gated_ppo
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The draft uses an article fallback so it can compile without a downloaded conference style. For final AAAI submission, place the official `aaai2026.sty` and `aaai2026.bst` from the AAAI author kit in this directory and replace the fallback preamble with the official template preamble.

Important limitation:

This is a paper-quality first draft from current experiments, not a final submission. The current evidence supports the PPO-vs-PPO ablation strongly and positions the method competitively against SAC on actuator feasibility, but it does not yet prove broad superiority over all off-policy baselines.
