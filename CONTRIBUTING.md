# Contributing Guidelines

Thanks for helping build the ARC JEPA × HRL stack. Please keep changes reproducible,
well‑tested, and aligned with the Beads workflow described in `Agents.md`.

## Environment & Dependencies
- Use `uv` for Python envs: `uv venv --python 3.11 .venv && source .venv/bin/activate`.
- Install runtime deps: `uv pip install --python .venv/bin/python -r requirements.txt`.
- Install dev/test extras when running tests: `uv pip install --python .venv/bin/python -r requirements-dev.txt`.
- RL work (Ray, Active Reasoner, SIGReg): `uv pip install --python .venv/bin/python -r requirements-rl.txt`. LeJEPA utilities are vendored under `lejepa/` (no external pip install required).

## Testing
- Run the full suite before pushing: `PYTHONPATH=. .venv/bin/pytest`.
- Targeted runs for quicker loops are fine, but avoid merging with red tests.
- Mock heavy dependencies in tests where practical (e.g., replace trainers/scorers or Ray envs with lightweight stubs) to keep the suite fast and stable.

## Style & Etiquette
- Follow existing code patterns; prefer type hints and small, focused functions.
- Keep planning docs in `history/` if you need them; don’t add TODO lists to code.
- Commit message convention: short, action-oriented, include Beads ID when applicable.

## Issue Tracking (Beads)
- All work is tracked in Beads (`bd ...`). Claim issues, update status, and close with code.
- Always commit `.beads/issues.jsonl` alongside code changes that touch issue state.

## Documentation
- Update relevant docs/configs when adding features or dependencies.
- Link new docs from `README.md` when they’re intended for contributors.
