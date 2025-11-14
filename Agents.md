# What we are building
Read the docs: Project_Blueprint.md
(Note keep this updated as the project plan evolves)

## Issue Tracking with bd (beads)

IMPORTANT: This project uses bd (beads) for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

Check for ready work:
```
bd ready --json
```

Create new issues:
```
bd create "Issue title" -t bug|feature|task -p 0-4 --json
bd create "Issue title" -p 1 --deps discovered-from:bd-123 --json
```

Claim and update:
```
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

Complete work:
```
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. Check ready work: `bd ready` shows unblocked issues
2. Claim your task: `bd update <id> --status in_progress`
3. Work on it: Implement, test, document
4. Discover new work? Create linked issue:
   - `bd create "Found bug" -p 1 --deps discovered-from:<parent-id>`
5. Complete: `bd close <id> --reason "Done"`
6. Commit together: Always commit the `.beads/issues.jsonl` file together with the code changes so issue state stays in sync with code state

### Auto-Sync

bd automatically syncs with git:
- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed!

### Python Environment (uv)

- Create the project virtualenv with `uv venv --python 3.11 .venv`.
- Activate via `source .venv/bin/activate` (or the platform equivalent) before running commands.
- Install dependencies with `uv pip install --python .venv/bin/python -r requirements.txt`.
- Add `requirements-dev.txt` for tests and `requirements-rl.txt` when working on RLlib training.
- See `docs/DEPENDENCIES.md` for the authoritative dependency list and update process.

### Python Environment (uv)

- Create the project virtualenv with `uv venv --python 3.11 .venv`.
- Activate via `source .venv/bin/activate` (or the platform equivalent) before running commands.
- Install dependencies with `uv pip install --python .venv/bin/python -r requirements.txt`.
- Add `requirements-dev.txt` for tests and `requirements-rl.txt` when working on RLlib training.
- See `docs/DEPENDENCIES.md` for the authoritative dependency list and update process.

### MCP Server (Recommended)

If using Claude or MCP-compatible clients, install the beads MCP server:

```
pip install beads-mcp
```

Add to MCP config (e.g., `~/.config/claude/config.json`):
```
{
  "beads": {
    "command": "beads-mcp",
    "args": []
  }
}
```

Then use `mcp__beads__*` functions instead of CLI commands.

### Managing AI-Generated Planning Documents

AI assistants often create planning and design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, and similar files

Best Practice: Use a dedicated directory for these ephemeral files

Recommended approach:
- Create a `history/` directory in the project root
- Store ALL AI-generated planning/design docs in `history/`
- Keep the repository root clean and focused on permanent project files
- Only access `history/` when explicitly asked to review past planning

Example .gitignore entry (optional):
```
# AI planning documents (ephemeral)
history/
```

Benefits:
- Clean repository root
- Clear separation between ephemeral and permanent documentation
- Easy to exclude from version control if desired
- Preserves planning history for archeological research
- Reduces noise when browsing the project

---

## Run Environments & Processes

### Local Development (MacBook Pro M3 Max, 36GB RAM)
- Use `uv` to manage Python: `uv venv --python 3.11 .venv` then `source .venv/bin/activate`.
- Install deps with `uv pip install --python .venv/bin/python -r requirements.txt` (+ `requirements-dev.txt` when running tests, `requirements-rl.txt` for RLlib work).
- What to run locally (fast iteration):
  - Unit/integration tests: `.venv/bin/pytest` (target is green).
  - Small synthetic datasets and sanity checks: `python scripts/generate_dataset.py ...`.
  - JEPA dry-run or tiny epochs: `python scripts/train_jepa.py --dry-run --config <jepa.yaml>`.
  - Latent env rollouts: `python scripts/rollout_latent_env.py --env-config <env.yaml> --jepa-config <jepa.yaml>`.
  - Option discovery/promotion and few-shot solver on small sets.
  - Meta-JEPA short runs on tiny JSONL to validate contrastive training.

Notes: CPU PyTorch is fine for smoke tests; keep epochs small and batch sizes minimal.

### GPU Training (Paperspace A6000, 8 CPUs, 48GB RAM)
- Intended for larger JEPA/guidance/meta-JEPA training runs and heavy evaluations.
- Process:
  1) Push code to GitHub from local.
  2) On Paperspace, pull latest (`git pull`), create env (`uv venv --python 3.11 .venv`), and install deps with `uv pip install --python .venv/bin/python -r requirements.txt` (plus extras as needed).
  3) Run training scripts with GPU-enabled PyTorch; use curated configs.
  4) Persist artifacts (weights, JSON reports) back to repo or external storage as appropriate.

### Scripts Overview
- `scripts/train_jepa.py` — object-centric JEPA helper (supports `--dry-run`).
- `scripts/rollout_latent_env.py` — simulate latent option env and produce traces.
- `scripts/train_guidance.py` — train DSL guidance scorer (neural heuristic).
- `scripts/train_meta_jepa.py` — train rule-family encoder on JSONL.
- `scripts/evaluate_arc.py` — run DSL-only and meta-guided ablations and emit JSON.
- `training/meta_jepa/prior.py` exposes `MetaJEPAPrior` for wiring trained rule-family
  embeddings into `GuidedBeamSearch` or `FewShotSolver.solve(..., meta_prior=...)`
  so Meta-JEPA biases DSL program ordering.

---

## Beads Tickets for In-Action Tests & Remaining Work
All hands-on tests and remaining build items are tracked in Beads and linked to the epic `arc-jepa-rl-fb5b`. Use `bd ready --json` to view what’s unblocked.

### Important Rules

- Use bd for ALL task tracking
- Always use `--json` flag for programmatic use
- Link discovered work with `discovered-from` dependencies
- Check `bd ready` before asking "what should I work on?"
- Store AI planning docs in `history/` directory
- Do NOT create markdown TODO lists
- Do NOT use external issue trackers
- Do NOT duplicate tracking systems
- Do NOT clutter repo root with planning documents

For more details, see README.md and QUICKSTART.md.
