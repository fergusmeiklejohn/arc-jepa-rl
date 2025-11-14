# Dependency Management

We manage Python dependencies with `uv` and keep machine-readable manifests in the
repo so teammates can reproduce the same environment.

## How to install

```bash
# 1. Create / activate the env (one time)
uv venv --python 3.11 .venv
source .venv/bin/activate

# 2. Install core runtime packages
uv pip install --python .venv/bin/python -r requirements.txt

# 3. (Optional) Install dev/test extras
uv pip install --python .venv/bin/python -r requirements-dev.txt

# 4. (Optional) Install RL stack extras
uv pip install --python .venv/bin/python -r requirements-rl.txt
```

> `uv pip install` keeps the environment reproducible without needing a global
> `pip` configuration. Always edit these requirement files when adding or
> removing dependencies.

## Package overview

| Package           | Notes / Purpose                                   | Manifest file          |
| ----------------- | ------------------------------------------------- | ---------------------- |
| `torch>=2.1`      | JEPA encoder, projection heads, RL models         | `requirements.txt`     |
| `numpy>=1.24`     | Grid utilities, numerical helpers                 | `requirements.txt`     |
| `pyyaml>=6.0`     | Config parsing for scripts and generators         | `requirements.txt`     |
| `gymnasium>=0.29` | RLlib/Gym env wrappers for latent option env      | `requirements.txt`     |
| `pytest>=8.0`     | Test suite                                       | `requirements-dev.txt` |
| `ray[rllib]>=2.9` | PPO/A2C training stack (optional feature)         | `requirements-rl.txt`  |
| `beads-mcp`       | Optional MCP server for issue tracking workflows  | (documented in Agents.md) |

If you need `gym` instead of `gymnasium`, install it alongside the base
requirements (the code can fall back to either implementation).

## Adding or updating dependencies

1. Decide which environment needs the package (runtime, dev/test, RL).
2. Add it to the appropriate `requirements-*.txt`.
3. Re-run the relevant `uv pip install --python .venv/bin/python -r ...`
   command locally.
4. Update this document (and other docs if necessary) describing why the new
   dependency is required.
5. Commit the updated files together with the code change that requires them.
