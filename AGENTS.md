# Repository Instructions

This repository is a learning portfolio project for building a small GPT-style
Japanese text generator from literary texts. Keep the work understandable for a
beginner-to-intermediate reader who wants to explain each part in an interview.

## Workflow

- Work on exactly one ticket from `tickets.md` at a time.
- Do not start the next ticket until the current ticket is verified and committed.
- After finishing a ticket, stop and report the commit hash, changed files, and verification commands.
- Keep changes small and tied to the active ticket. Avoid unrelated cleanup.
- Use local commits as the completion point. This repository currently has no remote.

## Development Defaults

- Use `uv` for dependency management and command execution.
- Use Python 3.12, as declared in `.python-version` and `pyproject.toml`.
- Prefer plain PyTorch for model and training code once implementation begins.
- Prefer Apple Silicon `mps` when available, with a CPU fallback.
- Keep code explicit and teachable: visible config, readable variable names, and comments for non-obvious ML tensor flow.
- Do not use experiment trackers, web services, or heavy frameworks unless a later ticket explicitly adds them.

## Data And Artifacts

- Do not commit full raw Aozora Bunko text.
- Commit manifests, scripts, docs, and small metadata needed to reproduce data preparation.
- Keep generated data, checkpoints, model outputs, plots, and local caches out of Git unless a ticket explicitly says otherwise.
- Document dataset limitations, smoke-test scope, and generation failures honestly.

## Verification

Before committing a ticket:

- Run syntax or smoke checks appropriate to the changed files.
- Run `git diff --check`.
- Inspect `git status --short` and confirm only intended files are included.
- If dependencies change, keep `pyproject.toml` and `uv.lock` consistent.

## Documentation Expectations

- Update `README.md` when commands, project scope, outputs, or expected behavior change.
- Update `tickets.md` when a ticket starts or finishes.
- Put experiment summaries, generation examples, and failure analysis in `experiments/results.md` once that ticket exists.
