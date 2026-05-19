# Git Collaboration Workflow

This project uses GitHub Flow for a three-person team.

## Core Rules

- `main` should always be usable.
- All work happens on feature or fix branches.
- All merges into `main` happen through pull requests.
- Every PR needs at least one teammate approval.
- Initial development uses `main` as the latest stable baseline; no release tags are required yet.

## Daily Workflow

Start from the latest `main`:

```bash
git switch main
git pull origin main
```

Create a task branch:

```bash
git switch -c feature/qc-module
git switch -c fix/readme-conflict
git switch -c docs/workflow-guide
```

Commit focused changes:

```bash
git add .
git commit -m "Add basic FASTA QC"
```

Push and open a PR:

```bash
git push origin feature/qc-module
```

After the PR is merged:

```bash
git switch main
git pull origin main
git branch -d feature/qc-module
```

## Branch Names

- `feature/...`: new functionality
- `fix/...`: bug fixes
- `docs/...`: documentation
- `refactor/...`: code restructuring
- `test/...`: tests

## Pull Requests

Keep each PR focused on one goal. Include:

- What changed
- Why it changed
- How it was checked
- Remaining limitations or TODOs

Changes that affect analytical reliability must include tests or a clear validation note.

## Conflict Handling

Refresh from `main` before opening a PR:

```bash
git fetch origin
git rebase origin/main
```

The branch owner resolves conflicts and reruns tests. Direct commits to `main` are not allowed.

Avoid force pushes. If a branch rewrite is necessary, use:

```bash
git push --force-with-lease
```

Tell the team before doing this.

## Suggested Early Ownership

- Developer A: `io.py`, `qc.py`, input validation
- Developer B: `alignment.py`, `mutations.py`, numbering mapper
- Developer C: `clade.py`, `resistance.py`, report/UI

Shared files such as `README.md`, `pyproject.toml`, and `environment.yml` need careful review.

## Recommended GitHub Settings

Set branch protection for `main`:

- Require a pull request before merging
- Require at least 1 approval
- Dismiss stale approvals when new commits are pushed
- Require conversation resolution before merging
- Block force pushes
- Block deletions

When CI is enabled, require the test workflow to pass before merging.
