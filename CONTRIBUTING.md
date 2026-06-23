# Contributing

Influmatics uses GitHub Flow for three-person collaboration. Keep `main` stable, do
all work on focused branches, and merge through reviewed pull requests.

## Development Priorities

Please keep early pull requests focused on reliability rather than UI polish.

1. Input validation and QC
2. Reference coordinate and numbering mapper
3. Alignment wrapper and mutation parser
4. Nextclade wrapper
5. Marker/site table curation with tests

## Branches and Pull Requests

- Start from the latest `main` before every task:

```bash
git switch main
git pull origin main
```

- Create one focused branch per task:

```bash
git switch -c feature/qc-module
git switch -c fix/readme-conflict
git switch -c docs/workflow-guide
```

- Use these branch prefixes:
  - `feature/...` for new functionality
  - `fix/...` for bug fixes
  - `docs/...` for documentation
  - `refactor/...` for code restructuring
  - `test/...` for tests
- Open a pull request for every merge into `main`.
- Require at least one teammate approval before merging.
- Include tests or a short reason tests are not practical yet.
- Do not commit private sequence data or large generated results.
- Preserve legacy scripts unless a migration issue explicitly removes or replaces them.

## Pull Request Checklist

Every PR should explain:

- What changed
- Why it changed
- How it was checked
- Remaining limitations or TODOs

Avoid PRs that mix unrelated goals. Shared project files such as `README.md`,
`pyproject.toml`, and `environment.yml` should receive extra careful review.

## Conflict Handling

- Pull or rebase from `main` before opening a PR:

```bash
git fetch origin
git rebase origin/main
```

- The branch owner resolves conflicts and reruns tests.
- Do not commit directly to `main`.
- Do not use `git push --force`. If rewriting a branch is unavoidable, use
  `git push --force-with-lease` only after telling the team.

## Data Policy

Do not commit:

- `.DS_Store`
- `.Rhistory`
- Raw FASTQ/FASTA inputs
- GISAID-derived sequence data or metadata
- Large generated analysis outputs

## Code Style

Run before opening a pull request:

```bash
python -m pytest
python -m compileall influmatics
```

For external tool wrappers, document manual checks in the PR description. This
includes MAFFT, Nextclade, BLAST, Medaka, and HyPhy.
