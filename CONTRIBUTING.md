# Contributing

## Development Priorities

Please keep early pull requests focused on reliability rather than UI polish.

1. Input validation and QC
2. Reference coordinate and numbering mapper
3. Alignment wrapper and mutation parser
4. Nextclade wrapper
5. Marker/site table curation with tests

## Branches and Pull Requests

- Use small branches with focused changes.
- Include tests or a short reason tests are not practical yet.
- Do not commit private sequence data or large generated results.
- Preserve legacy scripts unless a migration issue explicitly removes or replaces them.

## Code Style

Run before opening a pull request:

```bash
ruff check .
pytest
```
