# Security Notes

This public extraction was reviewed to avoid shipping:

- API keys or token files
- browser cookies, profiles, or session exports
- Adam-specific machine paths
- raw run outputs from local benchmark sessions

## Expected placeholders

The repo intentionally mentions these generic names in code and docs:

- `HF_TOKEN`
- `https://router.huggingface.co/v1`
- model slugs such as `zai-org/GLM-5.2:fireworks-ai`

## Review checklist

Before publishing updates:

1. Delete `runs/`, `.venv/`, and `__pycache__/`.
2. Search tracked files for `/Users/`, token-looking strings, and copied private output.
3. Re-run the fake-mode smoke test.
4. If you changed HF paths, verify the repo still works with only `HF_TOKEN` or the standard Hugging Face token cache.
