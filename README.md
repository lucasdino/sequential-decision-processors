# Sequential Decision Processors
---
## Environment Setup

Ensure you have Java 1.8+ installed on your machine.

We recommend building with [uv](https://github.com/astral-sh/uv) for fast, reproducible installs.

1. Create a fresh virtual environment:
   ```bash
   uv venv .venv -p 3.11
   source .venv/bin/activate     # Linux / macOS
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

2. Install dependencies from `requirements.txt`:
    ```bash
    uv pip install -r requirements.txt
    ```

3. Register a jupyter kernel named `seq-dec-proc`:
    ```bash
    uv run python -m ipykernel install --user --name seq-dec-proc --display-name "Python (seq-dec-proc)"
    ```
    