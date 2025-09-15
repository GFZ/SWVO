# Contributing to SWVO
Thank you for your interest in contributing to [SWVO](https://github.com/GFZ/SWVO). This guide explains the development workflow, coding standards, and testing requirements.
## Getting Started
1.  Fork the repository and clone it locally:
    ```bash
    git clone https://github.com/<your-username>/SWVO.git
    cd SWVO
    ```
2.  Create a feature branch:
    ```bash
    git checkout -b my-feature
    ```
## Code Style and Linting
We use [Ruff](https://docs.astral.sh/ruff/) for linting, formatting, and import sorting.
1.  To check your code for issues:
    ```bash
    ruff check .
    ```
2. To automatically apply available fixes:
	```bash
	ruff check . --fix
	```
3. If Ruff cannot auto-fix an issue, fix it manually.
4. [Optional] Ruff is also integrated with `pre-commit`. To enable it:
    ```bash
    pip install pre-commit
    pre-commit install
    ```
    This ensures Ruff runs automatically on changed files before each commit.
    
## Running Tests
We use `pytest` for testing.
1.  Install dependencies:
    ```bash
    pip install -e .
    ```
    
2.  Run the test suite locally:
    ```bash
    python -m pytest tests/io
    ```
All new code should include tests when applicable.
## Pull Requests
1.  Push your branch:
    ```bash
    git push origin my-feature
    ```
    
2.  Open a Pull Request (PR) against `main`.
    
    -   Clearly describe the purpose of the changes.
    -   Reference related issues if applicable.
        
CI will automatically run Ruff and the test suite on your PR.
-   If Ruff fails and cannot auto-fix, you must resolve the issues before merging.
-   If tests fail, update your code or test cases accordingly.
    
## Commit Messages
-   Use clear, descriptive commit messages.
-   Example:
    ```
    Fix handling of missing satellite data in solar wind reader
    ```
## Review Process
-   At least one maintainer must review and approve your PR.
-   Be open to feedback and make the requested changes.
    
## Additional Notes
-   Keep PRs focused and small when possible.
-   Document new functions or modules.