"""Repo root, resolved from this file's location.

A leaf module with no intra-package imports on purpose: both `config.paths` and
`io.provenance` need the repo root, and routing it through either of them would
create an import cycle.

Works under an editable install because `__file__` still resolves to the real
source tree, not site-packages.
"""

from pathlib import Path

# <repo>/src/ieeg_ehr/_repo.py  ->  parents[0]=ieeg_ehr, [1]=src, [2]=<repo>
REPO_DIR = Path(__file__).resolve().parents[2]
