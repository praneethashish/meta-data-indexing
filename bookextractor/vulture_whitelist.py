# Vulture whitelist for known false positives.
# These are intentionally unused but required for compatibility.

# TYPE_CHECKING guarded import (not used at runtime, only for type hints)
fitz

# Stub class constructor signatures must match real vLLM classes
kwargs
