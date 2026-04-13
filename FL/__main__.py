import sys

# Sniff --use_tf before any package import so backend flag is set
# before TF is conditionally imported.
_use_tf = "--use_tf" in sys.argv
if _use_tf:
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

from .backend import set_backend
set_backend(_use_tf)

from .main import main

if __name__ == "__main__":
    main()
