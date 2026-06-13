import os

# Keep the suite serial and deterministic: process-pool workers are not
# measured by coverage, and machine-dependent jobs defaults would leak into
# report-options assertions. Explicit parallel tests override per-call.
os.environ.setdefault("FASCAT_JOBS", "1")
