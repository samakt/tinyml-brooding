# Data

CSV files are not committed (about 12 MB each compressed, 480 MB raw). Regenerate them:

    make data        # Python simulator, seed 11
    make data-java   # Java engine, seed 11

The same seed and parameters give a byte-identical file. Share a fixed copy through a GitHub Release or Google Drive, and note the seed in its name.
Real farm data (after consent) goes in `data/raw/` and is never committed.
