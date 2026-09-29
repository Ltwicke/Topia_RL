"""
eval/run_report — the plotting handbook of one complete training run.

A run is any number of segments (one per invocation of RL/train.py), listed
explicitly by the user and stitched along the checkpoint lineage:

    loader.py    segments, .log banner, lineage cut, robust CSV reading
    specs.py     what is plotted: scenario figures, training grid, cleaning
    plot.py      how it is plotted: template style, Panel, renderers
    gifs.py      flip-books of the per-update scenario renders
    __main__.py  CLI:  python -m eval.run_report --name <run> <segment> ...
"""
