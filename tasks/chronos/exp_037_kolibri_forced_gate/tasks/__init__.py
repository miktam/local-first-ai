"""exp_036 task loaders, prompt renderers and the item-manifest builder (BUILD_SPEC §5.5).

Every loader reads local files only (paths from tasks/assets.py and the EXP036_* environment).
Nothing in this package opens a network connection, and nothing here writes item text or gold of a
withheld set (GPQA EN/DE, AIME-DE, RGB) anywhere except $EXP036_PRIVATE.
"""
