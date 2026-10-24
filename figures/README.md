# Figures for localfirstai.eu

The charts in the blog's posts, built only from files published in this repository, so every mark traces to a record you can open.

- **`lfa_viz.py`:** the components and the one shared style.
  - **Components:** small-multiple bars, dots with intervals, multi-series dot rows, slope, strip, scatter panels, diverging bars, a log dot-range, columns, and registered ranges against a projection.
  - **Style and colour:** inline SVG that follows the site's light and dark themes. The colours are the three categorical slots, validated on the blog's real backgrounds.
  - **Marks:** bars at most 24 px thick with rounded data ends, hairline grids, and text in ink, never in a series colour.
- **`posts/`:** one module per post. Each names its post (`POST`) and builds its figures (`figures(repo)`) by reading the experiment's published files.
- **`build.py`:** writes each figure into its post, between `<!-- lfa-fig:<name> -->` markers placed by hand once, and the shared style into the blog's head partial.

```
python3 figures/build.py --blog ~/local-first-ai-blog            # rebuild every figure in place
python3 figures/build.py --blog ~/local-first-ai-blog --check    # report what would change
```

**Where a value is derived rather than read,** the module computes it from published predictions and labels and says so in the caption. For example, Wilson intervals, or a budget agreement recomputed from `leads44.budget_agree`.

exp_050's figures predate this folder and are built by its own kit (`tasks/chronos/exp_050_society_vs_one_model/kit/figures50.py`).
