"""gen8 / gen8.5 pipeline package.

gen8  = Opus mono labeler + full new architecture (Tier 0-3).
gen8.5 = 7-way Sonnet labeler + same architecture.

Both share this directory; `pipeline.run(..., labeler="opus"|"sonnet7way")`
selects between them at runtime.
"""
