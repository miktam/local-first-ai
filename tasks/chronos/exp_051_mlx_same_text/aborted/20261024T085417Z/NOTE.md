# Aborted launch, 2026-10-24 08:54Z

`run51.sh` set `B="python3.14 -I kit/bench51.py"` and called `$B --arm O`. zsh does not split an unquoted parameter into words, so every arm failed with "no such file or directory" and no model call was made. `score51.py` then failed on the missing summaries, and the restore lines reloaded `gemma4:26b` and `gemma4:e4b`. Nothing was measured. Fixed in Amendment 1.
