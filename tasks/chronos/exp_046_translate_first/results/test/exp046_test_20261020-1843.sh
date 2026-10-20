.venv/bin/python3 scripts/translate46.py run test --arm native
.venv/bin/python3 scripts/translate46.py run test --arm translated
.venv/bin/python3 scripts/translate46.py score test > runs/exp046_test_score.json
.venv/bin/python3 scripts/leads44.py regress scripts/kev_knife.py
