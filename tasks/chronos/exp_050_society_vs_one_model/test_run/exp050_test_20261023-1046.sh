ollama stop gemma4:e4b || true
ollama ps
.venv/bin/python3 scripts/leads50.py run test G 1
.venv/bin/python3 scripts/leads50.py run test M 1
.venv/bin/python3 scripts/leads50.py run test G 2
.venv/bin/python3 scripts/leads50.py run test M 2
.venv/bin/python3 scripts/leads50.py run test G 3
.venv/bin/python3 scripts/leads50.py run test M 3
ollama stop gemma4:26b || true
.venv/bin/python3 scripts/leads50.py run test Gs 1
.venv/bin/python3 scripts/leads50.py run test Ms 1
.venv/bin/python3 scripts/leads50.py compare test > runs/exp050_test_compare.json
