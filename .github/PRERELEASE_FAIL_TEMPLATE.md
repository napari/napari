---
title: pip install --pre is failing
labels: [bug]
---
The pre-release cron job defined in `test_prereleases.yml` failed on {{ date | date("YYYY-MM-DD HH:mm") }} UTC

The last failing test was on {{ env.PLATFORM }} py{{ env.PYTHON }} {{ env.BACKEND }}
with commit: {{ sha }}

Full run: https://github.com/{{ payload.repository.full_name }}/actions/runs/{{ env.RUN_ID }}
