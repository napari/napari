---
title: "{{ env.TITLE }}"
labels: [bug]
---
The {{ workflow }} workflow failed on {{ date | date("YYYY-MM-DD HH:mm") }} UTC

The most recent failing test was on {{ env.PLATFORM }} py{{ env.PYTHON }} {{ env.BACKEND }}
with commit: {{ sha }}

Full run: https://github.com/napari/napari/actions/runs/{{ env.RUN_ID }}

(This post will be updated if another test fails, as long as this issue remains open.)
