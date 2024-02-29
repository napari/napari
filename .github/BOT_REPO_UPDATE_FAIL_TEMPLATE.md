---
title: "{{ env.TITLE }}"
labels: [bug]
---

Update of {{ env.BOT_REPO }} failed on {{ date | date("YYYY-MM-DD HH:mm") }} UTC


Full run: https://github.com/napari/napari/actions/runs/{{ env.RUN_ID }}

(This post will be updated if another test fails, as long as this issue remains open.)
