---
title: "[Automatic issue] Missing `_.trans()`."
labels: "good first issue"
---

It looks like one of our test cron detected missing translations.
You can see the latest output [here](https://github.com/napari/napari/actions/workflows/test_translations.yml).
There are likely new strings to either ignore, or to internationalise.

You can also Update the cron script to update this issue with better information as well.

Note that this issue will be automatically updated if kept open, or a new one will be created when necessary, if no open
issue is found and new `_.trans` call are missing.
