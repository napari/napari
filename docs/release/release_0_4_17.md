# napari 0.4.17

We're happy to announce the release of napari 0.4.17!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights


## New Features


## Improvements


## Bug Fixes


## API Changes


## Deprecations


## Build Tools


## Other Pull Requests

- support PyQt6, remove logic for `compile_qt_svgs` (#3707)
- Multi-color text with color encodings (#4464)
- Improvements on scale bar (#4511)
- Add "Processors": (generalize magicgui return type behavior outside of magicgui) (#4543)
- Do not add duplicate layers (#4544)
- allow use of validate_all=True on nested evented models (#4551)
- add ci for asv benchmark suite (#4554)
- Disconnect a removed slider's _pull_label callback from the axis label change event  (#4557)
- Fixes/modifications to nested list model and Group in prep for layergroups (#4560)
- Add option to use npe2 shim/adaptor for npe1 plugins (#4564)
- Update base docker image to ubuntu 22.04, optimize container size. (#4568)
- Fix Auto-open trans issue. (#4586)
- Forward port further 0.4.16 release note changes (#4592)
- Split coverage on parts (#4595)
- More updates to the ignored translations file and translation CI infrastructure (#4596)
- feat: add codespace (#4599)
- Discuss the approval of NAP-2 (conda-based packaging for napari) (#4602)
- Removed unused private Layer._position (#4604)
- Clean status message after leave canvas (#4607)
- Block lxml 4.9.0 version (#4616)
- Alternate fix for alt-text issues (#4617)
- Remove meshzoo from napari (#4620)
- Fix numpy version comparison for dev versions of numpy (#4622)
- Restore "show all key bindings" shortcut  (#4628)
- Raise error when rgb True but data not correct dims (#4630)
- Enable patch coverage reporting (#4632)
- Add tests for qt_progress_bar.py (#4634)
- Add versioned API docs workflow (#4635)
- Define different shortcuts for different screenshot actions in file menu (#4636)
- Add modal dialogs for window/application closure (#4637)
- Remove some sleepy tests, wait with timeout instead (#4638)
- Update installation instruction in documentation (#4639)
- Speed up notebook display tests  (#4641)
- build: slight updates/de-dups in setupcfg (#4645)
- test: Speed up test_viewer tests, split out pyqt5/pyside2 tests again (#4646)
- Move test examples on end of test run them in different CI job  (#4647)
- Optimization: convert point slice indices to floats (#4648)
- Allow to trigger test_translation manually (#4652)
- Limit comprehensive test concurrency to 1. (#4655)
- Do not run `asv` on push (#4656)
- Fix docs for shape/points layer properties (#4659)
- ci: fix tox factor names in tests (#4660)
- Do not use keyword argument for `QListWidgetItem` parent set in constructor (#4661)
- test: speed up magicgui forward ref tests (#4662)
- add layer reader in tooltip and info dialog (#4664)
- Embeding viewers in plugins widget example (#4665)
- Remove unused points _color state (#4666)
- fix: fix manifest reader extensions, ensure that all builtins go through npe2 (#4668)
- test: Cleanup usage of npe2 plugin manager in tests (#4669)
- add dependabot for github actions dependencies (#4671)
- Use `cache: pip` from actions setup-python (#4672)
- Remove `restore_settings_on_exit` test util (#4673)
- ci(dependabot): bump peter-evans/create-pull-request from 3 to 4 (#4675)
- ci(dependabot): bump docker/build-push-action from 2.5.0 to 3 (#4676)
- ci(dependabot): bump actions/setup-python from 3 to 4 (#4677)
- ci(dependabot): bump toshimaru/auto-author-assign from 1.3.4 to 1.5.0 (#4678)
- ci(dependabot): bump docker/metadata-action from 3 to 4 (#4679)
- Fix source of copied layer events by use `layer.as_layer_data_tuple` (#4681)
- Add NAP-3: Spaces (#4684)
- Fix NAP-1 status/type (#4685)
- bump npe2 version, use new features (#4686)
- [pre-commit.ci] pre-commit autoupdate (#4687)
- Tracks tail update fix beyond _max_length (#4688)
- Keep order of layers when select to save multiple selected layers (#4689)
- Remove unnecessary uses of make_napari_viewer (replace with ViewerModel) (#4690)
- Add new core devs to docs (#4691)
- Apply shown mask to points._view_size_scale (#4699)
- honor confirm close settings when quitting (#4700)
- Design issue template assignees (#4701)
- Refactor label painting by moving code from mouse bindings onto layer, adding staged history and paint event (#4702)
- add 'napari-xpra' target to dockerfile (#4703)
- Use custom color field classes in all models (#4704)
- Split out builtins into another top-level module (#4706)
- use python 3.9 for mac tests until numcodecs 3.10 wheels are available (#4707)
- napari.viewer.current_viewer fallback ancestor (#4715)
- Color edit without tooltip fix (#4717)
- Remove print statement in settings (#4718)
- Use some features from npe2 v0.5.0 (#4719)
- fix presentation and enablement of npe1 plugins (#4721)
- use same coverage report like in pull requests (#4722)
- fix clim init for dask arrays (#4724)
- split Image `interpolation` on two properties, `interpoltion2d` and `interpolation3d`  (#4725)
- Add public API for dims transpose (#4727)
- Update translation documentation link in pull request template (#4728)
- Add a public API for the setGeometry method of _qt_window (#4729)
- Fix bug in deepcopy of utils.context.Expr (#4730)
- Cleanup list of non-translatable strings. (#4732)
- Add mini interactive prompt to edit string_list.json (#4733)
- Expose points layer antialiasing control publicly (#4735)
- Use `npe2 list` to show plugin info (#4739)
- Fix `_get_state` for empty Vectors and Points layers (#4748)
- Fix Image parameter docs (#4750)
- Do not inform about coverage untill all CI jobs are done (#4751)
- Allow pandas.Series as properties values (#4755)
- Update deprecation warnings with versions (#4757)
- Stop slider animation on display and dimension changes (#4761)
- Refactor `NapariQtNotification` test for better cleaning Qt objects (#4763)
- Make references to examples link there (#4767)
- Set focus on `QtViewer` object after creating main window (#4768)
- ci(dependabot): bump styfle/cancel-workflow-action from 0.9.1 to 0.10.0 (#4770)
- ci(dependabot): bump actions/download-artifact from 2 to 3 (#4771)
- ci(dependabot): bump actions/github-script from 5 to 6 (#4772)
- ci(dependabot): bump bruceadams/get-release from 1.2.2 to 1.2.3 (#4773)
- ci(dependabot): bump docker/login-action from 1.9.0 to 2 (#4774)
- Add missed call of `running_as_bundled_app` in `__main__` (#4777)
- Bump vispy 0.11.0 (#4778)
- [pre-commit.ci] pre-commit autoupdate (#4779)
- Allow stack to be specified multiple times (#4783)
- Refactor: use app-model and in-n-out (#4784)
- Fix deepcopy for trans objects when passing args (#4786)
- Fix documentation link errors. (#4787)
- test: fix ability to use posargs with tox (#4788)
- Fixing readme typo (#4791)
- Allow dunder methods use in `PublicOnlyProxy` (#4792)
- add update_mesh method (#4793)
- Optimization: convert vector slice indices to floats (#4794)
- Add Talley to the steering council (#4798)
- Update documenation : install plugin from URL  (#4799)
- mock npe1 plugin manager during tests (fix tests with npe1 plugins installed) (#4806)
- [Automatic] Update albertosottile/darkdetect vendored module (#4808)
- Improve "select all" logic in points layer, adding "select all across slices" behavior (#4809)
- Accept NAP-2: Distributing napari with conda-based packaging (#4810)
- Make docked widgets of `qt_viewer` lazy (#4811)
- Restore Labels, Points and Shapes layer mode after hold key (#4814)
- Improve using Qt flags by direct use enums. (#4817)
- Confirmation on close, alternative approach (#4820)
- Fixes duplicate label warnings in the documentation (#4823)
- Make newly added points editable (#4829)
- add path specificity scoring (#4830)
- Allow empty tuple layer to be returned #4571 (#4831)
- added description of how to save layers without compression (#4832)
- Disable notifications if no main window (#4839)
- Move searching parent outside NapariQtNotification constructor (#4841)
- Fix multiscale level selection when zoomed out (#4842)
- [Automatic] Update albertosottile/darkdetect vendored module (#4845)
- Move vector mesh generation to napari vispy visual and fix vectors refesh (#4846)
- Enable plugin menu contributions with app-model (#4847)
- Raise better errors from lazy __getattr__ (#4848)
- Add the ability to return a List[Layer] in magicgui (#4851)
- Add plugin reader warning dialog (#4852)
- Fix creation `QtViewer` for `ViewerModel` with already added layers (#4854)
- Update hub.py (#4863)
- Fix viewer.reset() throws TypeError (#4868)
- Fix NAP-3 table of contents (#4872)
- MAINT: disable failing conda bundling. (#4878)
- Add option for forcing plugin choice when opening files in the GUI (#4882)
- MAINT: Work around conda bundling issue. (#4883)
- fix icon urls for close dialog (#4887)
- remove a few more qtbot.wait calls in tests (#4888)
- Fix missing slice from dims (#4889)
- ci(dependabot): bump toshimaru/auto-author-assign from 1.5.0 to 1.6.1 (#4890)
- NAP 4: asynchronous slicing (#4892)
- use npe2api in plugin install dialog (#4893)
- Make `viewer.events.layers_change` event deprecated (#4895)
- Add perfmon directory for shared configs and tools (#4898)
- Docker: Update xpra's apt too (#4901)
- Add saving folders to preferences (#4902)
- Conda: fix MacOS signing (#4904)
- Raise exception when layer created with a scale value of 0 (#4906)
- Correct minor typo in quick_start.md (#4908)
- Add Google Calendar ID to directive (#4914)
- Fix docs dependencies (#4915)
- Minor type fix (add Optional) (#4916)
- Maintenance: Add title to example and remove unused redirect configuration (#4921)
- Use `napari/packaging` for Conda CI (#4923)
- Correct minor typos on 'Contributing resources' page (#4927)
- Add napari.imshow to return viewer and layers (#4928)
- DOC Add docs on napari models and events (#4929)
- DOC Fix add image examples (#4932)
- add SciPy 2022 materials (#4934)
- Fix for display of label colors (#4935)
- Remove repeated items in gallery ToC. (#4936)
- DOC Remove references to old `add_volume` (#4939)
- Fix calendar ID (#4944)
- add user keymap (#4946)
- Initialize plugins before validating widget in `napari -w {plugin}` (#4948)
- MAINT: temporary mypy disabling as it's failing everywhere. (#4950)
- Adjust the slider value to include current_size (#4951)
- Option to load all dock widgets (#4954)
- DOC Add docs on how to run napari headless (#4955)
- Removes deprecated configuration options for notebooks from docs build (#4958)
- Update async slicing discussion section with PR feedback (#4959)
- MAINT: re-enable type checking workflow (#4960)
- DOC Add section on downloading CI built docs (#4961)
- DOC Expand explanation of interaction_box_image example (#4962)
- DOC: Fix some incorrect parameters names. (#4965)
- MAINT: bump napari-console to 0.0.6. (#4967)
- [pre-commit.ci] pre-commit autoupdate (#4968)
- DOC Add headless doc to toc (#4970)
- Fix: bump app-model, add test for layer buttons (#4972)
- Remove preliminary admonition for last two release notes (#4974)
- Enabe tags in gallery examples. (#4975)
- Add `FUNDING.yml` file to enable sponsor button and increase discoverability of option to sponsor napari (#4978)
- Fix resize window problem by to long text in help  (#4981)
- Fix throttled status updates by ensuring GUI updates occur on the main thread (#4983)
- Fix throttled status updates by use `QSignalThrottler` in `_QtMainWindow` constructor (#4985)
- Update viewer screenshot in docs and make more prominent on napari.org (#4988)
- Fix `python` name in macOS menu bar (#4989)
- Move the docs downloading screenshots to LFS (#4990)
- DOC Enable intersphinx linking in examples (#4992)
- Start adding a CODEOWNERS file. (#4993)
- Explicit convert color to string when compile resources (#4997)
- Add emacs temporary & local files to .gitignore (#4998)
- Set fixpoint before move for shapes (#4999)
- Fix broken links in Napari Code of Conduct (#5005)
- Fix sys.path issue with subprocess relaunch in macOS (#5007)
- MAINT: update issue failing template (#5012)
- Add link to docs artifact to pull requests (#5014)
- [pre-commit.ci] pre-commit autoupdate (#5015)
- Feature: register a keyboard shortcut for `preserve_labels` checkbox (#5017)
- Add option to edit second shortcut in settings (#5018)
- Fix shift selection for points (#5022)
- Revert "Fix sys.path issue with subprocess relaunch in macOS" (#5027)
- Use Source object to track if a layer is duplicated (#5028)
- Revert "Revert "Fix sys.path issue with subprocess relaunch in macOS"" (#5029)
- Add release notes for 0.4.17 (#5031)
- MAINT: fix a couple of trans._. (#5034)
- Ensure non-negative edge width (#5035)
- add #4954 to release notes (#5038)
- add new PRs merged for 0.4.17 to release notes (#5045)
- Small shortcuts preference pane wording update after #5018 (#5049)

## 40 authors added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
- [Cajon Gonzales](https://github.com/napari/napari/commits?author=cajongonzales) - @cajongonzales
- [chili-chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [cnstt](https://github.com/napari/napari/commits?author=cnstt) - @cnstt
- [Curtis Rueden](https://github.com/napari/napari/commits?author=ctrueden) - @ctrueden
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [dependabot[bot]](https://github.com/napari/napari/commits?author=dependabot[bot]) - @dependabot[bot]
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [Gabriel Selzer](https://github.com/napari/napari/commits?author=gselzer) - @gselzer
- [github-actions[bot]](https://github.com/napari/napari/commits?author=github-actions[bot]) - @github-actions[bot]
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Guillaume Witz](https://github.com/napari/napari/commits?author=guiwitz) - @guiwitz
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [James Ryan](https://github.com/napari/napari/commits?author=jamesyan-git) - @jamesyan-git
- [Jeremy Asuncion](https://github.com/napari/napari/commits?author=codemonkey800) - @codemonkey800
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kandarp Khandwala](https://github.com/napari/napari/commits?author=kandarpksk) - @kandarpksk
- [Kim Pevey](https://github.com/napari/napari/commits?author=kcpevey) - @kcpevey
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Kushaan Gupta](https://github.com/napari/napari/commits?author=kushaangupta) - @kushaangupta
- [Kyle I S Harrington](https://github.com/napari/napari/commits?author=kephale) - @kephale
- [Lia Prins](https://github.com/napari/napari/commits?author=liaprins-czi) - @liaprins-czi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Markus Stabrin](https://github.com/napari/napari/commits?author=mstabrin) - @mstabrin
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Pierre Thibault](https://github.com/napari/napari/commits?author=pierrethibault) - @pierrethibault
- [pre-commit-ci[bot]](https://github.com/napari/napari/commits?author=pre-commit-ci[bot]) - @pre-commit-ci[bot]
- [rwkozar](https://github.com/napari/napari/commits?author=rwkozar) - @rwkozar
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [vcwai](https://github.com/napari/napari/commits?author=victorcwai) - @victorcwai


## 27 reviewers added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [cnstt](https://github.com/napari/napari/commits?author=cnstt) - @cnstt
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Isabela Presedo-Floyd](https://github.com/napari/napari/commits?author=isabela-pf) - @isabela-pf
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justin Kiggins](https://github.com/napari/napari/commits?author=neuromusic) - @neuromusic
- [Justine Larsen](https://github.com/napari/napari/commits?author=justinelarsen) - @justinelarsen
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kim Pevey](https://github.com/napari/napari/commits?author=kcpevey) - @kcpevey
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Kushaan Gupta](https://github.com/napari/napari/commits?author=kushaangupta) - @kushaangupta
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Markus Stabrin](https://github.com/napari/napari/commits?author=mstabrin) - @mstabrin
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=potating-potato) - @potating-potato

