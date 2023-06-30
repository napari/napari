# Survey/Q&A

This guide contains questions that were submitted to our survey on testing.  

## This guide covers:   
- [What are the best practices to test a plugin with multiple sequential steps?](#what-are-the-best-practices-to-test-a-plugin-with-multiple-sequential-steps)  
- [How do you test widgets, the napari viewer, graphical user interfaces, and Qt in general?](#how-do-you-test-widgets-the-napari-viewer-graphical-user-interfaces-and-qt-in-general)  
- [How to find the different signals or slots?](#how-to-find-the-different-signals-or-slots)  
- [How do you avoid github tests failing?](#how-do-you-avoid-github-tests-failing)  
- [How do you make a process cancellable](#how-do-you-make-a-process-cancellable)  
- [Are there testing environments in napari?](#are-there-testing-environments-in-napari)  
- [Introduction to npe2? Migrating to new plug-in architecture?](#introduction-to-npe2-migrating-to-new-plugin-architecture)  
- [What is the optimal setup to quickly iterate in widget development?](#what-is-the-optimal-setup-to-quickly-iterate-in-widget-development)  
  
## What are the best practices to test a plugin with multiple sequential steps?
e.g. Is it ok to rely on the "results" of a test to run the next test or should they all be fully independent?  

Answer:  
* Ideally, aim for unit testing.  
* Tests should not pass/fail together.  
* Use [fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) to provide a test with inputs, even if you have to make them up.  
* Use [mocks (mock-ups)](https://docs.python.org/3/library/unittest.mock.html) to assert that specific calls are made, without necessarily caring about what happens after that call is made.  

*This is definitely an art form. It takes time. Be patient.*

## How do you test widgets, the napari viewer, graphical user interfaces, and Qt in general?
Answer:
* Try not to!
* You should generally trust that a button click (for example) will call your callback and focus on testing that your callback does what it's supposed to do given that it gets called following some UI interaction.
* However: If you have a scenario where you are actually creating a complicated widget directly in Qt, see `pytest-qt` for lots of tips, specifically `qtbot`.
    - [pytest-qt](https://pytest-qt.readthedocs.io/en/latest/intro.html)
    - [qtbot](https://pytest-qt.readthedocs.io/en/latest/reference.html?highlight=qtbot#module-pytestqt.qtbot)
* Oftentimes, this comes down to knowing and/or learning the Qt API really well.  
* Please see also the [In-depth guide to plugin testing](../testing_workshop_docs/index.md).
## How to find the different signals or slots?
Question: How can we find the different signals/slots we can connect callbacks to as the user interacts with the core napari interface e.g. creating/editing/deleting a `points` or `shapes` layer?

Answer: 
[https://napari.org/guides/stable/events_reference.html](https://napari.org/stable/guides/events_reference.html)  

Granted, this is a work in progress. 

For example, these events are emitted when the user interacts with the layer list: 
```console  
    Viewer.layers.events.inserted  
    Viewer.layers.events.removed  
    Viewer.layers.events.moved  
    Viewer.layers.events.changed  
    Viewer.layers.events.reordered  
```    

Getting an event when the user is editing the data inside a `points` or `shapes` layer (outside of the GUI interface) is complicated, because the user will be directly editing the native array object.

## How do you avoid github tests failing?  
Answer:  
* First make sure all your tests are passing locally.  
* After that, it's complicated. More background or context is needed to answer this question.  
  
## How do you make a process cancellable? 
Question: How do you make a process cancellable to interrupt a method that is running in a for loop, for example?  

Answer:  
* In single-threaded python, use `Ctrl-C`  
* In multithreaded python, there are many different patterns. Consider using a [generator-based thread worker](https://napari.org/stable/guides/threading.html#generators-for-the-win).    

## Are there testing environments in napari?
Answer: Napari does not create or otherwise manage environments.  
  
## Introduction to npe2? Migrating to new plugin architecture?     
Answer:  
* The primary difference is in how plugins are discovered:  
    - npe1 used decorators, requiring module import.  
    - npe2 uses static manifests (`napari.yaml`), describing contributions without requiring import.  
    - See also the [Your First Plugin tutorial](https://napari.org/stable/plugins/first_plugin.html)  
  
Additional resources:
* [Contributions Reference](https://napari.org/stable/plugins/contributions.html)  
* [Guides for each type of contribution](https://napari.org/stable/plugins/guides.html)  
* [Migration guide](https://napari.org/stable/plugins/npe2_migration_guide.html)  
  
## What is the optimal setup to quickly iterate in widget development?
Answer:   
* Create a script that will start napari and load your widget without any UI interaction.  
* Don't test as a plugin. Start by directly calling `viewer.window.add_dock_widget` with a manually created widget.  
* Familiarize yourself with the [IPython auto-reload features](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html).   
* Consider using `watchmedo` from [watchdog](https://github.com/gorakhargosh/watchdog).
  This will monitor a file/directory for changes, and re-run a command each time (which is why step #1 is also useful).  


## Other guides in this series:

* [Virtual environments](./1-virtual-environments.md)   
* [Deploying your plugin](./2-deploying-your-plugin.md)  
* [Version management](./3-version-management.md)  
* [Developer tools](./4-developer-tools.md)

This is the last guide in this series. 
