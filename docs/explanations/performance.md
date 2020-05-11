# Napari Performance

Performance is a core feature of napari. Performance is a concern for every
feature that napari implements. Without adequate performance users will migrate
to other visualization tools, even if napari has impressive functionality.

## Types of Performance

There are two main types of performance:

**Objective Performance**
* How long operations take when timed with a stopwatch.
* Most (but not all) times will vary based on the data.

**Subjective Performance**
* The user’s experience as it relates to performance.
* Is the user’s experience pleasant or frustrating?

Both types of performance are important. No amount of slickness can make up for
an application that is fundamentally too slow. And even a relatively fast
application can feel clunky or frustrating if not designed well.

## Objective Performance

**Focus On Real Use Cases**

* It’s easy to waste time optimizing things no one cares about or no one will
  notice.
* Focus on cases that matter to lots of people.
* If a dataset is unreasonable or out of scope or very fringe, don’t waste time
  trying to make it run fast.

**Always Be Timing**
* Build timers into the software that always run.
* They do not necessarily have to be visible, but power users and developers
  should be able to see them.
* This gives people an “ambient awareness” of how long things take. Users will
  often load data no developer has seen, useful to know how slow it is.
* Allows users to report concrete performance numbers instead of “it seemed
  slow” they’ll say “it ran at 10Hz.”. Instead of “it took a long time” they can
  see “it took 4 minutes and 30 seconds”
* Teaches users to be aware how different hardware impacts performance.

**Performance System Tests**
* Create automatic tests that time specific operations in specific known datasets.
* Time many different operations on a nice selection of different datasets.

**Performance Unit Tests** (benchmarks)
* Create automatic tests that time one small operation to monitor for
  regressions. Like a single matrix operation.
* Interesting to see how different hardware performs.
* Napari has some of these today.

**Run All Tests Often**
* Saver results to a database or somewhere.
* It’s useful to catch a regression right when it happens and not weeks or
  months later.
* It’s important for developers to see how new features run on large datasets
  they might not have tested.

## Subjective Performance

Subjective performance is an ongoing battle. Napari should strive to have these properties:

**Responsive**
* The full operation happens or the interface clearly indicates the input was
  received and the operation was started.
* The UI should never seem dead, the user should never be left wondering if
  napari has crashed.
* For click or keypress events the ideal response is 100ms.
* For drag events or animations the ideal refresh is 60Hz which is 16.7ms per
  frame.

**Interruptible**
* Modeless operations can be interrupted by simply performing some other action.
  If imagery is loading in the background, you can interrupt it just by
  navigating to somewhere else.
* Modal operations that disable the UI should have a cancel button when possible
  unless they are very short.
* The user should never feel “trapped”.

**Progressive**
* Show intermediate results as they become available, instead of showing nothing
  until the full result is ready.
* Sometimes progressive results are better even if they slow things down a bit,
  which is not necessarily intuitive.

**Informative**
* Clearly show what controls are enabled or disabled.
* If progressive display is not possible, show progress bars.
* Show time estimates for super long operations.
* Show a “busy” animation as the last resort, never look totally locked up.
* Let power users see timings, bandwidth, FPS, etc.

## Performance Is Never Done

Performance is never "done" for several reasons:

**New Features**

* Every new feature must be evaluated for performance.
* Often new features work great on small datasets, but fall apart on large ones
  that the developer never tested.

**Regressions** 

* New features can slow down existing features.
* New versions of dependencies can slow things down.
* New hardware generally helps performance but not always.

**Scope Changes**

* New types of users adopt napari and have new needs.
* Existing users change their usage over time, e.g more network/remote viewing.
* New file formats are invented or become more common.
* New data types or sizes become more common.

## New Features

1. New features should be evaluated on their objective and subjective
   performance.
1. New features should be tested on a variety of data types and sizes, including
   the largest data sets that are supported.
1. It's easy to create features that do not "scale" well to large datasets.
   Either the feature should scale well, or the limitations of the feature
   should be well documented. It can be hard to impossible to "add performance
   in later".
1. Existing features should be tested both automatically and manually for
   performance. Well meaning new features can slow down existing features.
