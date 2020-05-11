# Napari Performance

Performance is a core feature of napari. Performance is a concern for every
feature that napari implements. Without adequate performance users will migrate
to other visualization tools, even if napari has impressive functionality.

There are two main types of performance:

**Objective Performance**
* How long operations take when timed with a stopwatch.
* Most times will vary based on the data being viewed.

**Subjective Performance**
* The user’s experience as it relates to performance.
* Is the user’s experience pleasant or frustrating?
* Does napari "seem fast"?

Both types of performance are important. No amount of slickness can make up for
an application that is fundamentally too slow. And even a relatively fast
application can feel clunky or frustrating if not designed well.

## Objective Performance

**Focus On Real Use Cases**

* Focus on cases that matter to lots of people.
* It’s easy to waste time optimizing things no one cares about or no one will
  notice.
* If a dataset is unreasonable or out of scope or fringe, don’t waste time
  trying to make it run fast.

**Always Be Timing**
* Build timers into the software that always run.
* If not always visible, power users and developers should be able to toggle them on.
* This gives people an ambient awareness of how long things take.
* Allows users to report concrete performance numbers.
  *  “it seemed slow” becomes “it ran at 10Hz”.
  *  “it took a long time” becomes “it took 2 minutes and 30 seconds”.
* Teaches users how different hardware impacts performance.

**Performance System Tests**
* Create automatic tests that time specific operations in specific known datasets.
* Time many different operations on a nice selection of different datasets.

**Performance Unit Tests**
* Time one small operation to monitor for regressions.
* Interesting to see how different hardware performs.
* Napari has some of these today as "benchmarks".

**Run All Tests Every Merge**
* Save results to a database or somewhere.
* Catch a regression right when it happens and not weeks or
  months later.
* See how new features run on large datasets no one tested.

## Subjective Performance

Napari should strive to have these properties:

**Responsive**
* Two cases:
  * The full operation happens right away.
  * The interface clearly indicates the input was received and the operation was
    started.
* The UI should never seem dead, the user should never be left wondering if
  napari has crashed.
* For click or keypress events the **ideal response is 100ms**.
* For drag events or animations the **ideal refresh is 60Hz** which is 16.7ms per
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
* Show an busy animation as the last resort, never look totally locked up.
* Let power users see timings, bandwidth, FPS, etc.

## Performance Is Never Done

Performance is never "done" for several reasons:

**New Features**

* Every new feature must be evaluated for performance.
* Often new features work great on small datasets, but fall apart on large ones that the developer never tested.

**Regressions** 

* New features can slow down existing features.
* New versions of dependencies can slow things down.
* New hardware generally helps performance but not always.

**Scope Changes**

* New types of users adopt napari and have new use cases.
* Existing users change their usage over time, e.g more remote viewing.
* New file formats are invented or become more common.
* New data types or sizes become more common.

## New Features

* New features should be evaluated on their objective subjective and performance before merging to master.
* New features should be tested on a variety of data types and sizes, including the largest data sets that are supported.
* Test widely for regressions since new features can easily slow down existing features.
* The new feature should scale to large datasets, or the performance limitations of the feature should be well documented.
* It can be hard to impossible to "add performance in later".

## Conclusion

Achieving and maintaining performance requires an extreme amount of effort
and diligence, but the payoff is felt in every minute usage, if users are
delighted and able to productively do their work.
