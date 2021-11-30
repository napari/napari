(napari-performance)=
# napari performance

With offline analysis tools performance dictates how long the user has to wait
for a result, however with an interactive tool like napari performance is even
more critical. Therefore performance is a core feature of napari.

Inadequate performance will leave the user frustrated and discouraged and they
will migrate to other tools or simply give up on interactive exploration of
their data altogether. In contrast, excellent performance will create a joyful
experience that encourages longer and more intensive exploration, yielding
better scientific results.

There are two main types of performance:

1. [Objective performance](#objective-performance)

   * How long operations take when timed with a stopwatch.
   * Most times will vary based on the data being viewed.

2. [Subjective performance](#subjective-performance)

   * The user’s experience as it relates to performance.
   * Is the user’s experience pleasant or frustrating? Does napari "seem fast"?

Both types of performance are important. No amount of slickness can make up for
an application that is fundamentally too slow. And even a relatively fast
application can feel clunky or frustrating if not designed well.

## Objective performance

How to keep napari objectively fast:

### Focus on real use cases

* Focus on cases that matter to lots of people.
* It’s easy to waste time optimizing things no one cares about or no one will
  notice.
* If a dataset is unreasonable or out of scope or fringe, don’t spend too
  many resources trying to make it run fast.

### Always be timing

* Build timers into the software that always run.
* If not always visible, power users and developers should be able to toggle them on.
* This gives people an ambient awareness of how long things take.
* Allows users to report concrete performance numbers:
  * *it seemed slow* → *it ran at 10Hz*.
  * *it took a long time* → *it took 2 minutes and 30 seconds*.
* Teaches users how different hardware impacts performance.
  * For example seek times with SSD are radically faster than HDD.
  * Become familiar with the impact of local vs. networked file systems.

### Performance system tests

* Create automatic tests that time specific operations in specific known datasets.
* Time many different operations on a nice selection of different datasets.

### Performance unit tests

* Time one small operation to monitor for regressions.
* Napari has some of these today as "benchmarks".
* Interesting to see how different hardware performs as time goes on.

### Run all tests every merge

* Save results to a database maybe using [ASV](https://asv.readthedocs.io/en/stable/index.html).
* Catch a regression right when it happens and not weeks or
  months later.
* See how new features run on large datasets no one tested.

## Subjective performance

Napari should strive to have these properties:

### Responsive

* React to input one of two ways:
  * The full operation happens right away.
  * The interface clearly indicates the input was received and the operation was
    started.
* For click or keypress events the **ideal response is 100ms**.
* For drag events or animations the **ideal refresh is 60Hz** which is 16.7ms per
  frame.
* The UI should never seem dead, the user should never be left wondering if
  napari has crashed.

### Interruptible

* Modeless operations are best. They can interrupted by simply performing some
  other action. For example if imagery is loading in the background you can
  interrupt it just by navigating to somewhere else.
* Modal operations that disable the UI should have a cancel button when possible
  unless they are very short.
* The user should never feel “trapped”.

### Progressive

* Show intermediate results as they become available instead of showing nothing
  until the full result is ready.
* Sometimes progressive results are better even if they slow things down a bit,
  which is not necessarily intuitive.

### Informative

* Clearly show what controls are enabled or disabled.
* If progressive display is not possible, show a progress bar.
* Show a busy animation as the last resort, never look totally locked up.
* Show time estimates for super long operations.
* Let power users see timings, bandwidth, FPS, etc.
* Revealing internal state that explains why it's taking time is helpful.

## Performance is never done

Performance is never "done" for several reasons:

### New features

* The objective and subjective performance of new features should be scrutinized
  before merging to main.
* New features should be tested on a variety of data types and sizes, including the largest data sets that are supported.
* The new feature should scale to large datasets, or the performance limitations of the feature should be well documented.
* It can be hard to impossible to "add performance in later". The best time to
  ensure the new feature performs well is when the feature is first added.

### Regressions*

* We must be on guard for new features slowing down existing features.
* New versions of dependencies can slow things down.
* New hardware generally helps performance but not always.

### Scope changes

* As new types of users adopt napari they will have new use cases.
* Existing users will change their usage over time such as more remote viewing.
* New file formats will be invented or become more common.
* New data types or sizes will become more common.

## Conclusion

Achieving and maintaining performance requires an extreme amount of effort and
diligence, but the payoff will be felt in every minute of usage. The end result
is delighted and productive users.
