# Profiling

In comparison to performance tracing profiling does not provide information about time of events but provide 
combined information about time of execution of functions and allow producing call graph which simplify tracing 
relations between functions.   

![Example part of execution graph](images/execution_graph.png)

The basic tool for perform profile is build in python module `cProfile`. For profile whole script use:

```bash
python -m cProfile path_to_script.py
```

The output of this call will be such table:

```
         2334264 function calls (2267576 primitive calls) in 2.242 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    21241    0.252    0.000    0.252    0.000 {method 'reduce' of 'numpy.ufunc' objects}
80547/57578    0.080    0.000    0.765    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
     1048    0.060    0.000    0.060    0.000 {built-in method marshal.loads}
    62683    0.052    0.000    0.052    0.000 {built-in method numpy.array}
  122/117    0.042    0.000    0.047    0.000 {built-in method _imp.create_dynamic}
     5152    0.042    0.000    0.053    0.000 stride_tricks.py:114(_broadcast_to)
     5102    0.037    0.000    0.048    0.000 decomp_qr.py:11(safecall)
     2551    0.031    0.000    0.386    0.000 transform_utils.py:153(decompose_linear_matrix)
     2651    0.029    0.000    0.067    0.000 transforms.py:342(__call__)
2172/2108    0.029    0.000    0.087    0.000 {built-in method builtins.__build_class__}
    18283    0.027    0.000    0.278    0.000 fromnumeric.py:70(_wrapreduction)
...
```
To understand this output we suggest reading: https://docs.python.org/3/library/profile.html#instant-user-s-manual.
Because output for napari will be very long we suggest ot pip output to `less` command or save it to file,
which could be investigated later using command:

```bash
python -m cProfile -o result.pstat path_to_script.py
```

There are few options for check content of `pstat` file.  

1.  The Stat object.
    
    Profiling output could be parsed and viewed using `Stats` object from `pstat` library. Example usage:
    ```python
    from pstats import Stats
    stat = Stats("path/to/result.pstat")
    stat.sort_stats("tottime").print_stats(10)
    ```
    Documentation here https://docs.python.org/3/library/profile.html#the-stats-class

2.  Snakeviz.
    
    If You do not have `snakeviz` command available then ensure than your developer environment 
    is active and call `pip install snakeviz`. To visualize profiling use command:
    ```bash
    snakeviz path/to/result.pstat   
    ```
    Then in your browser should be opened a new tab with similar content
    ![Snakeviz example view](images/snakeviz.png)

3.  gprof2dot

    For using this method there is need to have [`graphviz`](https://graphviz.org/) installed in your system.  
    To install:
    
    * Ubuntu: `sudo apt install graphviz`
    * MacOS with brew: `brew install graphviz`
    * Windows with choco `choco install graphviz`

    then use `gprof2dot` (install with `pip install gprof2dot`) to convert `.pstat` to `.dot` file and use graphviz:

    ```bash
    $ python -m gprof2dot -f pstats  -n 5  result.pstat -o result.dot
    $ dot -Tpng -o result.png result.dot
    ```
    
    If the shell support piping it could be combined into one command:
    
    ```bash
    $ python -m gprof2dot -f pstats  -n 5  result.pstat -o | dot -Tpng -o result.png
    ```

4.  PyCharm professional allows to view `.pstat` file using Tools > Open CProfile snapshot.

cProfile allows also for profile not only whole script, but also part of code. These needs code modification.
We suggest usage of `cProfile.Profile()` as a context manager:

```python
import cProfile

with cProfile.Profile() as pr:
    code_for_profile()
pr.dump_stats("result.pstat")
```

Then It could be visualized using same methods.

To profile code which needs first few steps done in viewer code, then code for profile could be hidden under some button.

For example:

```python
def testing_fun():
    with cProfile.Profile() as pr:
        code_for_profile()
    pr.dump_stats("camera_layers.pstat")

testing_btn = QPushButton("Profile")
testing_btn.clicked.connect(testing_fun)
viewer.window.add_dock_widget(testing_btn)
```

Alternative profilers available in python are:

*  `yappi` with support for multithreading
*  `vmprof` 

Both could be installed from using `pip`.