# 4: Test coverage

## Other lessons in this tutorial:   

* 1: [Python’s assert keyword](./1-pythons-assert-keyword.md) 
* 2: [Pytest testing framework](./2-pytest-testing-frameworks.md)  
* 3: [Readers and fixtures](./3-readers-and-fixtures.md)  
* 4: This lesson (Test coverage)   
* Resource links: [testing resources](./testing-resources.md)   

### This lesson covers:   
* [Coverage](#coverage)
* [pytest --cov](#pytest---cov)    
 
#### Resources  
The example plugin and all the tests discussed in this lesson are available in [this GitHub repository](https://github.com/DragaDoncila/plugin-tests).    

## Coverage  
How do we know when we have tested everything? 

This is where _coverage_ comes in. `pytest-cov` can find out what the coverage of our tests is. Install using 
`pip install pytest-cov`. This feature provides coverage data for pytest. Coverage is the lines of code that were executed when the tests were run. If there are lines of code that didn’t run at all, they represent a code path we didn’t test, which could hold potential bugs. Coverage gives you an idea of other tests that are needed. 

## pytest --cov  

To run tests with coverage, run `pytest` and add `--cov` pointing to the module you're covering. There is also an option to generate an html report, which we do in this case. [Note that the `.` (period character) is part of the command.]  

```console
(napari-env) user@directory % pytest --cov=plugin_tests --cov-report=html .  

=========================== test session starts ==========================  
platform darwin -- Python 3.9.7, pytest-6.2.5, py-1.11.0, pluggy-1.0.0  
PyQt 5.15.6 -- Qt runtime 5.15.2 -- Qt compiled 5.15.2  
rootdir: qt-4.0.2, napari-plugin-engine-0.2.0, napari-0.4.13, cov-3.0.0  
collected 2 items  

src/plugin_tests/_tests/test-reader.py ..                           [100%]  

--------- coverage: platform darwin, python 3.0.7-final-0 --------  
Coverage HTML written to dir htmlcov  

=========================== 2 passed in 6.72s ==========================  
```

This command runs the tests, gets the coverage, and then writes the Coverage HTML report to the `htmlcov` directory.

There is a large folder (`htmlcov`) in the directory where the tests were run (`plugin_tests`). 

```console
(napari-env) plugin-tests % tree -L 1`  
.  
├── LICENSE    
├── MANIFEST.in  
├── README.md  
├── _pycache__  
├── example_func.py  
├── htmlcov		        # <<=============== directory created by pytest-cov
├── requirements.txt  
├── setup.cfg  
├── setup.py  
├── src  
└── tox.ini  
```

If we open the `index.html` file from the list of files in the left panel (to the left of line 32) in a browser, we can see the coverage report. 

![htmlcov directory](../../images/test_coverage_htmlcov_directory.png)

![Coverage Report](../../images/coverage_report.png)

We are interested in `_reader.py`. The file containing the reader code has 86% coverage (see below). Clicking ok on the `2 missing` box below highlights the lines that were never run at all. They are highlighted in red (lines 22 and 26): 

![Lines not run highlighted in red](../../images/lines_not_run_highlighted_in_red.png)

Because we never provided a list of paths, we don't know what will happen in that case. We also never ran code that tests not returning a reader. In other words, we never tested the failed cases. We can and should add those tests. The first one is `test_get_reader_pass`. We'll call it with a file that doesn't end in `.npy` and assert that it returned `None`. Then we'll create a second test to call with a list of paths.

Using the `write_im_to_file` fixture again, we can write two files, call `napari_get_reader` with two paths and assert that it still returns a callable.
```python
def test_get_reader_pass():  
    """Calling get_reader with non-numpy file path returns None"""  
    reader = napari_get_reader("fake.file")  
    assert reader is None  
    
def test_get_reader_path_list(write_im_to_file):  
    """Calling get_reader on list of numpy files returns callable"""  
    pth1, _ = write_im_to_file("myfile1.npy")
    pth2, _ = write_im_to_file("myfile2.npy")
    
reader = napari_get_reader([pth1, pth2])  
assert callable(reader)  
```

If we re-run `pytest`, the coverage report is also updated and coverage should improve.

The coverage report goes to the same folder, `htmlcov`, so we should be able to refresh the page without opening the file again. We've now got 100% coverage of `_reader.py` now. See below.

![second coverage report](../../images/second_coverage_report.png)    

There could be other, more complicated cases that we have not tested, but at the very least, we are executing all lines of code.

If the html report seems cumbersome, we can print the coverage directly to the terminal with `--cov-report=term-missing`. This command is on the [slides](https://docs.google.com/presentation/d/1vD1_jhK6Xjqltmlp5Q2auXkgkvQTrr2d77_a9TqD6yk/edit#slide=id.g10c4a0816be_0_24). That will print the exact lines of code you haven’t tested.

We've got 100% coverage in the reader, no lines missed. Many `napari` plugins contain widgets though - testing widgets is a topic for another time.
