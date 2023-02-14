# 2: Pytest testing framework  

This lesson explains how to use the [pytest testing framework](https://docs.pytest.org/en/7.2.x/) to make testing easier.

## Other lessons in this tutorial:   

* 1: [Python’s assert keyword](./1-pythons-assert-keyword.md) 
* 2: This lesson (Pytest testing framework)  
* 3: [Readers and fixtures](./3-readers-and-fixtures.md)  
* 4: [Test coverage](./4-test-coverage.md)  
* Resource links: [Testing resources](./testing-resources.md)  
  
### This lesson covers:  
* [Testing framework features](#testing-framework-features)  
* [Parametrization](#parametrization)  

### Resources  
The example plugin and all the tests discussed in this lesson are available in [this GitHub repository](https://github.com/DragaDoncila/plugin-tests).
  
## Introduction  
We are using pytest as a testing framework. It provides convenience tools to assist with testing. For example, it can discover tests for you if you point it to a directory or a file.  It can be installed using `pip install pytest`.

## Testing framework features  
Testing frameworks provide a whole host of useful features, including:  
* Test discovery - directories can be crawled (searched) to find things that look like tests and run them
* Housekeeping and ease of use - convenient methods for writing tests and cleaning up after running the tests  
  
Pytest goes through the target destination, such as a file or directory, finding any method or function prefaced with the word `test`. It runs all the methods and functions prefaced with the word `test` but _not_ the code under the main block. When `pytest` runs against `example_test.py` (refer to the [Python's assert keyword](./1-pythons-assert-keyword.md) lesson), it finds several tests that all pass.  

If the tests fail, `pytest` is very good at tracing back the reason they failed and showing their values throughout test execution. In more complicated examples, this traceback mechanism can be very helpful. In this example, the message is that we got a `Pass` but were expecting a `Fail`. See the lines below that show the `assert` keyword and the errors.  

```console
(napari-env) user@directory % pytest   

example_test.py  
    
========================== test session starts ==========================  
Platform darwin -- 3.9.7, pytest-6.2.5, py-1.11.0, pluggy-1.0.0  
PyQt5 5.15.6 -- Qt runtime 5.15.2 -- Qt compiled 5.15.2  
rootdir: /Users/user/directory  
Plugins: qt-4.0.2, napari-plugin-engine-0.2.0, napari-0.4.13  
Collected 2 items  

example_func.py .F  
    
================================= FAILURES ===============================  
—-------------------------- test_get_grade_fail --------------------------  
def test_get_grade_fail():  
    grade = get_grade_from_mark(65)  
>   assert grade == “Fail”, f”Expected 65 to fail but result was {grade}”  
E   AssertionError: Expected 65 to fail but result was Pass  
E   assert ‘Pass’ == ‘Fail’  
E     - Fail  
E     + Pass  
    
example_func_py: 16: AssertionError  
======================== short test summary info =========================  
FAILED example_func_py::test_get_grade_fail - AssertionError: Expected 65 to fail, but result was Pass  
====================== 1 failed in 0.56s =======================  
(napari-env) user@directory %   
```

## Parametrization  
Another very useful tool that pytest provides is parametrization.  
    
We've tested these functions with a single value. We need to be more thorough. Pytest allows us to parametrize tests. We decorate our function with `@pytest.mark.parametrize` and pass the decorator a parameter name, `mark`, as a string, and a list of values for which we’d like to run the test function. Note that we pass in 50 as an edge case; it's the lowest mark that will pass.  

```python
import pytest

def get_grade_from_mark(mark):
    if mark > 50
        return "Pass"
    else: 
        return "Fail"

@pytest.mark.parametrize("mark", [65, 80, 50])
def test_get_grade_pass(mark):
    grade = get_grade_from_mark(mark):
    assert grade == "Pass", f"Expected {mark} to pass, but result was{grade}"

@pytest.mark.parametrize("mark", [40, 25])
def test_get_grade_fail(mark):
    grade = get_grade_from_mark(mark):
    assert grade == "Fail", f"Expected {mark} to fail, but result was{grade}"

if _name_ == "_main_";
    test_get_grade_pass(65)
    test_get_grade_fail(30)
    print("All passing.")
```

We run `pytest` which finds and runs `test_get_grade_pass(mark)`. `test_get_grade_pass(mark)` fails because the code says `mark > 50`, so a mark of 50 still fails. The code should say `>=50` for a mark to pass.  
   
```console
example_func.py ..F..
================================= FAILURES ===============================  
—------------------------ test_get_grade_pass[50] ------------------------  
mark = 50  
@pytest parametrize(‘mark’, [65, 80, 50])  
def test_get_grade_pass(mark):  
    grade = get_grade_from_mark(mark)  
>   assert grade == “Pass”, f”Expected {mark} to pass, but result was {grade}”  
E   AssertionError: Expected 50 to pass but result was fail  
E   assert ‘Fail’ == ‘Pass’  
E     - Pass  
E     + Fail  

example_func_py:12: AssertionError  
======================== short test summary info =========================  
FAILED example_func_py::test_get_grade_pass[50] - AssertionError: Expected 50 to pass, but result was Fail   
```

Another valuable feature of `pytest` is the `pytest-cov` option discussed in the [Test coverage](./4-test-coverage.md) lesson  

The next lesson in this tutorial on testing is the [Readers and fixtures](./3-readers-and-fixtures.md) lesson. 
