# 1: Python's assert keyword

This tutorial defines the assert keyword in Python and shows how it can be used to write test cases for a simple function.  

## Other lessons in this tutorial:  

* 1: This lesson (Python's assert keyword) 
* 2: [Pytest testing framework](./2-pytest-testing-frameworks)  
* 3: [Readers and fixtures](./3-readers-and-fixtures)  
* 4: [Test coverage](./4-test-coverage)  
* Resource links: [Testing resources](./testing-resources)  
  
### This lesson covers:  
[Assert keyword](#assert-keyword)  
[Test for Pass](#test-for-the-pass-case)  
[Test for Fail](#test-for-the-fail-case)  

### Resources
The example plugin and all the tests discussed in this lesson are available in [this GitHub repository](https://github.com/DragaDoncila/plugin-tests).

## Assert keyword  
The key to testing in Python is the [assert](https://realpython.com/python-assert-statement/) keyword. We *assert* a Boolean expression is true and create an error message that appears when that expression is false. 

```python
assert <Boolean expression>, <error message>    
```

If it is true, code execution continues as though the assert statement doesn’t exist. If the Boolean expression is false, an `AssertionError` is thrown, an exception raised, and the error message displayed. 
  
Here is a simple function, `get_grade_from_mark`. It takes a mark (score) from zero to 100. If the mark (score) is more than 50, the grade is `Pass`; if it’s less than 50, the grade is `Fail`.  

```python
def get_grade_from_mark(mark):
    if mark > 50: 
        return "Pass"
    else:   
        return "Fail"
```

`get_grade_from_mark` can be tested by writing two test functions. The first one is for when the mark is > 50.   
  
  
## Test for the Pass case
When the mark is > 50, call `get_grade_from_mark` and assert that the grade is what we expect (either `Pass` or `Fail`). When testing the passing case, test that the grade is `Pass`. If it's not `Pass`, the best practice is to write a helpful error message like, `"Expected {mark} to pass but result was {grade}."`

```python
def test_get_grade_pass(mark):
    grade = get_grade_from_mark(mark):
    assert grade == "Pass", f"Expected {mark} to pass, but result was{grade}"
```
  
## Test for the Fail case
Test the same thing for `Fail` to test all options. Everything is almost the same, so copy and paste and change a few words to create the second test.  

```python  
def test_get_grade_fail(mark):
    grade = get_grade_from_mark(mark):
    assert grade == "Fail", f"Expected {mark} to fail, but result was{grade}"
```

We can now write code to run both of the functions with expected values. For example, we expect 65 to pass and 43 to fail. After running both functions, if no exception has been raised we print “All passing.”  
```python  
if __name__ == "__main__":
    test_get_grade_pass(65)
    test_get_grade_fail(43)
    print("All passing.")
```
We can place all this code in a Python file, e.g. [example_test.py](https://github.com/DragaDoncila/plugin-tests/blob/main/example_func.py), and run it from the command line. 
```console
(napari-env) user@directory % python example_test.py
All passing.
```

We now update the test to see what a failure looks like:  
```python
if __name__ == "__main__":
    test_get_grade_pass(65)
    test_get_grade_fail(70)  # updated this to 70 to force failure
    print("All passing.")
```
If we assert that 70 fails, the `AssertionError`, “Expected something to fail, but the result was pass.” would be thrown, which is correct. It would look like this:  
```console
(napari-env) user@directory % python example_func.py  
Traceback (most recent call last):  
File “/Users/user/directory/example_test.py” line 20, in <module> test_get_grade_fail(70)  
File “/Users/user/directory/example_test.py” line 16, in <module> test_get_grade_fail  
Assert grade == “Fail”, f”Expected {mark} to fail, but result was {grade}”  
AssertionError: Expected 70 to fail, but result was Pass.   
```
Note that when the assertion fails, a traceback occurs. 

This example is a simple way to demonstrate the use of the assert keyword, but it’s not particularly useful for testing a larger codebase. This test function has to be called explicitly to test different marks. There’s not much detail when the code is running. We just get `“All passing.”` and there’s no information about other tests when one of the tests fails.  
  
Making testing more convenient is where [testing frameworks](./2-pytest-testing-frameworks), like [pytest](https://docs.pytest.org/) come in.  
