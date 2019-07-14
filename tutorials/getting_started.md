# napari

Napari
is mainly developed in Python programming language. One can
use napari from any Python scripting setup or from a Jupyter notebook,
and what you should be importing napari:


<table border="0">
 <tr>
    <td><b style="font-size:30px">Scripting Usage</b></td>
    <td><b style="font-size:30px">Notebook Usage</b></td>
 </tr>
 <tr>
   <td>

```python
import napari

with napari.qui_qt():
    # Code Here
```
   </td>
   <td>

```python
%gui qt5
import napari

# Code Here
```
   </td>
 </tr>
</table>

Other than the initial setup and importing step basically napari API is
same for different usage modes.

## Napari Conventions

We love numpy arrays in napari. We are using them heavily. We also love pythonic
way of doing things.
