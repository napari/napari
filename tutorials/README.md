# Napari Tutorials

Welcome to Napari Tutorials. We provided couple tutorials to 
explain the usage modes and methods of Napari. Before we dive 
into further topics, let's check what Napari actually is.

## What is Napari

Napari is fast, interactive, multi-dimensional image viewer 
software. It is being developed by bunch of enthusiasts. It is a 
community project. These tutorials are targeting people who want to use 
napari as a user, if you are interested in contributing then 
please check [Contributing Guidelines](../CONTRIBUTING.md). Napari 
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

## Continue with

- [Viewer tutorial](viewer.md)
- [nD tutorial](arbitrary_dimensional.md)
- [Image layer tutorial](images.md)
- [Label layer tutorial](labels.md)
- [Point layer tutorial](points.md)
- [Shape layer tutorial](shapes.md)
- [Pyramid layer tutorial](pyramid.md)
- [Vector layer tutorial](vectors.md)
