# Napari Tutorials

Welcome to Napari Tutorials. We provided couple tutorials to 
explain the usage modes and methods of Napari. Before we dive 
into further topics, let's check what Napari actually is.

## What is Napari

Napari is fast, interactive, multi-dimensional image viewer 
software. These tutorials are targeting people who want to use 
napari as a user, if you are interested in contributing then 
please check [Contributing Guidelines](../CONTRIBUTING.md). It 
is mainly developed in Python programming language. One can 
use napari from any Python shell or from a Jupyter notebook, and


<table border="0">
 <tr>
    <td><b style="font-size:30px">Scripting Usage</b></td>
    <td><b style="font-size:30px">Notebook Usage</b></td>
 </tr>
 <tr>
   <td>

        import napari
        from napari.util import app_context
        
        with app_context():
            # Code Here
   </td>
   <td>
   
        %gui qt5
        import napari       
   </td>
 </tr>
</table>