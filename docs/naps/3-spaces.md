(nap3)=

# NAP-3 — Spaces

```{eval-rst}
:Author: Lorenzo Gaifas
:Created: 2022-06-08
:Status: Draft
:Type: Standards Track
``` 

# Abstract

`napari` is currently limited to holding (and rendering) data belonging to a single, universal space (which we often refer to as *world space*). However, it is often useful to have quick and easy access to different parts of a dataset that do no belong to the same absolute coordinate system; for example, there is often no reason to relate between the spatial coordinates of `image 1` and `image 2` from the same microscopy data collection, but it might be useful to quickly switch between the two to compare the effectiveness of a processing step. Currently, to do so, a user is forced to either load everything into the layer list (pretending that the coordinate spaces do indeed coincide), or write a plugin that manages layers externally and feeds them to `napari` as desider.

This NAP discusses the reasons why a native napari approach would be better than these two options. It then proposes the introduction of `spaces` as a way to manage different coordinate spaces in the same viewer.

# Motivation
