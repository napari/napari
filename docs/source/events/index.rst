Events and Threading
====================

.. toctree::
   :maxdepth: 1

   event_loop
   threading

If you'd like to start customizing the behavior of napari, it pays to
familiarize yourself with the concept of an Event Loop. For an introduction to
event loops and connecting your own functions to events in napari, see the
:ref:`intro-to-event-loop`.

If you use napari to view and interact with the results of long-running
computations, and would like to avoid having the viewer become unresponsive
while you wait for a computation to finish, you may benefit from reading about
:ref:`multithreading-in-napari`.
