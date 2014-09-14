Fado
====

Julia language: Forward automatic differentiation using overloading.  Low performance, high functionality approach

History and goal
----------------
This is fresh stuff, first uploaded to GitHub in September 2014.  You are basicaly beta-testing, and your feedback is welcome.
The objective is to inline this package with the interfaces defined or being defined, in the JuliaDiff project.

The strategy used (Arrays{Any} containing objects which are tree structures, so with pointers and garbage collection everywhere)
will be found to yield low performance.  

However, this package has a very functionality, and will be usefull, as a replacement until more performant systems are
in place, and as a reference when debuggin.

Functionality
-------------
Fado allows differentiation to arbitrary levels, with respect to scalars or arrays of any dimension, 
derivatives of functions contain derivatives, cross derivatives.

Fado does not create sparse derivatives.  It was created with finite element methods in mind, with the goal
to make the implementation of new elements faster. Element matrices (e.g. derivative of forces with respect to displacements)
are full, and are then to be assembled, outside this package, into sparse matrices.

Fado comprises of two modules

MORE COMMING
