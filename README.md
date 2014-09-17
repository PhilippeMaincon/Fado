Fado
====

Julia language: Forward automatic differentiation using overloading.  A compact low performance, high functionality approach.

History and goal
----------------
This is fresh stuff, first uploaded to GitHub in September 2014.  You are basicaly beta-testing, and your feedback is welcome. An objective for further development is to align this Fado with the interfaces defined or being defined, in the JuliaDiff project.

The strategy used here (Arrays{Any} containing objects which are tree structures, so with pointers and garbage collection everywhere) will means that this code will not be fast.  

However, this package has a wide functionality, and will be usefull, as a placeholder until more performant systems are in place, and as a reference when debugging.

Functionality
-------------
Fado allows differentiation to arbitrary levels, with respect to scalars or arrays of any dimension, derivatives of functions contain derivatives, cross derivatives.

Fado does not create sparse derivatives.  It was created with finite element methods in mind, with the goal to make the implementation of new elements faster. Element matrices (e.g. derivative of forces with respect to displacements) are full, and are then to be assembled, outside this package, into sparse matrices.

Fado comprises of two modules: Fado and Leibnitz each in the correspondingly name file.

Fado.jl
-------
Fado.jl defines the type Fado, which is a "nested dual".  A dual number is a number of the form a+e*b, where e^2=0 and a and b are the "standard" and "tangent" parts respectively.  If a is the variable, and b = da/dx, thens it turns out that adding, multiplying and dividing duals will also cause the tangent of the result to still be the derivative.

In the following example, the function "stir" creates a Fado (and a tag which we will worry about later), with standard part equal to the input, and tangent part equal to da/da = 1 (and an "id" which we will worry about later).

a       = 3.
(A,da)  = stir(3.)  # (3.,1.,1)
B       = A*A       # (3. *3.,3. *1.+1.*3.,1)=(9.,6.,1)
b       = st(B)     # 9.
b_a     = tg(B,da)  # 6.

Fados can nest (nested duals) to allow computing higher derivatives.

a = 3.
b = 4.
(A,da) = stir(a) # (3.,1.,1)
(B,db) = stir(b) # (4.,1.,2)
C      = A*B    # ((c,dc/da,1),(dc/db,ddc/dadb,1),2) = ((12.,4.,1),(3.,1.,1),2)
c      = st(st(C))        # 12.
c_a    = tg(st(C),da)     # 4.
c_b    = st(tg(C,db))     # 3.
c_ab   = tg(tg(C,db),da)  # 1.

OK, the unpacking is a wee bit cryptical. It helps to thing of multiple branchings down a tree structure.

Note that A*B produced ((c,dc/da,1),(dc/db,ddc/dadb,1),2) (as wanted), not (a*b,a+b,?).  That is because A and B have different id's.

The system will work correctly if for example c happens not to be a function of a (this is why tags like da and db are needed)
a = 3.
b = 4.
(A,da) = stir(a) # (3.,1.,1)
(B,db) = stir(b) # (4.,1.,2)
C      = A*A     # (9.,6.,1)
c      = st(st(C))        # 9.
c_a    = tg(st(C),da)     # 6.
c_b    = st(tg(C,db))     # 0.
c_ab   = tg(tg(C,db),da)  # 0.

We can also compute higher order derivatives

(A,da) = stir(a,2) # ((a,da/da=1.,1),(da/da=1.,dda/dada=0.,1),2)
B      = foo(A)
b      = tg(B)
b_a    = st(tg(B,da))
b_a    = tg(st(B),da))
b_aa   = tg(tg(B,da),da)

Note that there are two ways to retrive b_a, because it is store twice, and innefficiency inherent to this system.

The function stir can be applied to a Fado, and everything will work out just fine.  This implies in particular that one can differentiate a function that itself happens to use differentiation, and this is completely transparent.

One can "stir" a vector, matrix or array of any dimension.

a      = randn(2,3)
(A,da) = stir(a)
B      = foo(A)
b      = st(B)    # size (4,5,6)
b_a    = tg(B,da) # size (2,3,4,5,6)
