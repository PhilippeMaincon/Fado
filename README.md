Fado
====

Julia language: Forward automatic differentiation using overloading.  A compact low performance, high functionality approach.

History and goal
----------------
This is fresh stuff, first uploaded to GitHub in September 2014.  You are basicaly beta-testing, and your feedback is welcome. An objective for further development is to align this Fado with the interfaces defined or being defined, in the JuliaDiff project.

The strategy used here (`Arrays{Any}` containing objects which are tree structures, so with pointers and garbage collection everywhere) will means that this code will not be fast.  

However, this package has a wide functionality, and will be usefull, as a placeholder until more performant systems are in place, and as a reference when debugging.

Functionality
-------------
Fado allows differentiation to arbitrary levels, with respect to scalars or arrays of any dimensions, derivatives of functions that are themselves implemented using derivatives, and cross derivatives.

Fado does not create sparse derivatives.  Fado was created with (originaly) finite element methods in mind, with the goal to make the implementation of new elements faster. Element matrices (e.g. derivative of forces with respect to displacements) are full, and are then to be assembled, outside this package, into sparse matrices.

Fado comprises of two modules: `Fado` and `Leibnitz` each in the correspondingly named file.

`Fado.jl`
---------
`Fado.jl` defines the type Fado, which is a "nested dual".  A dual number is a number of the form `a+e*b`, where `e^2=0` and `a` and `b` are the "standard" and "tangent" parts respectively.  If `a` is the variable, and `b = da/dx`, thens it turns out that adding, multiplying and dividing duals will also cause the tangent of the result to still be the derivative.

In the following example, the function `stir` creates a `Fado` (and a tag `da` which will be discussed later).  This `Fado` has a standard part equal to the input, and a tangent part equal to `da/da = 1` (and an `id` which we will worry about later). `stir` can only be called with a `Float32` (or a `Fado` as will be discussed later).

    # BASIC USE
    a       = 3.
    (A,da)  = stir(3.)  # (st=3.,tg=1.,id=1)
    B       = A*A       # (3. *3.,3. *1.+1.*3.,1)=(9.,6.,1)
    b       = st(B)     # 9.
    b_a     = tg(B,da)  # 6.

The operation `B = A*A` creates a new `Fado` `B` (Variables of type `Fado` in this documentation will be uppercase), which is then "unpacked" using functions `st` (standard part) and `tg`(tangent part).

Only access the standard and tangent parts of a `Fado` using the functions `st` and `tg`.  The following code should be avoided  (breach of implementation hiding, and more).

    # DON'T
    # A = stir(3.)
    # A._st      # Please, just don't
    # A._tg      # unless you are studying the code or debbuging

`Fado`s can nest (nested duals) to allow computing higher derivatives and/or cross derivatives. In other words the `tg` or `st`part of a `Fado` can itself be a `Fado`.  There are two distinct ways in which nested `Fado`s can appear.  The first is the "cross derivatives scenario".  The nested `Fado` is created by the line `C = A*B` in the following:

    # CROSS DERIVATIVES
    a = 3.
    b = 4.
    (A,da) = stir(a) # (3.,1.,1)
    (B,db) = stir(b) # (4.,1.,2)
    C      = A*B     # ((c,dc/da,1),(dc/db,ddc/dadb,1),2) = ((12.,4.,1),(3.,1.,1),2)
    c      = st(st(C))        # 12.
    c_a    = tg(st(C),da)     # 4.
    c_b    = st(tg(C,db))     # 3.
    c_ab   = tg(tg(C,db),da)  # 1.

OK, the unpacking is a wee bit cryptical: It helps to thing of multiple branchings down a tree structure, and `st` and `tg` return one branch from a node.  Note that `A*B` produced `((c,dc/da,1),(dc/db,ddc/dadb,1),2)` (as wanted), not `(a*b,a+b,???)`.  That is because `A` and `B` have different `id`s (that's where they come in...)

The other scenario in which nested `Fado`s appear is when a `Fado` is `stir`ed.

    # HIGHER ORDER DERIVATIVES
    (A,da) = stir(a,2) # ((a,da/da=1.,1),(da/da=1.,dda/dada=0.,1),2)
    # The line above is a shorthand for 
    # (A,dummy) = stir(a)
    # (A,da   ) = stir(A) # stir a Fado
    B      = foo(A)
    b      = tg(B)
    b_a    = st(tg(B,da))
    b_a    = tg(st(B),da))
    b_aa   = tg(tg(B,da),da)

Note that there are two ways here to retrieve `b_a`, because it is stored twice in the tree structure.  For higher order derivatives with respect to the same variable, this is an inefficiency inherent to Fado. How bad this gets for third and higher derivatives is left as an exercise to the reader.

So the function `stir` can be applied to a `Fado`, and everything will work out just fine.  This implies in particular that one can differentiate a function that itself happens to use differentiation, and this is completely transparent for both the programmer of the function (as long as the function is "weakly typed" enough to accept a `Fado`) or the programmer of the calling code (who will not have to wonder wether it implements analytic, automatic or finite differentiation).

    # TRANSPARENCY
    function foo(a)
        (A,da) = stir(a)
        B      = A*A
        return tg(B,da)
    end
    x      = 3.
    (X,dx) = stir(x)
    Y      = foo(X)
    y_x    = tg(Y,dx)

The system will also work correctly if for example `c` happens not to be a function of `a`.  This is where tags like `da` and `db` come in, they contain enough information to allow returning a "zero" of the right type and size. The scenario is not unrealistic: one can be differentiating a function which, over some "intervals" (`if`...) has constant values. 

    # FADO NOT FOOLED BY CONSTANTS
    a = 3.
    b = 4.
    (A,da) = stir(a) # (3.,1.,1)
    (B,db) = stir(b) # (4.,1.,2)
    C      = A*A     # (9.,6.,1)
    c      = st(st(C))        # 9.
    c_a    = tg(st(C),da)     # 6.
    c_b    = st(tg(C,db))     # 0.
    c_ab   = tg(tg(C,db),da)  # 0.

One can `stir` a `Vector`, `Matrix` or `Array` of any dimension.

    # DERIVATES OF ARRAYS AND/OR WRT ARRAYS
    a      = randn(2,3)
    (A,da) = stir(a)
    B      = foo(A)
    b      = st(B)    # size (4,5,6)
    b_a    = tg(B,da) # size (2,3,4,5,6)

Here, `b_a` is an `Array` which first dimensions are the dimensions of `a` and last dimensions the dimensions of `b`.  In the case of multiple derivatives, the dimensions of the variable `stir`ed first appear first in the derivative. The derivative of a `Float64` with respect to a `Float64` is a `Float64`.  Any other combination (`Array` wrt `Float64`, etc.) yields an `Array`.

It is possible to index into "arrays of Fados" XXXXXX

`Leibnitz.jl`
-------------
The module `Leibnitz` provides a (for the time - limited) set of functions for differentiation.  These are implemented using `Fado`, but the user of `Leibnitz` should not need to deal with `Fado`.

An objective for the future is to adapt `Leibnitz` so that it provides the same interface as other implementations of automatic differentiation in Julia.

`Leibnitz` implements the function `derive` with the following methods: `float` can be `Float64`, `Fado` or Àrray{Any}` thereof.

    y::float    = derive(       f::Function,x::float)   
    y::float    = derive(n::Int,f::Function,x::float) 
    g::Function = derive(       f::Function         )     
    g::Function = derive(n::Int,f::Function         )   
    
A synonym for `derive` is `∂`.  For example: `g = ∂(n,f)`.  Further, `f'`is a shorthand for `derive(f)`.  For example: `y = tan'(x)`
    
`Leibnitz` also implements the function `derivatives`, which returns an `Array{Any}` containing the `n` first derivatives of a function at a given point.

    z::Array{float} = derivatives(f::Function,x::float,         nx::Int=1          )
    z::Array{float} = derivatives(f::Function,x::float,y::float,nx::Int=1,ny::Int=1)
    
For the second syntax, for example, `z[ix+1,iy+1]` will contain the value of `d^(ix+iy)z/(dx^ix*dy^iy)` at `(x,y)`. So `z[1,1]` is simply `f(x,y)` and `z[1,2]` is `df/dy(x,y)`.

The function `Taylor(f::Function,x::Float64,nx::Int=1)` returns a function which, given a value `Dx` will compute the value at `x+Dx` of the Taylor development of `f` up to the `n`th derivative at `x`.  For the time being, only scalar valued function of ascalar are acceptable. Known issue: the Taylor development function itself cannot be `derive`d.
