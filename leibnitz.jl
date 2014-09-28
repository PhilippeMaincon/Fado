module leibnitz
export derive,∂,derivatives,Taylor
using  fado

function derive(fun::Function,x)
   (x,dx) = stir(x)
   return tg(fun(x),dx)
end
function derive(n::Int,f::Function,x)
   dx         = Nothing
   for i      = 1:n
      (x,dx)  = stir(x)
   end
   y          = f(x)
   for i      = 1:n
      y       = tg(y,dx)
   end
   return y
end
derive(       fun::Function) = x->derive(  fun,x)
derive(n::Int,fun::Function) = x->derive(n,fun,x)
∂                            = derive
ctranspose(fun::Function)    = derive(fun)

function derivatives(f::Function,x,nx::Int=1)
   (x,dx)     = stir(x,nx)
   z          = f(x)
   r          = Array(Any,nx+1)
   for ix     = 0:nx
      t = z
      for jx= 1:ix    t=tg(t,dx) end
      for jx= ix+1:nx t=st(t   ) end
      r[ix+1] = t
   end
   return r
end
function derivatives(f::Function,x,y,nx::Int=1,ny::Int=1)
   (x,dx)     = stir(x,nx)
   (y,dy)     = stir(y,ny)
   z          = f(x,y)
   r          = Array(Any,nx+1,ny+1)
   for ix     = 0:nx
      for iy  = 0:ny
         t = z
         for jy= 1:iy    t=tg(t,dy) end
         for jy= iy+1:ny t=st(t   ) end
         for jx= 1:ix    t=tg(t,dx) end
         for jx= ix+1:nx t=st(t   ) end
         r[ix+1,iy+1] = t
      end
   end
   return r
end

function Taylor(f::Function,x::Float64,nx::Int=1) #todo generalise to x::Array
   r = derivatives(f,x,nx)
   fac = 1
   for i = 1:nx
      fac *= i
      r[i+1]/=fac
   end
   return function(Dx) # todo error when differentiating this function
             o = r[nx+1]
             for i = nx:-1:1
                o = Dx*o+r[i]
             end
             return o
          end
end
end # module leibnitz
