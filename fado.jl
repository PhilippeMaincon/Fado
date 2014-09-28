module    fado
importall Base
export    Fado,FadoTag,stir,tg,st,st_tg,test

function forall(t::Type,s::Tuple,f::Function)
   b       = Array(t,s...)
   for i   = 1:prod(s)
      b[i] = f(i)::t
   end
   return b
end

function all(a::Array,f::Function)
   for ai in a
      if !f(ai)
         return false
      end
   end
   return true
end

# in expression f replace symbol a with symbol b
function repsymb(f,a,b)
   if f == a
      e = deepcopy(b)
   else
      e = deepcopy(f)
      if typeof(e) == Expr
         for i = 1:length(e.args)
            e.args[i] = repsymb(e.args[i],a,b)
         end
      end
   end
   return e
end

immutable Fado
   _id :: Int64
   _st :: Any
   _tg :: Any
   Fado(id,st,tg::Array{Float64}) = all(tg,ia->ia==0.) ? st : new(id,st,tg)
   Fado(id,st,tg                ) = new(id,st,tg)
end
immutable FadoTagScalar
   _id       :: Int64
end
immutable FadoTagArray
   _id       :: Int64
   _size     :: Tuple
end
typealias Nu64          Union(Float64,Int64)                   # Array{Any} operations initialise summations with a Int64 zero
typealias aNu64{d}      Union(Array{Float64,d},Array{Int64,d}) # so I need to cope with that
typealias FadoTag Union(FadoTagScalar,FadoTagArray)

#### create Fado
global idcounter  = 0

function L(a::Float64   ,s,i)
   b    = zeros(Float64,s...)
   b[i] = 1.0
   return b
end
L(a::Float64       ) = 1.0
L(a::Fado,s,i) = L(a._st,s,i)
L(a::Fado    ) = L(a._st    )

function stir(a::Array)
   global idcounter += 1
   s    = size(a)
   dual = forall(Any,s,i->Fado(idcounter,a[i],L(a[i],s,i)))
   tag  = FadoTagArray(idcounter,s)
   return (dual,tag)
end
function stir(a::Union(Float64,Fado))
   global idcounter += 1
   dual = Fado(idcounter,a,L(a))
   tag  = FadoTagScalar(idcounter)
   return (dual,tag)
end
function stir(a,n::Int)
   tag = Nothing
   for i      = 1:n
      (a,tag) = stir(a)
   end
   return (a,tag)
end

#### operate on Fado
macro Op1(OP,A)
   Aa  = repsymb(A,:a,:(a._st))
   return quote
      $(esc(OP))(a::Fado)  = Fado(a._id,$(esc(OP))(a._st),$Aa.*a._tg)
      $(esc(OP))(a::Array)       = forall(Any,size(a),i->$(esc(OP))(a[i])) # method is necessary unless Fado<:Number
   end
end
macro Op2(OP,A,B)
   Aa  = repsymb(A ,:a,:(a._st))
   Aab = repsymb(Aa,:b,:(b._st))
   Bb  = repsymb(B ,:b,:(b._st))
   Bab = repsymb(Bb,:a,:(a._st))
   return quote
      function $(esc(OP))(a::Fado,b::Fado)
         if     a._id == b._id return    Fado(a._id,$(esc(OP))(a._st,b._st),$Aab.*a._tg+$Bab.*b._tg)
         elseif a._id <  b._id return    Fado(b._id,$(esc(OP))(a    ,b._st),            $Bb .*b._tg)
         else                  return    Fado(a._id,$(esc(OP))(a._st,b    ),$Aa.*a._tg             )
         end
      end
      $(esc(OP))(a::Nu64,b::Fado)  = Fado(b._id,$(esc(OP))(a    ,b._st ),            $Bb .*b._tg)
      $(esc(OP))(a::Fado,b::Nu64)  = Fado(a._id,$(esc(OP))(a._st,b     ),$Aa .*a._tg            )
      $(esc(OP))(a::Array,b::Fado) = forall(Any,size(a),i->$(esc(OP))(a[i],b   ))
      $(esc(OP))(a::Fado,b::Array) = forall(Any,size(b),i->$(esc(OP))(a   ,b[i]))
   end
end
for OP in (:(>),:(<),:(==),:(>=),:(<=),:(!=)) #...
   eval(quote
      $OP(a::Fado,b::Fado) = $OP(a._st,b._st)
      $OP(a::Nu64      ,b::Fado) = $OP(a    ,b._st)
      $OP(a::Fado,b::Nu64      ) = $OP(a._st,b    )
   end)
end
#    OP   dOP(a)/da
@Op1(+    ,1.           )
@Op1(-    ,-1.          )
@Op1(abs  ,a==0.0? 0. : (a>0.0? 1.: -1.))
@Op1(floor,a==floor(a)? Inf : (a>0.0? 1.: -1.))
@Op1(conj ,1            )
# Code below cloned from John Myles White's Calculus.jl
# TODO I should have imported his code and created a macro to generate the @Op1 calls below.
@Op1(sqrt,        1 / 2 / sqrt(a)                   )
@Op1(cbrt,        1 / 3 / cbrt(a)^2                 )
@Op1(abs2,        2 * a                             )
@Op1(inv,        -1 * abs2(inv(a))                  )
@Op1(log,         1 / a                             )
@Op1(log10,       1 / a / log(10)                   )
@Op1(log2,        1 / a / log(2)                    )
@Op1(log1p,       1 / (a + 1)                       )
@Op1(exp,         exp(a)                            )
@Op1(exp2,        log(2) * exp2(a)                  )
@Op1(expm1,       exp(a)                            )
@Op1(sin,         cos(a)                            )
@Op1(cos,        -sin(a)                            )
@Op1(tan,         (1 + tan(a)^2)                    )
@Op1(sec,         sec(a) * tan(a)                   )
@Op1(csc,        -csc(a) * cot(a)                   )
@Op1(cot,        -(1 + cot(a)^2)                    )
@Op1(sind,        pi / 180 * cosd(a)                )
@Op1(cosd,       -pi / 180 * sind(a)                )
@Op1(tand,        pi / 180 * (1 + tand(a)^2)        )
@Op1(secd,        pi / 180 * secd(a) * tand(a)      )
@Op1(cscd,       -pi / 180 * cscd(a) * cotd(a)      )
@Op1(cotd,       -pi / 180 * (1 + cotd(a)^2)        )
@Op1(asin,        1 / sqrt(1 - a^2)                 )
@Op1(acos,       -1 / sqrt(1 - a^2)                 )
@Op1(atan,        1 / (1 + a^2)                     )
@Op1(asec,        1 / abs(a) / sqrt(a^2 - 1)        )
@Op1(acsc,       -1 / abs(a) / sqrt(a^2 - 1)        )
@Op1(acot,       -1 / (1 + a^2)                     )
@Op1(asind,       180 / pi / sqrt(1 - a^2)          )
@Op1(acosd,      -180 / pi / sqrt(1 - a^2)          )
@Op1(atand,       180 / pi / (1 + a^2)              )
@Op1(asecd,       180 / pi / abs(a) / sqrt(a^2 - 1) )
@Op1(acscd,      -180 / pi / abs(a) / sqrt(a^2 - 1) )
@Op1(acotd,      -180 / pi / (1 + a^2)              )
@Op1(sinh,        cosh(a)                           )
@Op1(cosh,        sinh(a)                           )
@Op1(tanh,        sech(a)^2                         )
@Op1(sech,       -tanh(a) * sech(a)                 )
@Op1(csch,       -coth(a) * csch(a)                 )
@Op1(coth,       -csch(a)^2                         )
@Op1(asinh,       1 / sqrt(a^2 + 1)                 )
@Op1(acosh,       1 / sqrt(a^2 - 1)                 )
@Op1(atanh,       1 / (1 - a^2)                     )
@Op1(asech,      -1 / a / sqrt(1 - a^2)             )
@Op1(acsch,      -1 / abs(a) / sqrt(1 + a^2)        )
@Op1(acoth,       1 / (1 - a^2)                     )
@Op1(erf,         2 * exp(-square(a)) / sqrt(pi)    )
@Op1(erfc,       -2 * exp(-square(a)) / sqrt(pi)    )
@Op1(erfi,        2 * exp(square(a)) / sqrt(pi)     )
@Op1(gamma,       digamma(a) * gamma(a)             )
@Op1(lgamma,      digamma(a)                        )
@Op1(airy,        airyprime(a)                      )  # note: only covers the 1-arg version
@Op1(airyprime,   airy(2, a)                        )
@Op1(airyai,      airyaiprime(a)                    )
@Op1(airybi,      airybiprime(a)                    )
@Op1(airyaiprime, a * airyai(a)                     )
@Op1(airybiprime, a * airybi(a)                     )
@Op1(besselj0,   -besselj1(a)                       )
@Op1(besselj1,    (besselj0(a) - besselj(2, a)) / 2 )
@Op1(bessely0,   -bessely1(a)                       )
@Op1(bessely1,    (bessely0(a) - bessely(2, a)) / 2 )
#    OP   dOP(a,b)/da       dOP(a,b)/db
@Op2(+    ,1.               ,1.          )
@Op2(.+   ,1.               ,1.          )
@Op2(-    ,1.               ,-1.         )
@Op2(.-   ,1.               ,-1.         )
@Op2(*    ,b                ,a           )
@Op2(.*   ,b                ,a           )
@Op2(/    ,1./b             ,-a./b.^2    )
@Op2(./   ,1./b             ,-a./b.^2    )
@Op2(^    ,b.*a.^(b-1.)     ,log(a).*a.^b)
@Op2(.^   ,b.*a.^(b-1.)     ,log(a).*a.^b)
# ad lib.

#### retrieve from Fado
test(a::Array     ,tag::FadoTag)   = all([test(ai,tag) for ai in a])
test(a::Float64   ,tag::FadoTag)   = true
test(a::Fado,tag::FadoTag)   = a._id <= tag._id

st(a::Float64)        = a
st(a::Fado)     = a._st
st(a::Array{Float64}) = a
function st(a::Array)
   out =  all(a,ai->isa(st(ai),Float64))? Float64 : Any
   return forall(out,size(a),i->st(a[i]))
end

tg(a::Float64       ,tag::FadoTagScalar) = 0.
tg(a::Float64       ,tag::FadoTagArray)  = zeros(tag._size...)
tg(a::Array{Float64},tag::FadoTagScalar) = zeros(size(a)...)
tg(a::Array{Float64},tag::FadoTagArray)  = zeros(tag._size...,size(a)...)
tg(a::Fado    ,tag::FadoTag)       = a._tg
function tg(a::Array,tag::FadoTagArray)
   out        = all(a,ai->isa(tg(ai,tag),Array{Float64}))? Float64 : Any
   lenx       = prod(tag._size)
   dx         = Array(out,lenx,length(a))
   for i      = 1:length(a)
      dx[:,i] = reshape(tg(a[i],tag),lenx)
   end
   return reshape(dx,tag._size...,size(a)...)
end
function tg(a::Array,tag::FadoTagScalar)
   out        = all(a,ai->isa(tg(ai,tag),Array{Float64}))? Float64 : Any
   return forall(out,size(a),i->tg(a[i],tag))
end

st_tg(a,tag::FadoTag) = test(a,tag) ? (st(a),tg(a,tag)) : error("Unable to extract derivative")

#### show Duals
function show(io::IO,a::Fado)
   id = a._id
   st = a._st
   tg = a._tg
   if isa(tg,Array)
         println(io,"Fado(id=$id,st=$st,tg=")
         show(io,tg)
         println(io,")")
      else
         print(io,"Fado(id=$id,st=$st,tg=$tg)")
   end
end

end # module fado
