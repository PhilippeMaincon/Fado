# Tests and demos
using fado,Base.Test

(a,da)      = stir([1.,2.])
(b,db)      = stir([3.,4.])
c           = dot(a,b)
(c,c_b)     = st_tg(c,db)
(c,c_a)     = st_tg(c,da)
(c_b,c_ba)  = st_tg(c_b,da)
@test c == 11.
@test c_a == [3.,4.]
@test c_b == [1.,2.]
@test c_ba == eye(2)

(a,da)      = stir([1.,2.])
b           = a+2.
c           = dot(a,b)
(c,c_a)     = st_tg(c,da)
@test c   == 11.
@test c_a == [4.,6.]

(a,da)      = stir([1.,2.])
m           = [1. 2.;3. 4.]
b           = m*a
(b,b_a)     = st_tg(b,da)
@test b   == [5.,11.]
@test b_a == m'

(a1,da1)    = stir([1.,2.])
(a,da)      = stir(a1)
m           = [1. 2.;3. 4.]
b           = 1/2*dot(a,m*a)
(bx,bdx)    = st_tg(b,da)
(bxx,bxdx)  = st_tg(bx,da1)
(bdxx,bdxdx)= st_tg(bdx,da1)
@test bxx   == 13.5
@test bxdx  ==[6.,10.5]
@test bdxx  ==[6.,10.5]
@test bdxdx ==[1. 2.5;2.5 4.]


