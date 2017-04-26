D = "data_dimless.txt"
a = 2; t = 0.02; p = 1
f(x) = a-(a-1)*exp(-t*x**p)
fit f(x) D u 1:2 via a, t
plot D u 1:2:3 w e, f(x), a
