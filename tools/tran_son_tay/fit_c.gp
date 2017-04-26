D = "data_dimless.txt"
a = 2; t = 0.02; p = 1
f(x) = a+(1-a)*exp(-t*x**p)
fit f(x) D u 1:4 via a, t
plot D u 1:4:5 w e, f(x), a
