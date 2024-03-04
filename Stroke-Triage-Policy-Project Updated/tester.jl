using Plots

x = randn(10^3)
p1 = histogram(x)

y = rand(10^3)
p2 = histogram(y)

z = rand(10^3)
p3 = histogram(z)

b = rand(10^3)
p4 = histogram(b)

plot(p1, p2, p3, p4, layout=(2, 2), legend=false)


