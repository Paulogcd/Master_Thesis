ρ = 1.5
φ = 2
ξ = 1
z = 1

oc(l) = (z/(ξ*l^φ))^(1/ρ)

ol(c) = (c^ρ)^(1/φ)

Plots.plot(0:0.1:10,oc)
Plots.plot!(xaxis = "Labor", yaxis = "Consumption")

Plots.plot(0:0.1:10,ol)
Plots.plot!(xaxis = "Consumption", yaxis = "Labor")

