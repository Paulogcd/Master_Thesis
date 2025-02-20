using Pkg
Pkg.activate(".")
using Plots

plotly()  # Use GR backend for 3D plotting

# Create coordinate grid
x = range(1, 100)
y = range(1, 100)
# y = range(-2, 2, length=50)
X = repeat(reshape(x, :, 1), 1, length(y))
Y = repeat(reshape(y, 1, :), length(x), 1)

# Create flat plane at z=1
Z = ones(size(X))  # Creates matrix of 1's matching X,Y dimensions

Z

# Plot the surface
surface(X, Y, Z, alpha=0.5, label="Flat Plane")

# Customize the plot
title!("Flat Plane at z = 1")
xlabel!("X")
ylabel!("Y")
zlabel!("Z")