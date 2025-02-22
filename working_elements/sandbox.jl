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

begin 
	# Save the plot as interactive with plotly for the website:
	
	# Create a Plotly plot
	p = Plotly.Plot(
		[Plotly.scatter(x=1:100, y=pop_choices_full_normal, name="Perfectly normal weather"),
		Plotly.scatter(x=1:100, y=pop_choices_full_deviation, name="Totally deviating weather")],
		Layout(xaxis_title="Period", yaxis_title="Aggregated utility"))
	
	# Save the plot as an HTML file
	open("./Framework_mode_1plot1.html", "w") do io
		PlotlyBase.to_html(io, p)
	end
end