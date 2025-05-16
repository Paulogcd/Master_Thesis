include("../../Environment/Final/sandbox_IV_simulation.jl")

using Plots
using OrdinalMultinomialModels

Plots.gr()
default(fontfamily = "Times")
default(linewidth=5)
default(titlefont=font("Times"))

# Plot the 5 plots of health transition probabilities,
# for a fixed level of temperature. 

# From Excellent: 

age_range = collect(1:100)

colors = palette(:RdYlGn_10,5,rev = true)

predicted_probabilities_excellent = Array{Any}(undef,100,5)

for t in 1:100
    predicted_probabilities_excellent[t,:] =
        health(Health_1 = 1, Temperature = 0.61, Age = t)
end

Plot_Excellent = Plots.plot(age_range,predicted_probabilities_excellent[:,1], label = "Excellent", color = colors[1])
Plots.plot!(age_range,predicted_probabilities_excellent[:,2], label = "Very Good",  color = colors[2])
Plots.plot!(age_range,predicted_probabilities_excellent[:,3], label = "Good",       color = colors[3])
Plots.plot!(age_range,predicted_probabilities_excellent[:,4], label = "Fair",       color = colors[4])
Plots.plot!(age_range,predicted_probabilities_excellent[:,5], label = "Poor",       color = colors[5])
Plots.plot!(xaxis = "Age", yaxis = "Probability", legend = :right) #, legend = false)
# Plots.plot!(title = "Transition probabilities from Excellent Health")

# Very Good:

predicted_probabilities_very_good = Array{Any}(undef,100,5)

for t in 1:100
    predicted_probabilities_very_good[t,:] =
        health(Health_1 = 2, Temperature = 0.61, Age = t)
end

Plot_Very_Good = Plots.plot(age_range,predicted_probabilities_very_good[:,1], label = "Excellent", color = colors[1])
Plots.plot!(age_range,predicted_probabilities_very_good[:,2], label = "Very Good", color = colors[2])
Plots.plot!(age_range,predicted_probabilities_very_good[:,3], label = "Good", color = colors[3])
Plots.plot!(age_range,predicted_probabilities_very_good[:,4], label = "Fair", color = colors[4])
Plots.plot!(age_range,predicted_probabilities_very_good[:,5], label = "Poor", color = colors[5])
Plots.plot!(xaxis = "Age", yaxis = "Probability", legend = :left)
# Plots.plot!(title = "Transition probabilities from Very Good Health")

# Good:

predicted_probabilities_good = Array{Any}(undef,100,5)

for t in 1:100
    predicted_probabilities_good[t,:] =
        health(Health_1 = 3, Temperature = 0.61, Age = t)
end

Plot_Good = Plots.plot(age_range,predicted_probabilities_good[:,1], label = "Excellent", color = colors[1])
Plots.plot!(age_range,predicted_probabilities_good[:,2], label = "Very Good", color = colors[2])
Plots.plot!(age_range,predicted_probabilities_good[:,3], label = "Good", color = colors[3])
Plots.plot!(age_range,predicted_probabilities_good[:,4], label = "Fair", color = colors[4])
Plots.plot!(age_range,predicted_probabilities_good[:,5], label = "Poor", color = colors[5])
Plots.plot!(xaxis = "Age", yaxis = "Probability", legend = :left)
# Plots.plot!(title = "Transition probabilities from Good Health")

# Fair:

predicted_probabilities_fair = Array{Any}(undef,100,5)

for t in 1:100
    predicted_probabilities_fair[t,:] =
        health(Health_1 = 4, Temperature = 0.61, Age = t)
end

Plot_Fair = Plots.plot(age_range,predicted_probabilities_fair[:,1], label = "Excellent", color = colors[1])
Plots.plot!(age_range,predicted_probabilities_fair[:,2], label = "Very Good", color = colors[2])
Plots.plot!(age_range,predicted_probabilities_fair[:,3], label = "Good", color = colors[3])
Plots.plot!(age_range,predicted_probabilities_fair[:,4], label = "Fair", color = colors[4])
Plots.plot!(age_range,predicted_probabilities_fair[:,5], label = "Poor", color = colors[5])
Plots.plot!(xaxis = "Age", yaxis = "Probability", legend = false)
# Plots.plot!(title = "Transition probabilities from Fair Health")

# Poor:

predicted_probabilities_poor = Array{Any}(undef,100,5)

for t in 1:100
    predicted_probabilities_poor[t,:] =
        health(Health_1 = 5, Temperature = 0.61, Age = t)
end

Plot_Poor = Plots.plot(age_range,predicted_probabilities_poor[:,1], label = "Excellent", color = colors[1])
Plots.plot!(age_range,predicted_probabilities_poor[:,2], label = "Very Good", color = colors[2])
Plots.plot!(age_range,predicted_probabilities_poor[:,3], label = "Good", color = colors[3])
Plots.plot!(age_range,predicted_probabilities_poor[:,4], label = "Fair", color = colors[4])
Plots.plot!(age_range,predicted_probabilities_poor[:,5], label = "Poor", color = colors[5])
Plots.plot!(xaxis = "Age", yaxis = "Probability", legend = :topleft)
# Plots.plot!(title = "Transition probabilities from Poor Health")


health_transition_plots = [Plot_Excellent,
    Plot_Very_Good,
    Plot_Good, 
    Plot_Fair, 
    Plot_Poor]

for (i,plot) in enumerate(health_transition_plots)
    
    Plots.plot(plot, 
    
    size = (2400, 1600),
    legendfontsize = 40,
    guidefontsize = 40,
    tickfontsize = 40,

    bottom_margin = 100Plots.px,
    top_margin = 100Plots.px,
    left_margin = 100Plots.px)

    savefig("working_elements/Draft/output/health_transition_$i.png")
    
end