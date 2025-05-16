
include("../../Environment/Final/sandbox_IV_simulation.jl")

using Plots
default(fontfamily = "Times")

age_range = collect(range(start = 1, stop = 100))

N = 10_000
T = 100

function health_history(;N::Number,T::Number,weather_path::Vector)
    
    health_history                  = Array{Float64}(undef, N,T)
    survival_probability_history    = Array{Float64}(undef, N,T)

    health_history[:,1] .= 1

    Threads.@threads for i in 1:N
        survival_probability_history[i,1] =
            instantaneous_survival(Age = 1,
                Health = health_history[i,1], 
                Temperature = weather_path[1])

        for t in 2:T
            probabilities_health =
                health(Health_1 = health_history[i,t-1],
                Temperature =  weather_path[t],
                Age = t)
            health_history[i,t] = sample(1:5, Weights(probabilities_health))
            survival_probability_history[i,t] =
                instantaneous_survival(Age = t,
                    Health = health_history[i,t], 
                    Temperature = weather_path[t])
            end
            
        end
    return(;survival_probability_history,health_history)
end

weather_path_good           = collect(range(start = 0.61, stop = 2, length = 100))
weather_path_intermediate   = collect(range(start = 0.61, stop = 3, length = 100))
weather_path_bad            = collect(range(start = 0.61, stop = 4, length = 100))

@time sim_good              = health_history(N = N,T = T, weather_path = weather_path_good)
@time sim_intermediate      = health_history(N = N,T = T, weather_path = weather_path_intermediate)
@time sim_bad               = health_history(N = N,T = T, weather_path = weather_path_bad)

average_health_1            = mean.(sim_good.health_history[:,t] for t in 1:T)
average_health_2            = mean.(sim_intermediate.health_history[:,t] for t in 1:T)
average_health_3            = mean.(sim_bad.health_history[:,t] for t in 1:T)

Plots.plot(age_range, average_health_1, label = "Optimistic scenario", linewidth = 5, color = "green")
Plots.plot!(age_range, average_health_2, label = "Intermediate scenario", linewidth = 5, color = "orange")
Plots.plot!(age_range, average_health_3, label = "Pessimistic scenario", linewidth = 5, color = "red")
Plots.plot!(xaxis = "Year", yaxis = "Average Health")

Plots.plot!(
        size = (2400, 1600),
        legendfontsize = 25,
        guidefontsize = 45,
        tickfontsize = 30,

        bottom_margin = 100Plots.px,
        top_margin = 100Plots.px,
        left_margin = 100Plots.px, 
        color = ["green", "blue", "orange","red"])

savefig("working_elements/Draft/output/average_health.png")

