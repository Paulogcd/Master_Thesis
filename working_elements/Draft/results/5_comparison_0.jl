
# Environment: estimates, health, and probabilities
include("../../Environment/Final/sandbox_IV_simulation.jl")

using NamedArrays

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


# Utility: 
function utility(;c::Float64,
                    l::Float64,
                    z::Float64,
                    w::Float64,
                    h::Float64,
                    ρ = 1.50::Float64,
                    φ = 2.00::Float64)::Float64
    return 100 + ( ((abs(c))^(1-ρ)) / (1-ρ) ) - ξ(w=w,h=h) * ( ((abs(l))^(1+φ)) / (1+φ) )::Float64
end

function budget_surplus(;c::Float64,
        l::Float64,
        sprime::Float64,
        s::Float64,
        z::Float64,
        r::Float64)::Float64
    if r == Inf
        return -Inf
    else
        return (l*z + s*(1+r) - c - sprime)::Float64
    end
end

function ξ(;w::Float64,h::Float64)::Float64
    return 1.00 # ((1 + abs(w)) * (1+1(h=="bad")))::Float64
end

# Performance function: 

function performance_metrics(TIMED_RESULT::NamedTuple)
    Value = TIMED_RESULT.value
    Error   = mean(TIMED_RESULT.value.budget_balance) # in average of budget clearing
    Time    = TIMED_RESULT.time # in seconds
    Memory  = TIMED_RESULT.bytes / 1_000_000 # in Mb
    return (;Value,Error,Time,Memory)
end

# Lifetime income: 
function compute_lifetime_income(optimal_choices::NamedArray,
                                  s_range::AbstractRange,
                                  z::AbstractVector,
                                  survival::AbstractVector,
                                  r::AbstractVector,
                                  s0::Float64 = 0.0)

    T = length(z)
    current_s = s0
    lifetime_income = 0.0

    cumulative_discount = 1.0
    cumulative_survival = 1.0

    for t in 1:T
        # Find index of closest match in the s_range
        s_idx = findmin(abs.(s_range .- current_s))[2]

        # Read optimal labor and next-period savings
        l_t = optimal_choices[t, s_idx, "l"]
        s_next = optimal_choices[t, s_idx, "sprime"]

        # Compute labor income and add to lifetime income
        income_t = z[t] * l_t
        println("t=$t | s_idx=$s_idx | s=$current_s | l=$l_t | income=$income_t | cumulative_income=$(lifetime_income)")
        lifetime_income += cumulative_discount * survival[t] * income_t

        # Update state: savings evolve according to the optimal policy
        current_s = s_next

        # Update survival and discount factors
        cumulative_survival *= survival[t]
        cumulative_discount /= (1 + r[t])
    end

    return lifetime_income
end

# Simulations: 

N = 10_000
T = 100

weather_path_boomer             = collect(range(start = 0.01, stop = 1.5, length = 100))
weather_path_good               = collect(range(start = 0.61, stop = 2, length = 100))
weather_path_intermediate       = collect(range(start = 0.61, stop = 3, length = 100))
weather_path_bad                = collect(range(start = 0.61, stop = 4, length = 100))

@time sim_boomer                = health_history(N = N,T = T, weather_path = weather_path_boomer)
@time sim_good                  = health_history(N = N,T = T, weather_path = weather_path_good)
@time sim_intermediate          = health_history(N = N,T = T, weather_path = weather_path_intermediate)
@time sim_bad                   = health_history(N = N,T = T, weather_path = weather_path_bad)

average_health_boomer             = mean.(sim_boomer.health_history[:,t] for t in 1:T)
average_health_good             = mean.(sim_good.health_history[:,t] for t in 1:T)
average_health_intermediate     = mean.(sim_intermediate.health_history[:,t] for t in 1:T)
average_health_bad              = mean.(sim_bad.health_history[:,t] for t in 1:T)

average_proba_boomer              = mean.(sim_boomer.survival_probability_history[:,t] for t in 1:T)
average_proba_good              = mean.(sim_good.survival_probability_history[:,t] for t in 1:T)
average_proba_intermediate      = mean.(sim_intermediate.survival_probability_history[:,t] for t in 1:T)
average_proba_bad               = mean.(sim_bad.survival_probability_history[:,t] for t in 1:T)

sim_boomer              = nothing 
sim_good                = nothing 
sim_intermediate        = nothing 
sim_bad                 = nothing

# Range parameters: 

s_range_2 			= 0.00:0.05:2.00 # 0.00:100:5_000 # 0.00:0.10:2.00 # 20
sprime_range_2 		= s_range_2 # 20 
consumption_range   = 0.00:0.10:3.00 # 0.00:100:4_000 # 0.00:0.10:3.00 # 30
small_r             = (fill(0.0178583,T))
labor_range         = 0.00:0.01:1.1 # 0.00:40:2_080 # 0.00:0.10:2.00 # 20
z = ones(T)

# β = 0.99
β = 1/(1+0.0178583)

println("5_comparison_0.jl DONE")