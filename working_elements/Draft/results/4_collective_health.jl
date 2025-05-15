
include("../../Environment/Final/sandbox_IV_simulation.jl")

function individual_probability_simulation(;T::Integer,
                                temperature_path::Vector{Float64},
                                GDP_path::Vector{Float64})
    
    # Initialization: 
    probability_history = Vector{Float64}(undef,T)
    health_history 		= Vector{Float64}(undef,T)

    individual_past_health 				= 1 	# Excellent health
    cumulative_survival_prob 			= 1 	# Birth probability
    
    for t in 1:T
        # Age : 
        age = t
            
        # The weather comes from the weather history
        weather_t = temperature_path[t]
            
        # Health status :
            # probability of being in good health: 
            individual_pgh = health(Age 		= age, 
                                    Health_1 	= individual_past_health,
                                    Temperature = weather_t)::Vector{Float64}
        
            # Health status draw:
            individual_health_t = 	
                sample(1:5,Weights(individual_pgh))

            # We add it to the history
            # individual_health_history[t] = individual_health_t
            # The current health becomes the past one for next period
            individual_past_health = individual_health_t
            health_history[t] = individual_past_health

        # Living status : 
        
        annual_survival = survival(Age 			= age,
                                    Health 		= individual_health_t, 
                                    Temperature 	= weather_t,
                                    GDP 			= GDP_path[t])::Float64
    
        # cumulative_survival_prob = 
        #    cumulative_survival_prob * annual_survival
            
        cumulative_survival_prob = annual_survival

        probability_history[t] = cumulative_survival_prob
    end
    return (;probability_history,health_history)
end

individual_probability_simulation(T = 10, 
    temperature_path = zeros(10), 
    GDP_path = fill(2000.0,10))

function collective_probability_simulation(;N::Integer, 
                                            T::Integer,
                                            temperature_path::Vector{Float64},
                                            GDP_path::Vector{Float64})

    collective_probability = Vector{Array}(undef,N)
    collective_health = Vector{Array}(undef,N)
    
    Threads.@threads for n in 1:N
        
        tmp = individual_probability_simulation(T = T, 
                                            temperature_path = temperature_path,
                                            GDP_path = GDP_path)
        
        collective_probability[n] 	= tmp[1]
        collective_health[n] 		= tmp[2]
    end
    
    return (;collective_probability,collective_health)
end


cp_benchmark_b = collective_probability_simulation(N = 20_000,
                                T = nperiods,
                                temperature_path = bad_scenario,
                                GDP_path = fill(GDP_2018,nperiods))

cp_benchmark_i = collective_probability_simulation(N = 20_000,
                                T = nperiods,
                                temperature_path = intermediate_scenario,
                                GDP_path = fill(GDP_2018,nperiods))

cp_benchmark_g = collective_probability_simulation(N = 20_000,
                                T = nperiods,
                                temperature_path = good_scenario,
                                GDP_path = fill(GDP_2018,nperiods))
