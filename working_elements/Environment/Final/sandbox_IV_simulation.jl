using Base.Threads

""" Gives the HP predicted"""
function HP(;Age, 
            Temperature)
        
    data = DataFrame(Age = Age, 
                    Temperature = Temperature)

    result =
        predict(REG1,
        data)

    result = result

    # result < 0 ? result = 0 : result = result

    return result
end

HP(Age = 50, Temperature = 0.61)[1]

""" Gives the survival probability """
function instantaneous_survival(;Age::Int64,
            Health::Int64,
            Temperature::Float64)::Float64

    HP_predicted = HP(Age = Age, Temperature = Temperature)

    data = DataFrame(Age = Age, 
            Health = Health, 
            HP_predicted = HP_predicted,
            av_annual_t = Temperature)

    result = GLM.predict(REG3, data)

    result = result[1]
    
    return result
end

instantaneous_survival(Age = 80,
    Health = 5, 
    Temperature = 0.61)

""" Gives the health transition probabilities for a given health state and predicted HP."""
function health(;Health_1, 
                Age, 
                Temperature)

    HP_predicted = HP(Age = Age, Temperature = Temperature)

    data = DataFrame(
        Health_1   = Health_1,
        HP_predicted = HP_predicted, 
        Age = Age)
    
    probabilities = OrdinalMultinomialModels.predict(REG2, data, kind = :probs)
    # probabilities = permutedims(probabilities)
    # rename!(probabilities, ["P(H=1)","P(H=2)","P(H=3)","P(H=4)","P(H=5)"])
    probabilities = Matrix(probabilities)[1, :]
    probabilities = convert(Vector{Float64}, probabilities)
    # probabilities .= Float64.(probabilities) 
    return probabilities
end
		
a = health(Health_1 = 1, Temperature = 0.61, Age = 40)

# Population: 

function population_simulation(;N::Int64,
                                T::Int64,
                                weather_history::Vector{Float64})::NamedTuple

    # Initialisation:
    collective_age 						= []
    collective_living_history 			= []
    collective_health_history 			= []
    collective_probability_history 		= []
    
    for i in 1:N # For each individual
        
        # Initialisation of individual results:
        individual_living_history 			= zeros(T)
        individual_health_history 			= zeros(T)
        individual_probability_history 		= zeros(T) # Setting it to 0 makes r ≠ Inf

        individual_past_health 				= 1 	# Excellent health
        cumulative_survival_prob 			= 1 	# Birth probability

        for t in 1:T # For each period 
        
            # Age : 
            age = t
            
            # The weather comes from the weather history
            weather_t = weather_history[t]
            
            # Health status :
                # probability of being in good health: 
                individual_pgh = health(Age 		= age, 
                                        Health_1 	= individual_past_health,
                                        Temperature = weather_t)
            
                # Health status draw:
                individual_health_t = 	
                    sample(1:5,Weights(individual_pgh))

                # We add it to the history
                individual_health_history[t] = individual_health_t
                # The current health becomes the past one for next period
                individual_past_health = individual_health_t

            # Living status : 
            
                annual_survival = instantaneous_survival(Age 			    = age,
                                            Health 		    = individual_health_t, 
                                            Temperature 	= weather_t)
            
                cumulative_survival_prob = annual_survival
                    # cumulative_survival_prob * annual_survival

                individual_probability_history[t] = cumulative_survival_prob
            
                # Realisation : 
                individual_living_status = 
                    rand(Binomial(1,cumulative_survival_prob))

                # Into its history :
                individual_living_history[t] = individual_living_status

            # When death comes : 
            if individual_living_status == 0
                push!(collective_age, age)
                push!(collective_living_history, individual_living_history)
                push!(collective_health_history, individual_health_history)
                push!(collective_probability_history, individual_probability_history)
                break
            end
            
        end # End of loop over periods
        
    end # End of loop over individuals

    life_expectancy = mean(collective_age)

    results = (;weather_history,
                collective_age,
                collective_living_history,
                collective_health_history, 
                collective_probability_history, 
                life_expectancy)
    println("Life expectancy in this population: ", life_expectancy)
    
    return(results)
end

population_simulation(N = 2000,
    T = 100, 
    weather_history = zeros(100)).life_expectancy
