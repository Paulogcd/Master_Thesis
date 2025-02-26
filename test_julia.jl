size = 200
periods = 100
future_results = Array{Vector}(undef,1+size)

using Distributions
begin
	"""
	Th function `population_simulation(N)` runs a simulation for `N` individuals. 
	
	It returns a 3 dimensions tuple with, for each individual : 

	- The weather history
	- Their age of death 
	- Their living status history 
	- Their health history 
	"""
	function population_simulation(size::Number,periods::Number)::Tuple

		future_individual_results = Array{Vector}(undef,1+size)
		# Initialise a common weather history for the population
		weather_history = rand(Normal(0,5), periods)
        collective_results = []
		
		# Initialize arrays with proper dimensions

		# all death age of individuals
		AOD = age_of_death = Array{Number}(undef, size)
		# all living history of individuals
		LH = living_history = Array{Vector{Float64}}(undef, size)
		# all health history of individuals
		HH = health_history = Array{Vector{String}}(undef, size)
		
		# For each individual
		for i in 1:size
			# individual_results[i] = Array{Vector}
			individual_results = []
			individual_living_history = zeros(100)
			individual_health_history = Vector{String}(undef,100)
	
			for t in 1:periods # For each period 

			   # At first period, the individuals are born in good health
				if t == 1
				   global individual_past_health = "g" # Initial good health
				end
		    
			    # The age : 
			    age = t
			    
			    # The weather comes from the weather history
			    weather_t = weather_history[t]
			    
			    # The health status :
				# probability of being in good health: 
				individual_pgh = probability_good_health(past_health,weather_t)
				# Health status draw:
				individual_health_t = "g" # health(individual_pgh)
	
				individual_health_history[t] = individual_health_t
				individual_past_health = individual_health_t
	
			    # The living status : 
			    # Probability:
				individual_pd = ζ(age,weather_t,individual_health_t)
			    # realisation : 
				individual_living_status = 0 # rand(Binomial(1,1-individual_pd))
			    # Into its history :
				global individual_living_history[t] = individual_living_status
	
				# When death comes : 
				if individual_living_status == 0
					# print("Agent died at ", t)
					push!(individual_results,age)
				    push!(individual_results,individual_living_history)
					push!(individual_results, individual_health_history)
				break
				end
			end # End of loop over periods
            push!(collective_results,individual_results)
			# We go to the next individual
		end # End of loop over individuals
		
		println("Life expectancy in this population: ", mean(AOD))
		population_results = (;age_of_death,living_history,health_history)
		return(population_results)
	end
end