begin 
	using CSV
	using DataFrames
	# using PlutoUI
	using CategoricalArrays
	using MLJ
	using MLJModels
	using MLJLinearModels
	using Plots
	using GLM
	using StatsBase
	# TableOfContents(title = "Empirical model 1")
end

begin 
	""" 
	The survival function returns the probability of survival given the Age, current health, annual temperature, and GDP. 
		
		function survival(;Age::Int64,
			Health::Int64,
			Temperature::Float64,
			GDP::Float64)::Float64
	
	"""
	function survival(;Age::Int64, Health::Int64, Temperature::Float64, GDP::Float64)::Float64

		# For reminder: 
		# model_health_age_temperature_gdp =
		# GLM.glm(@formula(Status ~
		# 				 	Age +
		# 					Health +
		# 					av_annual_t + 
		# 					GDP
		# 					),
		# 		DF, Bernoulli(), LogitLink())

		data = DataFrame(Age = Age, 
						Health = Health, 
						av_annual_t = Temperature, 
						GDP = GDP)

		result = GLM.predict(model_health_age_temperature_gdp,data)

		# result = Float64.(result)

		result = result[1]
		
		return result
	end
end

begin 
	"""
	The clean_health function takes out rows with Health values that correspond to NAs in the HRS data.
	"""
    clean_health = function(DF::DataFrame,COLUMN::AbstractString)
        DF = DF[DF[:,COLUMN] .!= -8, :]
        DF = DF[DF[:,COLUMN] .!= 8, :]
        DF = DF[DF[:,COLUMN] .!= 9, :]
        return DF
    end
end

df

begin 

	# We define some year-specific dataframes: 
	
	df_2022 = df[df[:,:Year] .== 2022,:]
	# rename!(df_2022, Dict("Health" => "Health_2022"))
	
	df_2020 = df[df[:,:Year] .== 2020,:]
	# rename!(df_2020, Dict("Health" => "Health_2020"))
	
	df_2018 = df[df[:,:Year] .== 2018,:]
	# rename!(df_2018, Dict("Health" => "Health_2018"))
	
	df_2016 = df[df[:,:Year] .== 2016,:]
	# rename!(df_2016, Dict("Health" => "Health_2016"))

	nothing

end

begin 
	
	# We group data by two years, to have the health at previous survey:
	df_2022_2020 = leftjoin(df_2022,df_2020, on = :ID, makeunique=true)
	df_2020_2018 = leftjoin(df_2020,df_2018, on = :ID, makeunique=true)
	df_2018_2016 = leftjoin(df_2018,df_2016, on = :ID, makeunique=true)

	dff = vcat(df_2022_2020,df_2020_2018,df_2020_2018)
	rename!(dff,Dict("Health" => "Health_t",
					 "Health_1" => "Health_t_1",
					"Age" => "Age_t"))
	dff = select(dff,[:ID,:Age_t,:Health_t,:av_annual_t,:Health_t_1])

	# We proceed to some last cleaning:
	dff = dropmissing!(dff)
	dff = clean_health(dff, "Health_t")
	dff = clean_health(dff, "Health_t_1")
	
end

begin 
	# Formatting: 
    dff.Health_t            = categorical(dff.Health_t)
    dff.Health_t_1          = categorical(dff.Health_t_1)
    dff.av_annual_t         = Float64.(dff.av_annual_t)
    dff.Age_t               = Float64.(dff.Age_t)
    y                       = coerce(dff.Health_t, Multiclass)
    X                       = select(dff, [:Health_t_1, :av_annual_t, :Age_t])

	# Encoding:
	HotEncoder      = @load OneHotEncoder pkg=MLJModels
    encoder         = HotEncoder()
    mach_encoder    = machine(encoder, X)
    fit!(mach_encoder)
    
    X_encoded       = MLJ.transform(mach_encoder, X)

    # Create and fit the model
    model           = MultinomialClassifier(penalty=:none)
    mach            = machine(model, X_encoded, y)
    MLJ.fit!(mach)

    probabilities   = MLJ.predict(mach, X_encoded)

	av_annual_range = 
		range(minimum(dff.av_annual_t)-1, maximum(dff.av_annual_t)+1, length=100)

    # 2. Get all Health_1 categories
    health_t_1_categories = levels(dff.Health_t_1)
    health_t_categories = levels(dff.Health_t)

    # Here, we need an age range: 
    age_range = range(minimum(dff.Age_t), maximum(dff.Age_t))
end

begin 

	# Create a plot for each Health_1 category
	
	P = Array{Any}(undef,5) 	# Initialise empty array for plots and probability matrix
	Prob_Matrix = Array{Any}(undef,5)
	i = 1 	# Initialise index
	
	for health_t_1 in health_t_1_categories
        # Create a DataFrame with fixed Health_1 and varying av_annual_t
        plot_data = DataFrame(
            Health_t_1   =
			fill(health_t_1, length(av_annual_range) * length(age_range)),
            av_annual_t  = repeat(av_annual_range, length(age_range)),
            Age_t        = repeat(age_range, inner = length(av_annual_range)))
        
        # Align categorical levels
        plot_data.Health_t_1 = categorical(plot_data.Health_t_1)
        levels!(plot_data.Health_t_1, health_t_1_categories)  # Force same levels

        plot_data_encoded = MLJ.transform(mach_encoder, plot_data) #? 
        
        # Get predicted probabilities
        probs = MLJ.predict(mach, plot_data_encoded) #? 

        # Reshape probabilities into a matrix (av_annual_t × Age_t × Health)
        prob_matrix =
			reshape([p.prob_given_ref[l] for p in probs, l in health_t_1_categories],
            (length(av_annual_range),
			 length(age_range),
			 length(health_t_1_categories)))

        # Create 3D plot
        p =
			Plots.surface(title = "Transition from Health_t_1 = $health_t_1",
            xlabel = "av_annual_t", 
            ylabel = "Age_t",
            zlabel = "Probability",
            # ylims = (minimum(age_range), maximum(age_range)),
            legend = :topright)
        
        # Plot a surface for each target health category
        for (i, health) in enumerate(health_t_1_categories)
            surface!(av_annual_range, age_range, prob_matrix[:, :, i]',
            label = "To $health", alpha = 0.7)
        end

		P[i] 			= p
		Prob_Matrix[i] 	= prob_matrix

		i = i + 1
        # display(p)
    end
end

plotly()
P[1]
DF = clean_health(df,"Health")

begin 
	Plots.plotly()

	# fixed_temperature = 0.616167 # Temperature deviation of 2018
	fixed_temperature = 0.61 # Temperature deviation of 2018
	
	model_health_age_temperature_gdp =
		GLM.glm(@formula(Status ~
						 	Age +
							Health +
							av_annual_t + 
							GDP
							),
				DF, Bernoulli(), LogitLink())

    Poor        =
		DataFrame(Age = 1:110,
				  Health = fill(5,110),
				  av_annual_t = fill(fixed_temperature,110),
				  GDP = 21354.1)
	
    Fair        = 
		DataFrame(Age = 1:110,
				  Health = fill(4,110),
				  av_annual_t = fill(fixed_temperature,110),
				  GDP = 21354.1)
	
    Good        = DataFrame(Age = 1:110,
							Health = fill(3,110),
							av_annual_t = fill(fixed_temperature,110),
							GDP = 21354.1)
	
    VeryGood    = DataFrame(Age = 1:110,
							Health = fill(2,110),
							av_annual_t = fill(fixed_temperature,110),
							GDP = 21354.1)
	
    Excellent   = DataFrame(Age = 1:110,
							Health = fill(1,110),
							av_annual_t = fill(fixed_temperature,110),
							GDP = 21354.1)

    pv  = GLM.predict(model_health_age_temperature_gdp,Poor)
    fv  = GLM.predict(model_health_age_temperature_gdp,Fair)
    gv  = GLM.predict(model_health_age_temperature_gdp,Good)
    vgv = GLM.predict(model_health_age_temperature_gdp,VeryGood)
    ev  = GLM.predict(model_health_age_temperature_gdp,Excellent)

    Plots.plot(pv, label = "Poor")
    Plots.plot!(fv, label = "Fair")
    Plots.plot!(gv, label = "Good")
    Plots.plot!(vgv, label = "Very good")
    Plots.plot!(ev, label = "Excellent")
    Plots.plot!(xaxis = "Age",
				yaxis = "Survival Probability")
end

begin 
	plotly()  # Use Plotly for 3D
	
	# Create age-temperature grid
	age_range2 = 1:110
	# temp_range = range(minimum(DF.av_annual_t), maximum(DF.av_annual_t), length=100)
	temp_range = range(-1, 1, length = 100)
		
	age_grid = repeat(age_range2', length(temp_range), 1)
	temp_grid = repeat(temp_range, 1, length(age_range2))
	
	# Health categories (assuming 1=Excellent, 5=Poor)
	health_levels = [1, 2, 3, 4, 5]  
	health_labels = ["Excellent", "VeryGood", "Good", "Fair", "Poor"]
	
	# Initialize plot
	plt = plot()
	
	for (health, label) in zip(health_levels, health_labels)
	    # Predict for all age-temp combinations
	    pred_grid = [GLM.predict(model_health_age_temperature_gdp, 
	                  DataFrame(Age=a, Health=health, av_annual_t=t, GDP=21354.1))[1]
	                  for t in temp_range, a in age_range2]
	    
	    # Add surface plot
	    surface!(plt, age_range2, temp_range, pred_grid, label=label)
	end
	
	# Customize plot
	plot!(plt, 
	    title="Survival Probability by Age and Temperature",
	    xlabel="Age", 
	    ylabel="Temperature (av_annual_t)",
	    zlabel="Survival Probability",
	    camera=(30, 45)  # Adjust viewing angle
	)
end

begin 
	""" 
	The survival function returns the probability of survival given the Age, current health, annual temperature, and GDP. 
		
		function survival(;Age::Int64,
			Health::Int64,
			Temperature::Float64,
			GDP::Float64)::Float64
	
	"""
	function survival(;Age::Int64, Health::Int64, Temperature::Float64, GDP::Float64)::Float64

		# For reminder: 
		# model_health_age_temperature_gdp =
		# GLM.glm(@formula(Status ~
		# 				 	Age +
		# 					Health +
		# 					av_annual_t + 
		# 					GDP
		# 					),
		# 		DF, Bernoulli(), LogitLink())

		data = DataFrame(Age = Age, 
						Health = Health, 
						av_annual_t = Temperature, 
						GDP = GDP)

		result = GLM.predict(model_health_age_temperature_gdp,data)

		# result = Float64.(result)

		result = result[1]
		
		return result
	end
end

survival(Age = 50, Health = 1, Temperature = 0.62, GDP = 2200.0)


begin 
	"""
	The function `health` returns a 5 elements vector for the transition probabilities of each of the health status.

			function health(;Age::Int64,
					Health_t_1::Int64,
					Temperature::Float64)::Vector{Float64}
	"""
	function health(;Age::Int64,
					Health_t_1::Int64,
					Temperature::Float64)::Vector{Float64}

		health_t_categories = categorical([1,2,3,4,5])
		        
			# Create a DataFrame with fixed Health_1 and temperature
        	plot_data = DataFrame(
        	    Health_t_1   = Health_t_1,
        	    av_annual_t  = Temperature,  
        	    Age_t        = Age)
        	
		# Align categorical levels
		plot_data.Health_t_1 = categorical(plot_data.Health_t_1)
		levels!(plot_data.Health_t_1, health_t_categories) # Force same levels
		
		plot_data_encoded = MLJ.transform(mach_encoder, plot_data) #? 
        
        # Get predicted probabilities
        probs = MLJ.predict(mach, plot_data_encoded) #? 

        # Reshape probabilities into a 5 sized vector
        prob_matrix =
			reshape([p.prob_given_ref[l] for p in probs, l in health_t_categories],
            (length(health_t_categories)))

		return prob_matrix
	end
		
end


function survival_to_hazard(p)
    return -log(p)
end

function hazard_to_survival(h)
    return exp(-h)
end

function corrected_survival(;Age, Health, Temperature, GDP)
    # Get baseline probability from model
    p = survival(Age=Age, Health=Health, Temperature=Temperature, GDP=GDP)
    
    # Convert to hazard and apply age acceleration
    h = survival_to_hazard(p) # * (1 + 0.05*max(0, Age-65))
    
    # Convert back to probability
    return hazard_to_survival(h)
end

begin 	
	""" 
	The function `population_simulation` allows for a simulation of the evolution of a population of size `N` for `T` periods, given a weather and a GDP path.
		
		population_simulation(;N::Int64,
								   T::Int64,
								   weather_history::Vector{Float64},
								   GDP::Vector{Float64})
	"""
	function population_simulation(;N::Int64,
								   T::Int64,
								   weather_history::Vector{Float64},
								   GDP::Vector{Float64})

		# We initialise the array that will contain all the results:
		# collective_results = []
		collective_age = []
		collective_living_history = []
		collective_health_history = []
		
		# Initialise a common weather history for the population
		# weather_history # = rand(Normal(μ,σ), T)

		# The first element of the array is the weather history
		# push!(collective_results,weather_history)
		
		# For each individual
		for i in 1:N # i = 1
			
			# We initialise the array that will contain the individual results:
			# individual_results = []
			
			individual_living_history = zeros(T) # T = 10
			individual_health_history = Vector{Int64}(undef,T)
	
			for t in 1:T # For each period 
                # t = 1

			   # Individuals are born in excellent health
				if t == 1
				   individual_past_health = 1
                   cumulative_survival_prob = 1
				end
		    
			    # The age : 
			    age = t
			    
			    # The weather comes from the weather history
			    weather_t = weather_history[t] # weather_history = zeros(10)
			    
			    # The health status :
					# probability of being in good health: 
					individual_pgh = health(Age = age, 
											Health_t_1 = individual_past_health,
                                            Temperature = weather_t)::Vector{Float64}
				
					# Health status draw:
					individual_health_t = 	
						sample(1:5,Weights(individual_pgh))

					# We add it to the history
					individual_health_history[t] = individual_health_t
					# The current health becomes the past one for next period
					individual_past_health = individual_health_t
	
			    # The living status : 
				    # Probability:
                    # GDP = zeros(T)
					# individual_pd = survival2(Age = age,
					# 						 Health = individual_health_t, 
					# 						 Temperature = weather_t,
					# 						 GDP = GDP[t])::Float64
                    
                    annual_survival = corrected_survival(
                        Age = age,
                        Health = individual_health_t,
                        Temperature = weather_t,
                        GDP = GDP[t])

                        cumulative_survival_prob *= annual_survival

				    # realisation : 
					individual_living_status = rand(Binomial(1,cumulative_survival_prob))
				    # Into its history :
					individual_living_history[t] = individual_living_status

				# When death comes : 
				if individual_living_status == 0
					push!(collective_age, age)
				    push!(collective_living_history, individual_living_history)
					push!(collective_health_history, individual_health_history)
					break
				end
				
			end # End of loop over periods
			
			# We add the information of the individual:
			# push!(collective_results,individual_results)
			# We go to the next individual
			
		end # End of loop over individuals

		results = (;weather_history,collective_age,collective_living_history,collective_health_history)
		# println("Life expectancy in this population: ", mean(collective_age))
		# population_results = (;age_of_death,living_history,health_history)
		return(results)
	end
end 

periods = 150
population_simulation(N = 1000,
                      T = periods, 
                      weather_history = fill(fixed_temperature,periods),
                      GDP = fill(21354.0,periods))