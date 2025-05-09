begin 
    using Plots
    using DataFrames
    using GLM
    using OrdinalMultinomialModels
    using MLJLinearModels, MLJ, MLJModels
end

begin
    include("../Climate/df.jl")
    include("../GDP/df.jl")
    include("../HRS_data/df.jl")
end

# Checking load of data:
begin 
    temperature
    gdp
    df = df_final
end

# Merging: 
begin 
    # Join df with gdp (leftjoin keeps all rows in df, even if no GDP data exists)
    combined = leftjoin(df, gdp, on=:Year)

    # Then join with temperature data (again, keeping all rows from the previous result)
    combined = leftjoin(combined, temperature[!, [:Year, :av_annual_t]], on=:Year)

    df = combined
    # df
    df = dropmissing!(df)
end

begin 
	Plots.gr()

	# fixed_temperature = 0.616167 # Temperature deviation of 2018
	# fixed_temperature = 0.61 # Temperature deviation of 2018
	
    age_range = range(minimum(df.Age), maximum(df.Age))
    temperature_2020 = Float64(temperature[temperature.Year .== 2020, :av_annual_t][1])
    temperature_2018 = Float64(temperature[temperature.Year .== 2018, :av_annual_t][1])
    temperature_2016 = Float64(temperature[temperature.Year .== 2016, :av_annual_t][1])
    temperature_2014 = Float64(temperature[temperature.Year .== 2014, :av_annual_t][1])
    GDP_2010 = gdp[gdp.Year .== 2010, :GDP][1]
    GDP_2012 = gdp[gdp.Year .== 2012, :GDP][1]
    GDP_2014 = gdp[gdp.Year .== 2014, :GDP][1]
    GDP_2016 = gdp[gdp.Year .== 2016, :GDP][1]
    GDP_2018 = gdp[gdp.Year .== 2018, :GDP][1]
    GDP_2020 = gdp[gdp.Year .== 2020, :GDP][1]
    GDP_2022 = gdp[gdp.Year .== 2022, :GDP][1]

	model_health_age_temperature_gdp =
		GLM.glm(@formula(Status ~
						 	Age +
							Health +
							av_annual_t +
							Age * Health * av_annual_t + 
							Health * av_annual_t +
							Age * av_annual_t + 
							# When including GDP, the solver does not converge
							log(GDP) + 
							Age * log(GDP) + 
							Year +
							Year * Age + 
							Age * Health
						),
				df, Bernoulli(), LogitLink())

    Poor        =
		DataFrame(Age = age_range,
				  Health = fill(5,length(age_range)),
				  av_annual_t = fill(temperature_2018,length(age_range)),
				  GDP = fill(GDP_2018, length(age_range)),
				  Year = fill(2018,length(age_range)))
	
    Fair        = 
		DataFrame(Age = age_range,
				  Health = fill(4,length(age_range)),
				  av_annual_t = fill(temperature_2018,length(age_range)),
				  GDP = fill(GDP_2018,length(age_range)),
				  Year = fill(2018,length(age_range)))
	
    Good        = DataFrame(Age = age_range,
							Health = fill(3,length(age_range)),
							av_annual_t = fill(temperature_2018,length(age_range)),
							GDP = GDP_2018,
							Year = 2018)
	
    VeryGood    = DataFrame(Age = age_range,
							Health = fill(2,length(age_range)),
							av_annual_t = fill(temperature_2018,length(age_range)),
							GDP = GDP_2018,
						   Year = 2018)
	
    Excellent   = DataFrame(Age = age_range,
	 						Health = fill(1,length(age_range)),
	 						av_annual_t = fill(temperature_2018,length(age_range)),
	 						GDP = GDP_2018,
	 					   Year = 2018)
	 
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
				yaxis = "Probability",
				title = "Survival probability as a function of age")
	Plots.plot!(legend = :bottomleft)
end

# Data: 

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

begin 
    # We define some year-specific dataframes: 
    df_2022 = df[df[:,:Year] .== 2022,:]	
    df_2020 = df[df[:,:Year] .== 2020,:]
    df_2018 = df[df[:,:Year] .== 2018,:]
    df_2016 = df[df[:,:Year] .== 2016,:]
    df_2014 = df[df[:,:Year] .== 2014,:]
    df_2012 = df[df[:,:Year] .== 2012,:]
    df_2010 = df[df[:,:Year] .== 2010,:]
    df_2008 = df[df[:,:Year] .== 2008,:]
    df_2006 = df[df[:,:Year] .== 2006,:]
    df_2022_2020 = leftjoin(df_2022,df_2020, on = :ID, makeunique=true)
	df_2020_2018 = leftjoin(df_2020,df_2018, on = :ID, makeunique=true)
	df_2018_2016 = leftjoin(df_2018,df_2016, on = :ID, makeunique=true)
	df_2016_2014 = leftjoin(df_2016,df_2014, on = :ID, makeunique=true)
	df_2014_2012 = leftjoin(df_2014,df_2012, on = :ID, makeunique=true)
	df_2012_2010 = leftjoin(df_2012,df_2010, on = :ID, makeunique=true)
	df_2010_2008 = leftjoin(df_2010,df_2008, on = :ID, makeunique=true)

	dff = vcat(df_2022_2020,
			   df_2020_2018,
			   df_2018_2016,
				df_2016_2014, 
				df_2014_2012,
				df_2012_2010,
				df_2010_2008)
	
	rename!(dff,Dict("Health" => "Health_t",
					 "Health_1" => "Health_t_1",
					"Age" => "Age_t"))
	dff = select(dff,[:ID,
					  :Age_t,
					  :Health_t,
					  :av_annual_t,
					  :Health_t_1,
					  :GDP])

	# We proceed to some last cleaning:
	dff = dropmissing!(dff)
	dff = clean_health(dff, "Health_t")
	dff = clean_health(dff, "Health_t_1")   
end

# Health transition model: 
begin 
    begin 
        # Formatting: 
        dff.Health_t            = categorical(dff.Health_t)
        dff.Health_t_1          = categorical(dff.Health_t_1)
        dff.av_annual_t         = Float64.(dff.av_annual_t)
        dff.Age_t               = Float64.(dff.Age_t)
        y                       = coerce(dff.Health_t, Multiclass)
        X                       = select(dff, [:Health_t_1, :av_annual_t, :Age_t])
    
        # Encoding:
        HotEncoder      = MLJ.@load OneHotEncoder pkg=MLJModels
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
end

begin 
	""" 
	The survival function returns the probability of survival given age, current health, annual temperature, and GDP. 
		
		function survival(;Age::Int64,
			Health::Int64,
			Temperature::Float64,
			GDP::Float64)::Float64
	
	"""
	function survival(;Age::Int64,
					  Health::Int64,
					  Temperature::Float64,
					  GDP::Float64)::Float64

		# For reminder: 
		# model_health_age_temperature_gdp =
		# GLM.glm(@formula(Status ~
		# 				 	Age +
		# 					Health +
		# 					av_annual_t + 
		# 					GDP
		# 					),
		# 		df, Bernoulli(), LogitLink())

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
								   GDP::Vector{Float64})::NamedTuple

		# Initialisation:
		collective_age 						= []
		collective_living_history 			= []
		collective_health_history 			= []
		collective_probability_history 		= []
		
		Threads.@threads for i in 1:N # For each individual
			
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
											Health_t_1 	= individual_past_health,
											Temperature = weather_t)::Vector{Float64}
				
					# Health status draw:
					individual_health_t = 	
						sample(1:5,Weights(individual_pgh))

					# We add it to the history
					individual_health_history[t] = individual_health_t
					# The current health becomes the past one for next period
					individual_past_health = individual_health_t
	
			    # Living status : 
				
					annual_survival = survival(Age 			= age,
											 Health 		= individual_health_t, 
											 Temperature 	= weather_t,
											 GDP 			= GDP[t])::Float64
				
            		cumulative_survival_prob = 
						cumulative_survival_prob * annual_survival

					individual_probability_history[t] = cumulative_survival_prob
				
				    # Realisation : 
					individual_living_status = 
						rand(Binomial(1,cumulative_survival_prob))#*individual_pd_2))

				# Possible alternative formulation: 
					# if rand() > cumulative_survival_prob
					# 	individual_living_status = 0
					# else
					# 	individual_living_status = 1
					# end

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

        life_expenctancy = mean(collective_age)
		results = (;weather_history,
				   collective_age,
				   collective_living_history,
				   collective_health_history, 
				  collective_probability_history, 
                  life_expenctancy)
		println("Life expectancy in this population: ", life_expenctancy)
		
		return(results)
	end
end 

begin 
	periods 			= 110
	fixed_temperature_1 = 0.61 # 2018
	
	population_1 = population_simulation(N 	= 1_000,
						  T 				= periods, 
						  weather_history 	= fill(fixed_temperature_1,periods),
						  GDP 				= fill(gdp[gdp.Year .== 2018,:GDP][1], periods), 
                          Year              = fill(2018, periods))
	
    population_1.life_expenctancy

end