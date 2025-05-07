### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 7f5a055e-4572-490d-8126-debfa842b82b
begin 
	using CSV
	using DataFrames
	using PlutoUI
	using CategoricalArrays
	using MLJ
	using MLJModels
	using MLJLinearModels
	using Plots
	using GLM
	using StatsBase
	using Base.Threads
	using NamedArrays
	using Loess
	using Random
	using JLD2
	TableOfContents(title = "Empirical model 1")
end

# ╔═╡ 03b5f43c-2806-11f0-1aad-913619158eaf
md"""# Introduction

This model is based upon the following data: 

* HRS data (for health, age, and status)
* Berkely climate data (for annual temperature)
* Federal Bank of Minessota (for GDP of te USA) 

They can be accessed here: 

- [https://hrsdata.isr.umich.edu](https://hrsdata.isr.umich.edu)
- [https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Complete_TAVG_complete.txt](https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Complete_TAVG_complete.txt)
- [https://fred.stlouisfed.org/series/MNNGSP](https://fred.stlouisfed.org/series/MNNGSP)


## HRS Data

After cleaning and merging the HRS data, we obtain:
"""

# ╔═╡ 37dd945d-d760-4f94-a46e-672e61c2057e
begin 
	df = CSV.read("data.csv", DataFrame)
	# df = select(df, Not([:ID]))
end

# ╔═╡ c2364d33-9afe-426b-ac1b-59504913c4df
describe(df)

# ╔═╡ 5ad9efa0-a367-4643-88e3-d0fe69d8132a
md""" ## Berkeley climate data """

# ╔═╡ ea58fb60-fdd3-4d00-b7fe-ceafe869adf0
begin 
	temperature = CSV.read("temperature.csv", DataFrame)
	Plots.gr()
	Plots.plot(temperature.Year,
			   temperature.av_annual_t,
			   legend = false, 
			   xaxis = "Year", 
			   yaxis = "Temperature",
			   title = "Average annual temperature deviation \n from the 1950s")
end

# ╔═╡ 8141f03e-7a82-4b90-9a5d-427201b3be45
begin 
	Plots.gr()
	Plots.plot(temperature.Year,
			   temperature.av_5_annual_t,
			   legend = false, 
			   xaxis = "Year", 
			   yaxis = "Temperature",
			   title = "Average 5 years temperature deviation \n from the 1950s")
end

# ╔═╡ 2c6a9943-dbcc-44c6-8f84-7035cf920208
md""" ## Minnesota Federal Bank Data """

# ╔═╡ 69649ff8-e60a-42d8-9f55-c291d6602ca2
begin 
	Plots.gr()
	gdp = CSV.read("gdp.csv", DataFrame)
	Plots.plot(gdp.Year,gdp.GDP, legend = false, 
			  xaxis = "Year", yaxis = "GDP", title = "USA GDP in Billion dollars")
	# gdp[gdp.Year .== 2020,:GDP][1]
end

# ╔═╡ 5a8a26ec-4d3c-4d8e-a709-f2a4fab66801
md"""# Environment 

The environment part of this model is subdivised into the following parts: 

- Health transition modelling, 
- Survival modelling
"""

# ╔═╡ b28eb188-6d11-43ef-86d1-cb739dab4646
md"""## Health


Let Health status at period $t$ be denoted $h_t$.

Basing myself on HRS data, I perform a multinomial logistic regression to explain Health status at period $t$ with Health status at period $t-1$ (the HRS survey is done each two years, so if $t = 2022$, $t-1 = 2020$) and temperature.


First, we must reconstruct the data to obtain a DataFrame with the following columns: 

- Year 
- Health at this year ($Y$),
- Health at previous survey ($X_1$), 
- Average temperature for this year ($X_2$).

"""

# ╔═╡ b7f90644-d4fe-4cb7-a60e-5ab180a35d50
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

# ╔═╡ 1a625bfe-ddc5-4fb3-b006-6b70b086559c
md"""To get the detail of the effect of Temperature, we can plot a heatmap:"""

# ╔═╡ 92ddcaa0-5105-482a-bc90-385b53270e8b
md"""These results are counter-intuitive: temperature seems to have a slightly positive effect on health transition in some cases, but this could be driven by the effect of age and survival.

However, another explanation could be due to the effect of economic growth: since the annual average of temperature is positively correlated with economic production, the isolated effect of temperature on other variables could be heavily affected by the omission of GDP.
"""

# ╔═╡ e5693cfe-19c3-4c35-a361-89a6560a7b21
begin 
	recent_gdp 			= gdp[gdp.Year .>= 1947, :GDP]
	recent_temperature 	= temperature[temperature.Year .>= 1947, :av_annual_t]

	Plots.scatter(recent_gdp,
				  recent_temperature, 
				 xaxis = "GDP",
				 yaxis = "Temperature", 
				 legend = false)
end

# ╔═╡ 1905c0a8-18e8-4f67-8c37-5c2daf756bb0
md""" ## Survival 

Regarding survival, the results are easier to get. Running a logistic regression to explain the living status (1 if alive, 0 if dead) on age, health, average annual temperature and GDP, we obtain similar results to the literature regarding health status influence on survival:
"""

# ╔═╡ 342da11c-cc36-4156-8d5a-e93d859e7a8c
begin
	DF = dropmissing!(df)
	DF = clean_health(df,"Health")
	
	# DF = DF[DF[:,:Year] .!= 2014,:]
	# DF = DF[DF[:,:Year] .!= 2016,:]
	# DF = DF[DF[:,:Year] .!= 2018,:]
	# DF = DF[DF[:,:Year] .!= 2020,:]
	# DF = DF[DF[:,:Year] .!= 2022,:]
end

# ╔═╡ 2336c8fd-263f-42c2-85d4-2b5a64f11a69
begin 

	# We define some year-specific dataframes: 
	
	df_2022 = DF[DF[:,:Year] .== 2022,:]	
	df_2020 = DF[DF[:,:Year] .== 2020,:]
	df_2018 = DF[DF[:,:Year] .== 2018,:]
	df_2016 = DF[DF[:,:Year] .== 2016,:]
	df_2014 = DF[DF[:,:Year] .== 2014,:]
	df_2012 = DF[DF[:,:Year] .== 2012,:]
	df_2010 = DF[DF[:,:Year] .== 2010,:]
	df_2008 = DF[DF[:,:Year] .== 2008,:]
	df_2006 = DF[DF[:,:Year] .== 2006,:]
	
	nothing

end

# ╔═╡ 7829dd48-681f-4fa1-93f9-9107848e0a98
begin 
	# We group data by two years, to have the health at previous survey:
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

# ╔═╡ babfe2e0-19b6-4a22-bc4f-74716ec50ec9
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

# ╔═╡ e512be5c-ef0d-4595-b516-d9117e703153
begin 

	plotly()
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

# ╔═╡ d4a5c3bc-4b3a-4936-adc9-68412109626c
Prob_Matrix

# ╔═╡ a32b576d-44c3-4f09-93b3-3386dd1ee1de
begin 
	plotly()
	Plots.plot(P[1])
end

# ╔═╡ 96e9ea8a-129c-4ab3-a2c3-c947fe4a83b8
Plots.plot(P[2])

# ╔═╡ 0d24cd79-fe1b-47fb-8a78-872b3335b6c5
Plots.plot(P[3])

# ╔═╡ ccfb6fc8-60a2-4b1d-a988-a4938e7c5b3b
Plots.plot(P[4])

# ╔═╡ 8a8e9034-f6be-4715-b481-2f62a554bbef
Plots.plot(P[5])

# ╔═╡ 388af98a-fec2-45e2-897a-6e818d586745
av_annual_range

# ╔═╡ 9bce461d-575e-4141-9cc8-a9e6ce449973
begin 
	# Examples to take:
	# Negative effect of temperature: 
	# 1 to 1, 
	# 1 to 2,
	# 1 to 3,
	# 5 to 5, 
	# Positive effect of temperature: 
	# 1 to 2
	# 5 to 1:4, 
	# 2 to 2,
	# 3 to 3
	
Plots.gr()
	ht = 3
	ht_1 = 5
	heatmap(age_range,
		av_annual_range,
		Prob_Matrix[ht][:,:,ht_1],
		title="2D Slice Check: Probability of going from $ht to $ht_1 \n",
		xlabel="Age",
		ylabel="Average annual temperature")
end

# ╔═╡ 50251579-de6d-43f3-af36-dc459585ad22
describe(DF)

# ╔═╡ 60194c76-1ae5-49af-8baf-0f3465c634e1
begin 
	#temperature_2022 = Float64.(temperature[temperature.Year .== 2022, :av_annual_t][1])
	temperature_2020 =
		Float64(temperature[temperature.Year .== 2020, :av_annual_t][1])
	temperature_2018 =
		Float64(temperature[temperature.Year .== 2018, :av_annual_t][1])
	temperature_2016 =
		Float64(temperature[temperature.Year .== 2016, :av_annual_t][1])
	temperature_2014 =
		Float64(temperature[temperature.Year .== 2014, :av_annual_t][1])
end

# ╔═╡ f97b719e-8b44-472b-b1ce-57eb5303ea02
describe(DF)

# ╔═╡ a28d672e-693f-44ba-9c38-9edf43bfff41
md""" Now, if we want to make both variables vary: """

# ╔═╡ 038f2359-69ca-4227-bb12-9bec3d4a3ae5
md""" ## Simulations

If now we use the above estimated functions, we obtain: """

# ╔═╡ 919c1aae-4e9b-489f-9950-17d11f59555c
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

# ╔═╡ 520abb10-18f1-4747-8d72-b5749834d713
md""" The life expectancy is coherent with the descriptive statistics of the data. Also, it's important to remember that the life expenctancy of an individual in 1950 (average year minus average age) was 69 years old. This discrepancy with the current life expectancy (77 years old) is due to the year of birth of the individuals observed by the HRS data. """

# ╔═╡ b3dd0d43-dd39-4b2a-9f16-7ecb1f6646c9
Random.seed!(1234);

# ╔═╡ 1dabd52f-2e78-40a1-b8dc-d213fdf673a2
md""" # Maximisation problem 

The agents maximizes: 

$$\max_{c_{t},l_{t},s_{t+1}}{\mathbb{E}\left[\sum_{t=1}^{\infty} \beta^{t}\cdot u(c_t,l_t)\right]}$$

Their utility function is: 

$$u(c_{t},l_{t}) = \frac{c_{t}^{1-\rho}}{1-\rho}-\xi_{t}\cdot \frac{l_{t}^{1+\varphi}}{1+\varphi}$$

With : 

-  $c$  the consumption
-  $l$  the quantity of labor supply provided by the agent
-  $h$  the health status
-  $w$  the weather variable, which is here temperature
-  $\xi$ the labor disutility coefficient

The agent is subject to: 

$$c_{t} + s_{t+1} \leq l_{t}\cdot z_{t} + s_{t}\cdot(1+r_{t})$$

With: 

-  $c_t$ the consumption at period $t$
-  $s_{t+1}$ the savings for period $t+1$
-  $l_t$ the labor supply provided by the agent at period $t$
-  $z_t$ the productivity at time $t$
-  $s_{t}$ the savings available at the beginning of period $t$
-  $r_{t}$ the interest rate at period $t$

Also, let us define the borrowing constraint as: 

$$s_{t}\geq \underline{s}, \forall t$$
"""


# ╔═╡ e29e2ee8-1e85-4236-96ca-47962898e105
md"""# Numerical methods

We are now going to solve this problem numerically. 

"""

# ╔═╡ b04ea071-adf7-4d2b-9ad9-2acb6338c321
md""" ## Auxiliary functions 

The auxiliary functions of the numerical the numerical solving process are: 

- The utility funtion, 
- The budget surplus function, 
- The $\xi$ function, that determines the labor disutility coefficient for a given period."""

# ╔═╡ 32daa090-3c9a-4d9f-b1a2-d969cf4e6966
begin 
	"""
	The `budget_surplus` function computes the budget states for certain levels of consumption, labor supply, productivity, and savings.

	Its syntax is:
		
		budget_surplus(;c::Float64,
			l::Float64,
			sprime::Float64,
			s::Float64,
			z::Float64,
			r::Float64)::Float64

	"""
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
end

# ╔═╡ 0dcbe2c1-2ed3-4911-bafe-09e76f269153
budget_surplus(c = 10.00,
			   l = 10.00,
			   sprime = 1.00,
			   s = 1.00,
			   z = 1.00,
			   r = (1-0.8)/0.8)

# ╔═╡ b55356ef-c4d8-4533-b90a-d2ba2adbbfc3
begin 
	""" 
	The `ξ` function returns the disutility of work in the utility function.

	Its syntax is: 
		
		ξ(w,h)

	For now, it returns 1.
	
		# (1+abs(w))*(1+1(h=="bad"))
	"""
	function ξ(;w::Float64,h::Float64)::Float64
		return 1.00 # ((1 + abs(w)) * (1+1(h=="bad")))::Float64
	end
end

# ╔═╡ db51d7d8-6e3c-46c6-bf80-47d72cdb097e
begin 
	"""
	The `utility` function is defined such that its syntax is:
	
		utility(;c,l,z,w,h,ρ=1.5,φ=2)
	
	It returns:

		(abs(c)^(1-ρ))/(1-ρ) - ξ(w,h) *((abs(l)^(1+φ)/(1+φ)))

	
	"""
	function utility(;c::Float64,
						l::Float64,
						z::Float64,
						w::Float64,
						h::Float64,
						ρ = 1.50::Float64,
						φ = 2.00::Float64)::Float64
		return 100 + ( ((abs(c))^(1-ρ)) / (1-ρ) ) - ξ(w=w,h=h) * ( ((abs(l))^(1+φ)) / (1+φ) )::Float64
	end
end

# ╔═╡ 2648949a-69c4-4981-912b-99f707b8b07d
utility(c = 1.00, 
	   l = 1.00, 
	   z = 1.00, 
	   w = 1.00, 
	   h = 1.00)

# ╔═╡ 982814bb-a80e-4ad6-b981-a3a60691fdf4
md""" ## Main functions 

The main function of the numerical solving process are: 

- The Bellman function, that maximises the current period and the next one,
- The backwards function, that calls the Bellman function for all considered periods.

Two versions of the Bellman exist: 

- `Bellman`, that uses the F.O.C. 1 to approximate labor,
- `Bellman_numerical`, that try for all possible values of labor.
"""

# ╔═╡ f89b93cd-111e-46fb-9364-03600edcaddf
begin 
	 """
	 The `Bellman` function is not to be used alone, but with the `backwards` function.
	 
			function Bellman(;s_range::AbstractRange,
						sprime_range::AbstractRange,
						consumption_range::AbstractRange,
						labor_range::AbstractRange,
						value_function_nextperiod::Any,
						β = 0.96::Float64, 
						z = 1::Float64,
						ρ = 1.5::Float64,
						φ = 2::Float64,
						proba_survival = 0.9::Float64,
						r = ((1-0.9)/0.9)::Float64,
						w = 0.0::Float64,
						h = 2.00::Float64,
						return_full_grid = true::Bool, 
						return_budget_balance = true::Bool)::NamedTuple
	 
	 """
	function Bellman(;s_range::AbstractRange,
						sprime_range::AbstractRange,
						consumption_range::AbstractRange,
						# labor_range::AbstractRange,
						value_function_nextperiod::Any,
						β 					= 0.96::Float64, 
						z 					= 1.00::Float64,
						ρ 					= 1.5::Float64,
						φ 					= 2.00::Float64,
						proba_survival 		= 0.90::Float64,
						r 					= ((1-0.9)/0.9)::Float64,
						w 					= 0.00::Float64,
						h 					= 2.00::Float64,
					 	ξ 					= 1.00::Float64,
						return_full_grid 	= true::Bool, 
						return_budget_balance = true::Bool)::NamedTuple

		@assert length(value_function_nextperiod) == length(s_range) "The value function of the next period has the wrong length."

		# Initialization

		# Grid of possible values
		grid_of_value_function = Array{Float64}(undef,length(s_range),
															length(consumption_range),
															length(sprime_range))
		
		# Optimal utility and choice
		Vstar 					= zeros(length(s_range))
		index_optimal_choice 	= Array{CartesianIndex}(undef,length(s_range))
		optimal_choice 			= Array{Float64}(undef,length(s_range),2)

		if return_budget_balance == true
			tmp_budget = Array{Float64}(undef, 
										length(s_range),
										length(consumption_range),
										length(sprime_range))
			budget_balance = Array{Float64}(undef,
											length(s_range))
		end

		# for all levels of endowment
		for (index_s,s) in enumerate(s_range)
			# for all levels of consumption
			for (index_consumption,consumption) in enumerate(consumption_range) 

				# We fix labor with the FOC 1: 
				labor = (consumption ^(-ρ)*z/ξ)^(1/φ)

				# for (index_labor,labor) in enumerate(labor_range) 
					
					# for all levels of savings
					for (index_sprime,sprime) in enumerate(sprime_range)

						# Compute the budget:
						tmp = budget_surplus(c 		= consumption,
											 l 		= labor,
											 sprime = sprime,
											 s 		= s,
											 z 		= z,
											 r 		= r)
						
						tmp_budget[index_s,
								   index_consumption,
								   index_sprime] = tmp

						# If the budget constraint is violated,
						# set the value function to minus infinity :
						
						if tmp < 0 
							
							grid_of_value_function[index_s,
											index_consumption,
											index_sprime] = -Inf

						# If the budget constraint is not violated,
						# set the value function to the utility plus the value 
						# function at the next period : 
							
						elseif tmp >= 0
							
							grid_of_value_function[index_s,
											index_consumption,
											index_sprime] =
								utility(c = consumption,
										l = labor,
										z = z,
										w = w,
										h = h,
										ρ = ρ,
										φ = φ) +
										β * proba_survival * value_function_nextperiod[index_sprime]
						end
					end # end of sprime loop
				# end # end of labor loop
			end # end of consumption loop

			# For a given level of initial endowment, 
			# find the maximum of the value function 
			# and the associated optimal choice
			
			Vstar[index_s],
			index_optimal_choice[index_s] =
				findmax(grid_of_value_function[index_s,:,:])
				
			#Vstar[index_s],
			#index_optimal_choice[index_s] =
			#	findmax(grid_of_value_function[index_s,:,:,:])

			ioc = index_optimal_choice[index_s]

			# optimal_choice[index_s,:] =
				# [consumption_range[ioc[1]],
				# labor_range[ioc[2]]]
				# sprime_range[ioc[3]]]
			
			optimal_choice[index_s,:] = [consumption_range[ioc[1]],
										sprime_range[ioc[2]]]
			     
			# Validation check
        	optimal_surplus = tmp_budget[index_s, ioc[1],
										 # ioc[2],
										 ioc[2]]
        	@assert optimal_surplus >= 0 "Infeasible optimal choice found!
				Surplus: $optimal_surplus."
			
			if return_budget_balance
				budget_balance[index_s] = optimal_surplus
			end

		end # end of s

		# Formatting the output for better readability:
		
		# Transforming the grid_of_value_function array into a Named Array:

		param1_names 			= ["s_i$i" for i in 1:length(s_range)]
		savings_value 			= ["s=$i" for i in s_range]
		consumption_value 		= ["c=$i" for i in consumption_range]
		# labor_value 			= ["l=$i" for i in labor_range]
		sprime_value 			= ["l=$i" for i in sprime_range]
		
		grid_of_value_function = NamedArray(grid_of_value_function,
											(savings_value,
											 consumption_value,
											 sprime_value))

		optimal_choice = NamedArray(optimal_choice,
									(param1_names,
									 ["c","sprime"]))

		# Returning results:
		if return_full_grid == true && return_budget_balance == true
			return (;grid_of_value_function,Vstar,index_optimal_choice,optimal_choice,budget_balance)
		elseif return_full_grid == true && return_budget_balance == false
			return (;grid_of_value_function,Vstar,index_optimal_choice,optimal_choice)
		elseif return_full_grid == false && return_budget_balance == true
			return (;Vstar,index_optimal_choice,optimal_choice,budget_balance)
		elseif return_full_grid == false && return_budget_balance == false
			return (;Vstar,index_optimal_choice,optimal_choice)
		end
		
	end
end

# ╔═╡ 99fb1863-6d74-423e-8ea0-f8f4fa4d25e3
 """
	 The `Bellman_numerical` function is not to be used alone, but with the `backwards` function.
	 
			function Bellman_numerical(;s_range::AbstractRange,
						sprime_range::AbstractRange,
						consumption_range::AbstractRange,
						labor_range::AbstractRange,
						value_function_nextperiod::Any,
						β = 0.96::Float64, 
						z = 1::Float64,
						ρ = 1.5::Float64,
						φ = 2::Float64,
						proba_survival = 0.9::Float64,
						r = ((1-0.9)/0.9)::Float64,
						w = 0.0::Float64,
						h = 2.00::Float64,
						return_full_grid = true::Bool, 
						return_budget_balance = true::Bool)::NamedTuple
	 
	 """
	function Bellman_numerical(;s_range::AbstractRange,
						sprime_range::AbstractRange,
						consumption_range::AbstractRange,
						labor_range::AbstractRange,
						value_function_nextperiod::Any,
						β 					= 0.96::Float64, 
						z 					= 1.00::Float64,
						ρ 					= 1.5::Float64,
						φ 					= 2.00::Float64,
						proba_survival 		= 0.90::Float64,
						r 					= ((1-0.9)/0.9)::Float64,
						w 					= 0.00::Float64,
						h 					= 2.00::Float64,
						return_full_grid 	= true::Bool, 
						return_budget_balance = true::Bool)::NamedTuple

		@assert length(value_function_nextperiod) == length(s_range) "The value function of the next period has the wrong length."

		# Initialization

		# Grid of possible values
		grid_of_value_function = Array{Float64}(undef,length(s_range),
															length(consumption_range),
															length(labor_range),
															length(sprime_range))
		
		# Optimal utility and choice
		Vstar 					= zeros(length(s_range))
		index_optimal_choice 	= Array{CartesianIndex}(undef,length(s_range))
		optimal_choice 			= Array{Float64}(undef,length(s_range),3)

		if return_budget_balance == true
			tmp_budget = Array{Float64}(undef, 
										length(s_range),
										length(consumption_range),
										length(labor_range),
										length(sprime_range))
			budget_balance = Array{Float64}(undef,
											length(s_range))
		end

		# for all levels of endowment
		for (index_s,s) in enumerate(s_range)
			# for all levels of consumption
			for (index_consumption,consumption) in enumerate(consumption_range) 

				# Insert F.O.C.1.
				
				# for all levels of labor
				for (index_labor,labor) in enumerate(labor_range)
					
					# for all levels of savings
					for (index_sprime,sprime) in enumerate(sprime_range)

						# Compute the budget:
						tmp = budget_surplus(c 		= consumption,
											 l 		= labor,
											 sprime = sprime,
											 s 		= s,
											 z 		= z,
											 r 		= r)
						
						tmp_budget[index_s,
								   index_consumption,
								   index_labor,
								   index_sprime] = tmp

						# If the budget constraint is violated,
						# set the value function to minus infinity :
						
						if tmp < 0 
							
							grid_of_value_function[index_s,
											index_consumption,
											index_labor, 
											index_sprime] = -Inf

						# If the budget constraint is not violated,
						# set the value function to the utility plus the value 
						# function at the next period : 
							
						elseif tmp >= 0
							
							grid_of_value_function[index_s,
											index_consumption,
											index_labor, 
											index_sprime] =
								utility(c = consumption,
										l = labor,
										z = z,
										w = w,
										h = h,
										ρ = ρ,
										φ = φ) +
										β * proba_survival * value_function_nextperiod[index_sprime]
						end
					end # end of sprime loop
				end # end of labor loop
			end # end of consumption loop

			# For a given level of initial endowment, 
			# find the maximum of the value function 
			# and the associated optimal choice
			
			Vstar[index_s],
			index_optimal_choice[index_s] =
				findmax(grid_of_value_function[index_s,:,:,:])

			ioc = index_optimal_choice[index_s]
			
			optimal_choice[index_s,:] = [consumption_range[ioc[1]],
										labor_range[ioc[2]],
										sprime_range[ioc[3]]]
			     
			# Validation check
        	optimal_surplus = tmp_budget[index_s, ioc[1], ioc[2], ioc[3]]
        	@assert optimal_surplus >= 0 "Infeasible optimal choice found!
				Surplus: $optimal_surplus."
			
			if return_budget_balance
				budget_balance[index_s] = optimal_surplus
			end

		end # end of s

		# Formatting the output for better readability:
		
		# Transforming the grid_of_value_function array into a Named Array:

		param1_names 			= ["s_i$i" for i in 1:length(s_range)]
		savings_value 			= ["s=$i" for i in s_range]
		consumption_value 		= ["c=$i" for i in consumption_range]
		labor_value 			= ["l=$i" for i in labor_range]
		sprime_value 			= ["l=$i" for i in sprime_range]
		
		grid_of_value_function = NamedArray(grid_of_value_function,
											(savings_value,
											 consumption_value,
											 labor_value,
											 sprime_value))

		optimal_choice = NamedArray(optimal_choice,
									(param1_names,
									 ["c","l","sprime"]))

		# Returning results:
		if return_full_grid == true && return_budget_balance == true
			return (;grid_of_value_function,Vstar,index_optimal_choice,optimal_choice,budget_balance)
		elseif return_full_grid == true && return_budget_balance == false
			return (;grid_of_value_function,Vstar,index_optimal_choice,optimal_choice)
		elseif return_full_grid == false && return_budget_balance == true
			return (;Vstar,index_optimal_choice,optimal_choice,budget_balance)
		elseif return_full_grid == false && return_budget_balance == false
			return (;Vstar,index_optimal_choice,optimal_choice)
		end
		
end

# ╔═╡ 25bf9601-ff3d-4d23-a011-ddb788e2b126
Bellman_numerical(s_range 					= 0.00:2.00,
				  sprime_range 				= 0.00:2.00,
				  consumption_range 			= 0.00:10.00,
				  labor_range 					= 0.00:10.00,
				  value_function_nextperiod 	= zeros(3))

# ╔═╡ 976923cb-7132-415c-881d-f46485032623
Bellman(s_range 					= 0.00:2.00,
	   sprime_range 				= 0.00:2.00, 
	   consumption_range 			= 0.00:10.00,
	   value_function_nextperiod 	= zeros(3))

# ╔═╡ fbf795ba-0065-4e35-944e-36762dff99fd
begin 
	"""
		function backwards(;s_range::AbstractRange,
				sprime_range::AbstractRange,
				consumption_range::AbstractRange,
				labor_range::AbstractRange,
				nperiods::Integer,
				z = ones(nperiods)::Array,
				β = 0.9::Float64,
				r = final_r::Array,
				ρ = 1.50::Float64, 
				φ = 2.00::Float64,
				proba_survival = 0.90::Float64,
				w = 0.00::Float64,
				h = "good"::AbstractString, 
				return_full_grid = false::Bool, 
				return_budget_balance = true::Bool)::NamedTuple
	"""
	function backwards(;s_range::AbstractRange,
				sprime_range::AbstractRange,
				consumption_range::AbstractRange,
				nperiods::Integer,
				z 						= ones(nperiods)::Array{Float64},
				β 						= 0.90::Float64,
				r 						= ones(nperiods)::Array{Float64},
				ρ 						= 1.50::Float64, 
				φ 						= 2.00::Float64,
				proba_survival 			= 0.90 .* ones(nperiods)::Array{Float64},
				w 						= zeros(nperiods)::Array{Float64},
				h 						= 2 .* ones(nperiods)::Array{Float64}, 
				return_full_grid 		= false::Bool, 
				return_budget_balance 	= true::Bool)::NamedTuple

		# Initialization: 

		# We define the name of the variables for the named arrays: 
		param1_names 			= ["t_$i" for i in 1:nperiods]
		param2_names 			= ["s_i$i" for i in 1:length(s_range)]
		param3_names 			= ["c_i$i" for i in 1:length(consumption_range)]
		# param4_names 			= ["l_i$i" for i in 1:length(labor_range)]
		param5_names 			= ["sprime_i$i" for i in 1:length(sprime_range)]
		choice_variable_name 	= ["c","sprime"]
								
		savings_value 			= ["s=$i" for i in s_range]
		consumption_value 		= ["c=$i" for i in consumption_range]
		# labor_value 			= ["l=$i" for i in labor_range]
		sprime_value 			= ["l=$i" for i in sprime_range]

		# From the given ranges, construct a grid of all possible values, 
		# And save its size: 
		grid_of_value_function 	= Array{Float64}(undef,
												 length(s_range),
												 length(consumption_range),
												# length(labor_range),
												 length(sprime_range))
		points 					= size(grid_of_value_function)

		if return_budget_balance == true
			budget_balance = Array{Float64}(undef,nperiods,length(s_range))
		end
		
		# Initialize empty arrays that will:
		# contain the values of the value function (V): 
		V = zeros(nperiods,
				  points[1],
				  points[2],
				  #points[3],
				  points[3])

		# the values at optimum (Vstar), 
		Vstar = zeros(nperiods,points[1])
		
		# The indices of optimal choice (index_optimal_choices),
		index_optimal_choices = Array{Array{CartesianIndex{2}}}(undef,nperiods)

		# and the values of choice variables at the optimum (optimal_choices): 
		optimal_choices 	= Array{Float64}(undef,
											 nperiods,
											 length(sprime_range),
											 2) # Time periods, level of initial savings, choice variables
		optimal_choices 	=
			NamedArray(optimal_choices,
					   (param1_names,
						savings_value,
						choice_variable_name))

		# First, we solve for the last period, in which the value function of next period is 0: 
		last_Bellman = Bellman(s_range 		= s_range::AbstractRange,
	 					sprime_range 		= sprime_range::AbstractRange,
	 					consumption_range 	= consumption_range::AbstractRange,
	 					# labor_range 		= labor_range::AbstractRange,
						value_function_nextperiod = zeros(length(s_range)),
	 					β 					= β::Float64,
			 			ρ 					= ρ::Float64,
						φ 					= φ::Float64,
						r 					= r[nperiods]::Float64,
						proba_survival 		= proba_survival[nperiods]::Float64,
						w 					= w[nperiods]::Float64,
						h 					= h[nperiods]::Float64,
						z 					= z[nperiods]::Float64,
						return_full_grid 	= true::Bool,
						return_budget_balance = return_budget_balance::Bool)::NamedTuple

		if return_budget_balance == true
			budget_balance[end,:] = last_Bellman[:budget_balance]
		end
		
		# Value of the value function: 
		if return_full_grid == true
			V[end,:,:,:,:]	= last_Bellman[:grid_of_value_function] 
		end 
		
		# Values at the optimum:
		Vstar[end,:] 							.= last_Bellman[:Vstar] 	 
		
		# Index of optimal choice:
		index_optimal_choices[end] 				= last_Bellman[:index_optimal_choice]
		
		optimal_choices[end,:,:]				= last_Bellman[:optimal_choice]

		# Values of the choice variables at optimum:
		# optimal_choice[end,:] = collect(grid_of_value_function)
		
		for index_time in (nperiods-1):-1:1
			
			tmp = Bellman(s_range 				= s_range,
					sprime_range 				= sprime_range,
					consumption_range 			= consumption_range,
					# labor_range 				= labor_range,
					value_function_nextperiod 	= last_Bellman[:Vstar],
					β 							= β,
					z 							= z[index_time],
					ρ 							= ρ,
					φ 							= φ,
					r 							= r[index_time], 
					proba_survival 				= proba_survival[index_time], 
					w 							= w[index_time],
					h 							= h[index_time],
					return_full_grid 			= true,
					return_budget_balance 		= return_budget_balance)::NamedTuple
			
			if return_full_grid == true
				V[index_time,:,:,:,:] 			= tmp[:grid_of_value_function] 
			end

			if return_budget_balance == true
				budget_balance[index_time,:] 	= tmp[:budget_balance]
			end
				
			Vstar[index_time,:] 				= tmp[:Vstar]
			index_optimal_choices[index_time] 	= tmp[:index_optimal_choice]
			optimal_choices[index_time,:,:] 	= tmp[:optimal_choice] 
			
			last_Bellman = tmp
			
		end # end of time loop

		# Rename in NamedArrays:
		Vstar = NamedArray(Vstar, (param1_names,savings_value))
		if return_budget_balance == true
			budget_balance = NamedArray(budget_balance, (param1_names,savings_value))
		end
		if return_full_grid == true
			V = NamedArray(V, (param1_names, savings_value, consumption_value, labor_value, sprime_value))
		end
		

		if return_full_grid && return_budget_balance
			return (;V,Vstar,index_optimal_choices,optimal_choices,budget_balance)
		elseif return_full_grid
			return (;V,Vstar,index_optimal_choices,optimal_choices)
		elseif return_budget_balance
			return (;Vstar,index_optimal_choices,optimal_choices,budget_balance)
		else 
			return (;Vstar,index_optimal_choices,optimal_choices)
		end

	end
end

# ╔═╡ da863575-eeeb-44b1-a9d4-edd92ffb09ca
begin 
	"""
		function backwards(;s_range::AbstractRange,
				sprime_range::AbstractRange,
				consumption_range::AbstractRange,
				labor_range::AbstractRange,
				nperiods::Integer,
				z = ones(nperiods)::Array,
				β = 0.9::Float64,
				r = final_r::Array,
				ρ = 1.50::Float64, 
				φ = 2.00::Float64,
				proba_survival = 0.90::Float64,
				w = 0.00::Float64,
				h = "good"::AbstractString, 
				return_full_grid = false::Bool, 
				return_budget_balance = true::Bool)::NamedTuple
	"""
	function backwards_numerical(;s_range::AbstractRange,
				sprime_range::AbstractRange,
				consumption_range::AbstractRange,
				labor_range::AbstractRange,
				nperiods::Integer,
				z 						= ones(nperiods)::Array{Float64},
				β 						= 0.90::Float64,
				r 						= ones(nperiods)::Array{Float64},
				ρ 						= 1.50::Float64, 
				φ 						= 2.00::Float64,
				proba_survival 			= 0.90 .* ones(nperiods)::Array{Float64},
				w 						= zeros(nperiods)::Array{Float64},
				h 						= 2 .* ones(nperiods)::Array{Float64}, 
				return_full_grid 		= false::Bool, 
				return_budget_balance 	= true::Bool)::NamedTuple

		# Initialization: 

		# We define the name of the variables for the named arrays: 
		param1_names 			= ["t_$i" for i in 1:nperiods]
		param2_names 			= ["s_i$i" for i in 1:length(s_range)]
		param3_names 			= ["c_i$i" for i in 1:length(consumption_range)]
		param4_names 			= ["l_i$i" for i in 1:length(labor_range)]
		param5_names 			= ["sprime_i$i" for i in 1:length(sprime_range)]
		choice_variable_name 	= ["c","l","sprime"]
								
		savings_value 			= ["s=$i" for i in s_range]
		consumption_value 		= ["c=$i" for i in consumption_range]
		labor_value 			= ["l=$i" for i in labor_range]
		sprime_value 			= ["l=$i" for i in sprime_range]

		# From the given ranges, construct a grid of all possible values, 
		# And save its size: 
		grid_of_value_function 	= Array{Float64}(undef,length(s_range),
															length(consumption_range),
															length(labor_range),
															length(sprime_range))
		points 					= size(grid_of_value_function)

		if return_budget_balance == true
			budget_balance = Array{Float64}(undef,nperiods,length(s_range))
		end
		
		# Initialize empty arrays that will:
		# contain the values of the value function (V): 
		V = zeros(nperiods,points[1],points[2],points[3],points[4])

		# the values at optimum (Vstar), 
		Vstar = zeros(nperiods,points[1])
		
		# The indices of optimal choice (index_optimal_choices),
		index_optimal_choices = Array{Array{CartesianIndex{3}}}(undef,nperiods)

		# and the values of choice variables at the optimum (optimal_choices): 
		optimal_choices 	= Array{Float64}(undef,nperiods,length(sprime_range),3) # Time periods, level of initial savings, choice variables
		optimal_choices 	= NamedArray(optimal_choices,(param1_names,savings_value,choice_variable_name))

		# First, we solve for the last period, in which the value function of next period is 0: 
		last_Bellman = Bellman_numerical(s_range 		= s_range::AbstractRange,
	 					sprime_range 		= sprime_range::AbstractRange,
	 					consumption_range 	= consumption_range::AbstractRange,
	 					labor_range 		= labor_range::AbstractRange,
						value_function_nextperiod = zeros(length(s_range)),
	 					β 					= β::Float64,
			 			ρ 					= ρ::Float64,
						φ 					= φ::Float64,
						r 					= r[nperiods]::Float64,
						proba_survival 		= proba_survival[nperiods]::Float64,
						w 					= w[nperiods]::Float64,
						h 					= h[nperiods]::Float64,
						z 					= z[nperiods]::Float64,
						return_full_grid 	= true::Bool,
						return_budget_balance = return_budget_balance::Bool)::NamedTuple

		if return_budget_balance == true
			budget_balance[end,:] = last_Bellman[:budget_balance]
		end
		
		# Value of the value function: 
		if return_full_grid == true
			V[end,:,:,:,:]	= last_Bellman[:grid_of_value_function] 
		end 
		
		# Values at the optimum:
		Vstar[end,:] 							.= last_Bellman[:Vstar] 	 
		
		# Index of optimal choice:
		index_optimal_choices[end] 				= last_Bellman[:index_optimal_choice]
		
		optimal_choices[end,:,:]				= last_Bellman[:optimal_choice]

		# Values of the choice variables at optimum:
		# optimal_choice[end,:] = collect(grid_of_value_function)
		
		for index_time in (nperiods-1):-1:1
			
			tmp = Bellman_numerical(s_range 				= s_range,
					sprime_range 				= sprime_range,
					consumption_range 			= consumption_range,
					labor_range 				= labor_range,
					value_function_nextperiod 	= last_Bellman[:Vstar],
					β 							= β,
					z 							= z[index_time],
					ρ 							= ρ,
					φ 							= φ,
					r 							= r[index_time], 
					proba_survival 				= proba_survival[index_time], 
					w 							= w[index_time],
					h 							= h[index_time],
					return_full_grid 			= true,
					return_budget_balance 		= return_budget_balance)::NamedTuple
			
			if return_full_grid == true
				V[index_time,:,:,:,:] 			= tmp[:grid_of_value_function] 
			end

			if return_budget_balance == true
				budget_balance[index_time,:] 	= tmp[:budget_balance]
			end
				
			Vstar[index_time,:] 				= tmp[:Vstar]
			index_optimal_choices[index_time] 	= tmp[:index_optimal_choice]
			optimal_choices[index_time,:,:] 	= tmp[:optimal_choice] 
			
			last_Bellman = tmp
			
		end # end of time loop

		# Rename in NamedArrays:
		Vstar = NamedArray(Vstar, (param1_names,savings_value))
		if return_budget_balance == true
			budget_balance = NamedArray(budget_balance, (param1_names,savings_value))
		end
		if return_full_grid == true
			V = NamedArray(V, (param1_names, savings_value, consumption_value, labor_value, sprime_value))
		end
		

		if return_full_grid && return_budget_balance
			return (;V,Vstar,index_optimal_choices,optimal_choices,budget_balance)
		elseif return_full_grid
			return (;V,Vstar,index_optimal_choices,optimal_choices)
		elseif return_budget_balance
			return (;Vstar,index_optimal_choices,optimal_choices,budget_balance)
		else 
			return (;Vstar,index_optimal_choices,optimal_choices)
		end

	end
end

# ╔═╡ e029f368-ec95-4317-8e87-2153ace4c123
md""" ## Results """

# ╔═╡ 5c51eae5-c28e-4be1-8a5c-61ca8038f1c5
begin 
	s_range 				= -2:0.1:5
	sprime_range 			= -2:0.1:5
	consumption_max 		= 5
	nperiods 				= 104
	ζ 						= (1 ./(1:nperiods))
	r_pi0 					= (1 .- ζ) ./ζ
	# r = (1:nperiods).^2
	# r = fill(10,nperiods)
	r_min 					= ((1-0.96)/(0.96))
	# r = r_min .+ r_pi0
	r 						= fill(0.1,nperiods)
	typical_productivity 	= [exp(0.1*x - 0.001*x^2) for x in 1:nperiods]
end

# ╔═╡ 96445695-d923-44bd-8c68-d89fd974dc13
begin 
	"""
	The `smoothing` function smoothes the results of jumping policy functions.
			
		smoothing(y::AbstractArray{Float64},
				span::Float64)
	"""
	function smoothing(y::AbstractArray{Float64},
				   span::Float64)
		
		x = 1:length(y)
		model = loess(x,
					  y,
					  span = span)
		return Loess.predict(model,x)::Vector{Float64}
		
	end	
end

# ╔═╡ 2f1d1532-dc3f-4c08-9576-f0b9c3a633bc
common_span = 0.99

# ╔═╡ d523f5d5-113b-44a0-a631-b05acc1a408a
md""" ### Value function"""

# ╔═╡ 47fab280-4f42-412d-ad35-adf8fa2e988f
md""" ### Consumption"""

# ╔═╡ aef9601d-f05e-4995-937f-b1b9cb140e73
md""" ### Savings"""

# ╔═╡ 5cae56c7-c5ba-447b-a9ea-ef7f1ad99dfe
md""" ### Budget clearing

Finally, we can control that the budget constraint is binding. The following value should be close to 0. """


# ╔═╡ 5ee88736-733b-480a-b9e2-929408cef55f
md""" ## Comparison with analytical results 

From the maximization program, we have: 

$$c_{t} = \left[\frac{z_{t}}{\xi_{t}\cdot l_{t}^{\varphi}}\right]^{-\frac{1}{\rho}}$$

"""

# ╔═╡ 383d864d-310a-4e98-be68-48801a1c3a9d
begin
		ρ = 1.50::Float64 
		φ = 2.00::Float64
end

# ╔═╡ b9259bf1-e248-4ba7-bd9c-ae6094e368dd
md"""
The following could be useful to identify the lagrangien multiplicator:
"""

# ╔═╡ c3acbb4e-1330-4c7d-9875-fc89d8a35453
md""" # Conclusion: Aggregate effect of health 

To assess the welfare loss through the change in health status induced by climate change, we could now compare two different scenarios: One with a stabilized temperature distribution around the annual averages of the last years, and one in which the rise in annual average temperature continues.

To do so, one could consider the whole effect of temperature deviation: 

- On survival probability, 
- On health transition, 
- On productivity, 
- On work disutility, 
- On GDP.

The comparison of outputs with two different can partially be done analytically, but requires necessarily a numerical simulation parts.

Another approach would be a comparison through individual simulations, to assess the life-time cost of climate change, taking into account the effect of temperature on health, and survival. This approach would be less holistic, but more tractable. 

"""

# ╔═╡ 6502e5f8-56b4-4f57-8851-91b826c45bbf
begin 
	GDP_2010 = gdp[gdp.Year .== 2010, :GDP][1]
	GDP_2012 = gdp[gdp.Year .== 2012, :GDP][1]
	GDP_2014 = gdp[gdp.Year .== 2014, :GDP][1]
	GDP_2016 = gdp[gdp.Year .== 2016, :GDP][1]
	GDP_2018 = gdp[gdp.Year .== 2018, :GDP][1]
	GDP_2020 = gdp[gdp.Year .== 2020, :GDP][1]
	GDP_2022 = gdp[gdp.Year .== 2022, :GDP][1]
end

# ╔═╡ badaea58-868b-4ef0-aee8-3c4583f7b3ea
begin 
	Plots.gr()

	# fixed_temperature = 0.616167 # Temperature deviation of 2018
	# fixed_temperature = 0.61 # Temperature deviation of 2018
	
	model_health_age_temperature_gdp =
		GLM.glm(@formula(Status ~
						 	Age +
							Health +
							av_annual_t  
							# Age * Health * av_annual_t
							# Health * av_annual_t +
							# Age * av_annual_t
							# When including GDP, the solver does not converge
							# log(GDP) + 
							# Age * log(GDP) + 
							# Year +
							# Year * Age + 
							# Age * Health
						),
				DF, Bernoulli(), LogitLink())

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

# ╔═╡ d7c398ec-4325-40ae-8d71-1aa306830131
model_health_age_temperature_gdp

# ╔═╡ 6c5c973f-9197-4114-9924-8e7a299b1472
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

# ╔═╡ d7ba6c4c-0f80-453b-a893-648d80b0c9b3
begin 
	survival(Age = 99, Health = 3, Temperature = 0.62, GDP = 2200.0)
end

# ╔═╡ 22019072-9fa5-40a3-bb22-89696e19a2d5
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

		results = (;weather_history,
				   collective_age,
				   collective_living_history,
				   collective_health_history, 
				  collective_probability_history)
		println("Life expectancy in this population: ", mean(collective_age))
		
		return(results)
	end
end 

# ╔═╡ ffe1f6bb-e4b6-4f61-9468-2009244b607f
begin 
	periods 			= 110
	fixed_temperature_1 = 0.61 # 2018
	fixed_temperature_2 = 0.71 # 0.10 increase from 2018
	
	population_1 = population_simulation(N 	= 1_000,
						  T 				= periods, 
						  weather_history 	= fill(fixed_temperature_1,periods),
						  GDP 				= fill(gdp[gdp.Year .== 2020,:GDP][1], periods))
	population_2 = population_simulation(N 	= 1_000,
						  T 				= periods, 
						  weather_history 	= fill(fixed_temperature_2,periods),
						  GDP 				= fill(gdp[gdp.Year .== 2020,:GDP][1], periods))
	
end

# ╔═╡ abda697c-044b-4b42-b386-aab9226b755a
begin 
	Plots.gr()
	
	cls1 	= []
	for i in 1:length(population_1[:collective_living_history])
		tmp = population_1[:collective_living_history][i]
		push!(cls1,tmp)
	end

	Plot1 	= Plots.plot(1:periods,sum(cls1[:, 1]),
						 # legend = false, 
						 label = "Average temperature = $fixed_temperature_1")

	cls2 	= []
	for i in 1:length(population_2[:collective_living_history])
		tmp = population_2[:collective_living_history][i]
		push!(cls2,tmp)
	end

	Plots.plot!(1:periods, sum(cls2[:, 1]),
						 # legend = false,
						 label = "Average temperature = $fixed_temperature_2")
	Plots.plot!(xaxis = "Time",
		yaxis = "Population",
		title = "Evolution of population: constant temperatures")
	
end 

# ╔═╡ 410fe8b7-79a4-44b6-abfa-42f8c0e0aa60
begin 
	println("Age 80 survival: ", survival(Age=100, Health=3, Temperature=2.0, GDP=21354.0))
	println("Age 90 survival: ", survival(Age=100, Health=3, Temperature=1.0, GDP=21354.0))
	println("Age 100 survival: ", survival(Age=100, Health=3, Temperature=0.61, GDP=21354.0))
end

# ╔═╡ e3c5099c-063c-4809-943f-557cf5269691
begin 
	"""
	The function `individual_probability_simulation` allows to simulate survival probability without actually drawing a survival status. It is used in the Numerical methods solving process.

		individual_probability_simulation(;T::Integer,
								   temperature_path::Vector{Float64},
								   GDP_path::Vector{Float64})::Vector{Float64}

	This is useful for the computation of the interest rate at each period.
	
	"""
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
										Health_t_1 	= individual_past_health,
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
		
			cumulative_survival_prob = 
				cumulative_survival_prob * annual_survival

			probability_history[t] = cumulative_survival_prob
		end
		return (;probability_history,health_history)
	end
end

# ╔═╡ ad02c350-0b75-49d7-8aad-253dd61e0062
individual_probability_simulation(T = 100,
					  temperature_path = 0.61 .* ones(100),
					  GDP_path = fill(gdp[gdp.Year .== 2020, :GDP][1],100))

# ╔═╡ cc131fbc-a2d2-42af-8160-b22e32aac512
begin 
	s_range_2 			= -1.00:0.1:2.00
	sprime_range_2 		= s_range_2
	growing_temperature = collect(range(start = 0.61,
										stop = 1.00,
										length = nperiods))
	Stable_GDP 			= gdp[gdp.Year .== 2020, :GDP][1]

	survival_probability =
		individual_probability_simulation(T = nperiods, 
										  temperature_path = growing_temperature,
										  GDP_path = fill(Stable_GDP,nperiods))

	dynamic_r = ((1 .- survival_probability[:probability_history]) ./ (survival_probability[:probability_history]))

	small_r = (fill(0.2,nperiods))
	
end

# ╔═╡ 0c94843d-2572-4713-bbaf-44952f398e73
Plots.plot(1:nperiods, 
		   survival_probability[1],
		  xaxis = "Year", 
		  yaxis = "Probability", 
		  legend = false)

# ╔═╡ a98ee210-bc95-4bbc-ac68-77724de41d62
Plots.plot(1:nperiods, 
		   survival_probability[2],
		  xaxis = "Year", 
		  yaxis = "Health state",
		  legend = false)

# ╔═╡ f65052f6-d532-426e-b4cc-a755d9955ae4
begin 
	@time benchmark = backwards(s_range			= s_range_2,
							sprime_range		= s_range_2,
							consumption_range 	= 0:0.01:consumption_max,
							# labor_range			= 0.00:0.05:2,
							nperiods 			= nperiods,
							r 					= small_r,
							z 					= 1 ./ survival_probability[2],
							w 					= growing_temperature,
							proba_survival 		= survival_probability[1],
							h 					= 2 .* ones(nperiods),
							ρ 					= 1.50,
							φ 					= 2.00,
							β 					= 0.96)
end

# ╔═╡ 25b0c3e1-eca9-4e09-9d06-e302f517e485
begin 
	@time numerical = backwards_numerical(s_range = s_range_2,
							sprime_range		= s_range_2,
							consumption_range 	= 0:0.01:consumption_max,
							labor_range			= 0.00:0.05:2,
							nperiods 			= nperiods,
							r 					= small_r,
							z 					= 1 ./ survival_probability[2],
							w 					= growing_temperature,
							proba_survival 		= survival_probability[1],
							h 					= 2 .* ones(nperiods),
							ρ 					= 1.50,
							φ 					= 2.00,
							β 					= 0.96)
end

# ╔═╡ f5052cbb-d54f-4b74-a57d-959578a989f6
begin 
	plotly()

	# Approximation: 
	plot_Vstar_approximation =
		Plots.plot(s_range_2,benchmark[:Vstar][1,:], label = "Period: 1")
	
	for t in 1:nperiods
		Plots.plot!(s_range_2,benchmark[:Vstar][t,:], label = "Period: $t")
	end
	
	Plots.plot!(xaxis = "Initial savings" , yaxis = "Value function")
	
	Plots.plot!(legend = false, title = "Approximation")

	# Numerical: 

	plot_Vstar_numerical =
		Plots.plot(s_range_2,numerical[:Vstar][1,:], label = "Period: 1")
	
	for t in 1:nperiods
		Plots.plot!(s_range_2,numerical[:Vstar][t,:], label = "Period: $t")
	end
	
	Plots.plot!(xaxis = "Initial savings" , yaxis = "Value function")
	
	Plots.plot!(legend = false, title = "Numerical solution")

	Plots.plot(plot_Vstar_approximation,plot_Vstar_numerical)
	
end

# ╔═╡ 0a0cb7c1-99ac-41db-8061-480090fd44f6
begin 
	plotly()

	# Approximation model: 
	
	plot_c_star_approximation = 
		Plots.plot(s_range_2,
				   smoothing(benchmark[:optimal_choices][1,:,"c"], common_span),
				   label = "Period: 1",
				   xaxis = "Initial savings",
				   yaxis = "Consumption")
	
	for t in 1:nperiods#10:10:nperiods
		Plots.plot!(s_range_2,
					smoothing(benchmark[:optimal_choices][t,:,"c"], common_span),
					label = "Period: $t")
	end

	Plots.plot!(xaxis = "Initial savings",
				yaxis = "Optimal consumption",
				title = "Approximation", legend = false)

	# Full-Numerical solution: 

		plot_c_star_numerical = 
		Plots.plot(s_range_2,
				   smoothing(numerical[:optimal_choices][1,:,"c"], common_span),
				   label = "Period: 1",
				   xaxis = "Initial savings",
				   yaxis = "Consumption")
	
	for t in 1:nperiods#10:10:nperiods
		Plots.plot!(s_range_2,
					smoothing(numerical[:optimal_choices][t,:,"c"], common_span),
					label = "Period: $t")
	end

	Plots.plot!(xaxis = "Initial savings",
				yaxis = "Optimal consumption",
				title = "Numerical solution", legend = false)
	
	Plots.plot(plot_c_star_approximation,plot_c_star_numerical)
end

# ╔═╡ 98bda9bb-ec29-4387-ad60-411006a9f309
begin 
	consumption_1_approximation = Plots.plot(s_range_2,
				   benchmark[:optimal_choices][1,:,"c"],
				   label = "Period: 1",
				 primary = false,
				   xaxis = "Initial savings",
				   yaxis = "Consumption",
			  title = "Approximation")

	consumption_1_numerical = Plots.plot(s_range_2,
				   numerical[:optimal_choices][1,:,"c"],
				   label = "Period: 1",
				 primary = false,
				   xaxis = "Initial savings",
				   yaxis = "Consumption",
			  title = "Numerical solution")

	Plots.plot(consumption_1_approximation,consumption_1_numerical)
end

# ╔═╡ 5e5cb6db-def1-4ac0-9fb7-51584d91970b
begin 
	plotly()

	# Approximation: 

	plot_sprime_star_approximation = Plots.plot(s_range_2,
								  smoothing(benchmark[:optimal_choices][1,:,"sprime"], common_span))
	Plots.plot!(label = "Period: 1", xaxis = "Initial savings", yaxis = "Savings at next period")
	for t in 1:nperiods # [50,80,100,104]
	 	Plots.plot!(s_range_2,
					smoothing(benchmark[:optimal_choices][t,:,:][:,"sprime"], common_span), label = "Period: $t")
	end
	Plots.plot!(title = "Approximation", legend = false)

	# Numerical solution: 


	plot_sprime_star_numerical = Plots.plot(s_range_2,
								  smoothing(numerical[:optimal_choices][1,:,"sprime"], common_span))
	Plots.plot!(label = "Period: 1", xaxis = "Initial savings", yaxis = "Savings at next period")
	for t in 1:nperiods # [50,80,100,104]
	 	Plots.plot!(s_range_2,
					smoothing(numerical[:optimal_choices][t,:,:][:,"sprime"], common_span), label = "Period: $t")
	end
	Plots.plot!(title = "Numerical solution", legend = false)
	
	# Ploting both:
	Plots.plot(plot_sprime_star_approximation,plot_sprime_star_numerical)
end

# ╔═╡ d4784e37-86a4-49d5-91bb-b68dc0ec7d24
begin
	savings_2_approximation = Plots.plot(s_range_2,
			   benchmark[:optimal_choices][1,:,"sprime"],
			   label = "Period: 1",
			   xaxis = "Initial savings",
			   yaxis = "Savings at next period",
			   legend = false,
			   title = "Approximation")
	
	Plots.plot!(s_range_2,
			   benchmark[:optimal_choices][75,:,"sprime"], 
			   label = "Period 75")

	savings_2_numerical = Plots.plot(s_range_2,
			   numerical[:optimal_choices][1,:,"sprime"],
			   label = "Period: 1",
			   xaxis = "Initial savings",
			   yaxis = "Savings at next period",
			   legend = false,
			   title = "Numerical solution")
	
	Plots.plot!(s_range_2,
			   numerical[:optimal_choices][75,:,"sprime"], 
			   label = "Period 75")

	Plots.plot(savings_2_approximation,savings_2_numerical)
end

# ╔═╡ 5508a281-254d-4d67-ab3d-ad32b2ff67f3
begin 
	plotly()
	
	budget_constraint_approximation = 
		Plots.plot(s_range_2,benchmark[:budget_balance][1,:], label = "Period 1")
	
	Plots.plot!(xaxis = "Initial savings",
				yaxis = "Budget surplus",
				title = "Approximation",
				legend = false)
	
	for t in 1:nperiods
	 	Plots.plot!(s_range_2,benchmark[:budget_balance][t,:], label = "Period: $t")
	end

	budget_constraint_numerical_solution = 
		Plots.plot(s_range_2,numerical[:budget_balance][1,:], label = "Period 1")
	
	Plots.plot!(xaxis = "Initial savings",
				yaxis = "Budget surplus",
				title = "Numerical solution",
				legend = false)
	
	for t in 1:nperiods
	 	Plots.plot!(s_range_2,numerical[:budget_balance][t,:], label = "Period: $t")
	end
	
	Plots.plot(budget_constraint_approximation,budget_constraint_numerical_solution)
end


# ╔═╡ 7fde6bdb-b9cc-4ca3-827f-8aaa4f03a82a
begin 
	Plots.gr()
	budget_clearing_1_approximation = Plots.plot(s_range_2,
			   benchmark[:budget_balance][1,:],
			   label = "Period 1",
			 primary = false,
			   xaxis = "Initial savings", 
			   yaxis = "Budget surplus", 
			   title = "Approximation")

	budget_clearing_1_numerical = Plots.plot(s_range_2,
			   numerical[:budget_balance][1,:],
			   label = "Period 1",
			 primary = false,
			   xaxis = "Initial savings", 
			   yaxis = "Budget surplus", 
			   title = "Numerical solution")

	Plots.plot(budget_clearing_1_approximation,budget_clearing_1_numerical)
end

# ╔═╡ a99f953d-2df4-4f30-9ca4-d3302f56571f
begin 
	plotly()

	lstar(c) = (c ^(-ρ)*1/1)^(1/φ)

	
	plot_l_star = Plots.plot(s_range_2,
							 smoothing(lstar.(benchmark[:optimal_choices][1,:,"c"].array),
									   common_span),
							 label = "Period: 1",
							 xaxis = "Initial savings",
							 yaxis = "Labor supply")
	
	for t in 1:nperiods
		Plots.plot!(s_range_2,
					smoothing(lstar.(benchmark[:optimal_choices][1,:,"c"].array), common_span),
					label = "Period: $t")
	end

	Plots.plot!(xaxis = "Initial savings",
				yaxis = "Labor supply",
				title = "Labor supply policy",
				legend = false)

	Plots.plot(plot_l_star)
end

# ╔═╡ 46ff7619-926c-445d-abf5-7bad5878169f
begin 
	plotly()

	c1 = benchmark[:optimal_choices][1,:,"c"].array
	c2 = benchmark[:optimal_choices][2,:,"c"].array

	l1 = lstar.(c1)
	l2 = lstar.(c2)

	Plots.plot(xaxis = "Labor",
			   yaxis = "Consumption")
	
	Plots.plot!(s_range_2,
			smoothing(c1,common_span),
			label 	= "Model - period 1")
	
	c_FOC1 = (1 .* l1.^φ .* 1^(-1)) .^ (-1/ρ)

	Plots.plot!(s_range_2,
				smoothing(c_FOC1,common_span),
				label 	= "FOC 1")

	β = 0.96
	
	c_FOC2 = Array{Float64}(undef,length(c2))
	
	c_FOC2 = 
		(β *
			survival_probability[1][1] .*
			(c2 .^ (-ρ)) .*
			survival_probability[2][1] .*
			(1+small_r[2])) .^ (-1/ρ)

	Plots.plot!(s_range_2,
				smoothing(c_FOC2,common_span),
				label = "FOC 2 - Euler")

	Plots.plot!(s_range_2,
				smoothing(c2,common_span),
				label = "Model - period 2")
end

# ╔═╡ 4258e19c-4277-428a-9164-3d8615520ad0
begin 
	"""
	This function returns a probability of survival for each period, given: 

	- A number of periods,
	- A temperature path,
	- A GDP path, 
	
		probability_path(;nperiods::Integer,
			temperature_start::Float64,
			temperature_stop::Float64,
			GDP_start::Float64,
			GDP_stop::Float64)
	"""
	function probability_path(;nperiods::Integer,
							  temperature_start::Float64,
							  temperature_stop::Float64, 
							 GDP_start::Float64,
							 GDP_stop::Float64)
		
		temperature_path = range(start 	= temperature_start,
								 stop 	= temperature_stop,
								 length = nperiods)

		GDP_path = range(start 	= GDP_start,
						 stop 	= GDP_stop,
						 length = nperiods)

		survival_probabilities =
			individual_probability_simulation(T = nperiods,
											  temperature_path = collect(temperature_path),
											  GDP_path = collect(GDP_path))[:probability_history]
		return survival_probabilities
	end
end

# ╔═╡ 7e61cdcc-3764-4cb8-b426-05939fc8e966
begin 
	plotly()  # Use Plotly for 3D
	
	# Create age-temperature grid
	# age_range2 = 1:110
	temp_range = range(minimum(DF.av_annual_t), maximum(DF.av_annual_t), length=100)
	#temp_range = range(-1, 1, length = 100)
		
	age_grid = repeat(age_range', length(temp_range), 1)
	temp_grid = repeat(temp_range, 1, length(age_range))
	
	# Health categories (assuming 1=Excellent, 5=Poor)
	health_levels = [1, 2, 3, 4, 5]  
	health_labels = ["Excellent", "VeryGood", "Good", "Fair", "Poor"]
	
	# Initialize plot
	plt = plot()
	
	for (health, label) in zip(health_levels, health_labels)
	    # Predict for all age-temp combinations
	    pred_grid = [GLM.predict(model_health_age_temperature_gdp, 
	                  DataFrame(Age=a,
								Health=health,
								av_annual_t=t,
								GDP=GDP_2018,
							   Year = 2018))[1]
	                  for t in temp_range, a in age_range]
	    
	    # Add surface plot
	    surface!(plt, age_range, temp_range, pred_grid, label=label)
	end
	
	# Customize plot
	plot!(plt, 
	    title="Survival Probability by Age and Temperature",
	    xlabel="Age", 
	    ylabel="Temperature",
	    zlabel="Probability",
	    camera=(30, 45)  # Adjust viewing angle
	)
end

# ╔═╡ a12ccacc-e6be-44bf-a46f-d513a5d3376d
begin 
	Plots.gr()

	temperature_path_1 = collect(range(start = 0.61, stop = 2.00, length = periods))
	temperature_path_2 = Normal(0.61, 0.1)
	
	population_3 = population_simulation(N 	= 1_000,
						  T 				= periods, 
						  weather_history 	= temperature_path_1,
						  GDP 				= fill(GDP_2022, periods))
	population_4 = population_simulation(N 	= 1_000,
						  T 				= periods, 
						  weather_history 	= rand(temperature_path_2,100),
						  GDP 				= fill(GDP_2022, periods))
	population_5 = population_simulation(N 	= 1_000,
						  T 				= periods, 
						  weather_history 	= fill(0.01,periods),
						  GDP 				= fill(GDP_2022, periods))
	cls3 	= []
	for i in 1:length(population_3[:collective_living_history])
		tmp = population_3[:collective_living_history][i]
		push!(cls3,tmp)
	end

	Plot2 	= Plots.plot(1:periods,sum(cls3[:, 1]),
						 # legend = false, 
						 label = "From 0.61 to 2 degrees")

	cls4 	= []
	for i in 1:length(population_4[:collective_living_history])
		tmp = population_4[:collective_living_history][i]
		push!(cls4,tmp)
	end

	Plots.plot!(1:periods, sum(cls4[:, 1]),
						 # legend = false,
						 label = "Normal temperature around 0.61")
	
	cls5 	= []
	for i in 1:length(population_5[:collective_living_history])
		tmp = population_5[:collective_living_history][i]
		push!(cls5,tmp)
	end

	Plots.plot!(1:periods, sum(cls5[:, 1]),
						 # legend = false,
						 label = "Fixed 0.01")
	
	Plots.plot!(xaxis = "Time",
		yaxis = "Population",
		title = "Evolution of population: constant temperatures")

end

# ╔═╡ 7435b69e-a57a-44d2-8978-622f9f4236d3
probability_path(nperiods = nperiods,
				 temperature_start = 0.00,
				 temperature_stop = 0.00,
				 GDP_start = GDP_2022,
				 GDP_stop = GDP_2022)

# ╔═╡ 387e3100-3372-4474-86d5-a38cae652b40
begin 

	probabilities_path = Array{Array}(undef,5)
	
	probabilities_path[1] = probability_path(nperiods = nperiods,
										 temperature_start = 0.00,
										 temperature_stop = 0.00,
										 GDP_start = GDP_2022,
										 GDP_stop = GDP_2022)

	probabilities_path[2] = probability_path(nperiods = nperiods,
										 temperature_start = 0.61,
										 temperature_stop = 0.61,
										 GDP_start = GDP_2022,
										 GDP_stop = GDP_2022)

	probabilities_path[3] = probability_path(nperiods = nperiods,
										 temperature_start = 2.00,
										 temperature_stop = 2.00,
										 GDP_start = GDP_2022,
										 GDP_stop = GDP_2022)

	probabilities_path[4] = probability_path(nperiods = nperiods,
										 temperature_start = 0.61,
										 temperature_stop = 1.50,
										 GDP_start = GDP_2022,
										 GDP_stop = GDP_2022)

	probabilities_path[5] = probability_path(nperiods = nperiods,
										 temperature_start = 0.61,
										 temperature_stop = 2.50,
										 GDP_start = GDP_2022,
										 GDP_stop = GDP_2022)
	probabilities_path
end

# ╔═╡ 20d9076a-147c-44bc-8ab2-e49c8b335bd7
begin 
	temperature_path = Array{Array}(undef,5)

	temperature_path[1] = collect(range(start = 0.00, 
							   stop = 0.00,
								length = nperiods))

	
	temperature_path[2] = (range(start = 0.61, 
							   stop = 0.61,
								length = nperiods))

	temperature_path[3] = (range(start = 2.00, 
							   stop = 2.00,
								length = nperiods))

	temperature_path[4] = (range(start = 0.61, 
							   stop = 1.50,
								length = nperiods))
		
	temperature_path[5] = (range(start = 0.61, 
							   stop = 2.50,
								length = nperiods))
end

# ╔═╡ 22c34c16-cf53-41b5-8286-6a6601d6796d
temperature_path

# ╔═╡ b0c61485-0ec0-4309-992e-a472b92b3e5f
begin
	Plots.gr()
	lifetime_utility = Array{Array}(undef,5)

	plot_v_t = Plots.plot(xaxis = "Initial endowment",
			   yaxis = "Value",
			  title = "Value function and Temperature")
	
	for i in 1:5
		tmp = backwards(s_range					= s_range_2,
							sprime_range		= s_range_2,
							consumption_range 	= 0:0.01:consumption_max,
							nperiods 			= nperiods,
							r 					= dynamic_r,
							z 					= ones(nperiods),
							w 					= temperature_path[i],
							proba_survival 		= probabilities_path[i],
							h 					= 2 .* ones(nperiods),
							ρ 					= 1.50,
							φ 					= 2.00,
							β 					= 0.96)
		
		lifetime_utility[i] = tmp.Vstar.array

		temperature_minimum = minimum(temperature_path[i])
		temperature_maximum = maximum(temperature_path[i])
		Plots.plot!(s_range_2,
				   lifetime_utility[i][1,:],
				label = "From $temperature_minimum to $temperature_maximum")
	end
end

# ╔═╡ 148790d4-4824-44c9-afbb-dcf318cf6321
begin 
	Plots.gr()
	plot_v_t
end

# ╔═╡ 20b60041-fe53-424e-b482-8b5409866b49
begin 
	println(lifetime_utility[1][30,1])
	println(lifetime_utility[2][30,1])
	println(lifetime_utility[3][30,1])
end

# ╔═╡ 6f975506-2ebd-4dab-b04d-d04538a777f8
md""" Another way to visualise this change is to check the convergence of the value functions in two periods: """

# ╔═╡ b0a4bf6f-758a-442c-a6db-976ec4e23d0c
begin 
	Plots.gr()
	
	Difference = Array{Float64}(undef,nperiods)
	for t in 1:nperiods
		Difference[t] = (lifetime_utility[2][t, 1]) -
			(lifetime_utility[5][t, 1])
	end
	Plots.plot(1:nperiods,
			   Difference,
			  xaxis = "Year",
			  yaxis = "Value", 
			  title = "Difference in value function: \n 0.61 scenario vs 0.61 to 2.5", 
			  legend = false)
end

# ╔═╡ 90111432-eb2b-4049-932f-9b49659f3bb3
begin 
	Plots.gr()
	probabilities_path_plot = Plots.plot(xaxis = "Year",
			   yaxis = "Probability")
	for i in 1:length(probabilities_path)
		Plots.plot!(1:nperiods,
					probabilities_path[i],
					label = mean(temperature_path[i]))
	end
	probabilities_path_plot
end

# ╔═╡ d1ca6172-aaf5-4aa8-9eec-f398b2d153ed
md""" # Extensions 

Here a list of possible extensions, in order of increasing difficulty:

- Including international data,
- Productivity types heterogeneity,
- Health types heterogeneity,
- Inter-individuals interest rate,
- Using data to calibrate the model: savings, add bequest motivation,
- Using mortality table to assess the effect on younger individuals,
- Using data on savings.

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
Loess = "4345ca2d-374a-55d4-8d30-97f9976e7612"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
MLJModels = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
NamedArrays = "86f7a689-2022-50b4-a561-43c23ac3c673"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.10.15"
CategoricalArrays = "~0.10.8"
DataFrames = "~1.7.0"
GLM = "~1.9.0"
JLD2 = "~0.5.12"
Loess = "~0.6.4"
MLJ = "~0.20.7"
MLJLinearModels = "~0.10.0"
MLJModels = "~0.17.9"
NamedArrays = "~0.10.3"
Plots = "~1.40.11"
PlutoUI = "~0.7.23"
StatsBase = "~0.34.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "3ecf56c5d9102eca22be22e96ec4ff5740b57720"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "678eb18590a8bc6674363da4d5faa4ac09c40a18"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.5.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b5bb4dc6248fde467be2a863eb8452993e74d402"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes"]
git-tree-sha1 = "926862f549a82d6c3a7145bc7f1adff2a91a39f0"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.15"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "aa87a743e3778d35a950b76fbd2ae64f810a2bb3"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.6.52"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "0b4190661e8a4e51a842070e7dd4fae440ddb7f4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.118"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FeatureSelection]]
deps = ["MLJModelInterface", "ScientificTypesBase", "Tables"]
git-tree-sha1 = "d78c565b6296e161193eb0f053bbcb3f1a82091d"
uuid = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6"
version = "0.2.2"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "7ffa4049937aeba2e5e1242274dc052b0362157a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.14"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "98fc192b4e4b938775ecd276ce88f539bcec358e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.14+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f93655dc73d7a0b4a368e3c0bce296ae035ad76e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "e663925ebc3d93c1150a7570d114f9ea2f664726"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.4"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "59545b0a2b27208b0650df0a46b8e3019f85055b"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "1059c071429b4753c0c869b75c859c44ba09a526"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.12"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "80d268b2f4e396edc5ea004d1e0f569231c71e9e"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.34"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd10d2cc78d34c0e2a3a36420ab607b611debfbb"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.7"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "825289d43c753c7f1bf9bed334c253e9913997f8"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.9.0"

[[deps.LearnAPI]]
deps = ["InteractiveUtils", "Statistics"]
git-tree-sha1 = "ec695822c1faaaa64cee32d0b21505e1977b4809"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "0.1.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7f6be2e4cdaaf558623d93113d6ddade7b916209"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.4"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "f749e7351f120b3566e5923fefdf8e52ba5ec7f9"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.6.4"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLFlowClient]]
deps = ["Dates", "FilePathsBase", "HTTP", "JSON", "ShowCases", "URIs", "UUIDs"]
git-tree-sha1 = "9abb12b62debc27261c008daa13627255bf79967"
uuid = "64a0f543-368b-4a9a-827a-e71edb2a0b83"
version = "0.5.1"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "FeatureSelection", "LinearAlgebra", "MLJBalancing", "MLJBase", "MLJEnsembles", "MLJFlow", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "Reexport", "ScientificTypes", "StatisticalMeasures", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "521eec7a22417d54fdc66f5dc0b7dc9628931c54"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.20.7"

[[deps.MLJBalancing]]
deps = ["MLJBase", "MLJModelInterface", "MLUtils", "OrderedCollections", "Random", "StatsBase"]
git-tree-sha1 = "f707a01a92d664479522313907c07afa5d81df19"
uuid = "45f359ea-796d-4f51-95a5-deb1a414c586"
version = "0.1.5"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LearnAPI", "LinearAlgebra", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "RecipesBase", "Reexport", "ScientificTypes", "Serialization", "StatisticalMeasuresBase", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "6f45e12073bc2f2e73ed0473391db38c31e879c9"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "1.7.0"
weakdeps = ["StatisticalMeasures"]

    [deps.MLJBase.extensions]
    DefaultMeasuresExt = "StatisticalMeasures"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatisticalMeasuresBase", "StatsBase"]
git-tree-sha1 = "84a5be55a364bb6b6dc7780bbd64317ebdd3ad1e"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.4.3"

[[deps.MLJFlow]]
deps = ["MLFlowClient", "MLJBase", "MLJModelInterface"]
git-tree-sha1 = "508bff8071d7d1902d6f1b9d1e868d58821f1cfe"
uuid = "7b7b8358-b45c-48ea-a8ef-7ca328ad328f"
version = "0.5.0"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "ad16cfd261e28204847f509d1221a581286448ae"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.6.3"

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "7f517fd840ca433a8fae673edb31678ff55d969c"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.10.0"

[[deps.MLJModelInterface]]
deps = ["REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "66626f80d5807921045d539b4f7153b1d47c5f8a"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.1"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "09381923be5ed34416ed77badbc26e1adf295492"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.17.9"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase", "StatisticalMeasuresBase"]
git-tree-sha1 = "38aab60b1274ce7d6da784808e3be69e585dbbf6"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.8.8"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "b14c7be6046e7d48e9063a0053f95ee0fc954176"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.9.1"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "4abc63cdd8dd9dd925d8e879cda280bedc8013ca"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.30"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "58e317b3b956b8aaddfd33ff4c3e33199cd8efce"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg", "Scratch"]
git-tree-sha1 = "63603b2b367107e87dbceda4e33c67aed17e50e0"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "31b3b1b8e83ef9f1d50d74f1dd5f19a37a304a1f"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.12.0"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "0e1340b5d98971513bddaa6bbed470670cebbbfe"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.34"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "24be21541580495368c35a6ccef1454e7b5015be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.11"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5152abbdab6488d5eec6a01029ca6697dff4ec8f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.23"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "142ee93724a9c5d04d78df7006670a93ed1b244e"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "4d083ffec53dbd5097a6723b0699b175be2b61fb"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.1.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StatisticalMeasures]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Distributions", "LearnAPI", "LinearAlgebra", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "StatisticalMeasuresBase", "Statistics", "StatsBase"]
git-tree-sha1 = "c1d4318fa41056b839dfbb3ee841f011fa6e8518"
uuid = "a19d573c-0a75-4610-95b3-7071388c7541"
version = "0.1.7"

    [deps.StatisticalMeasures.extensions]
    LossFunctionsExt = "LossFunctions"
    ScientificTypesExt = "ScientificTypes"

    [deps.StatisticalMeasures.weakdeps]
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "e4f508cf3b3253f3eb357274fe36fb3332ca9896"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "9022bcaa2fc1d484f1326eaa4db8db543ca8c66d"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.4"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "c950ae0a3577aec97bfccf3381f66666bc416729"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.8.1+0"
"""

# ╔═╡ Cell order:
# ╠═7f5a055e-4572-490d-8126-debfa842b82b
# ╟─03b5f43c-2806-11f0-1aad-913619158eaf
# ╠═37dd945d-d760-4f94-a46e-672e61c2057e
# ╠═c2364d33-9afe-426b-ac1b-59504913c4df
# ╟─5ad9efa0-a367-4643-88e3-d0fe69d8132a
# ╠═ea58fb60-fdd3-4d00-b7fe-ceafe869adf0
# ╠═8141f03e-7a82-4b90-9a5d-427201b3be45
# ╟─2c6a9943-dbcc-44c6-8f84-7035cf920208
# ╠═69649ff8-e60a-42d8-9f55-c291d6602ca2
# ╟─5a8a26ec-4d3c-4d8e-a709-f2a4fab66801
# ╟─b28eb188-6d11-43ef-86d1-cb739dab4646
# ╠═b7f90644-d4fe-4cb7-a60e-5ab180a35d50
# ╠═2336c8fd-263f-42c2-85d4-2b5a64f11a69
# ╠═7829dd48-681f-4fa1-93f9-9107848e0a98
# ╠═babfe2e0-19b6-4a22-bc4f-74716ec50ec9
# ╠═e512be5c-ef0d-4595-b516-d9117e703153
# ╠═d4a5c3bc-4b3a-4936-adc9-68412109626c
# ╠═a32b576d-44c3-4f09-93b3-3386dd1ee1de
# ╠═96e9ea8a-129c-4ab3-a2c3-c947fe4a83b8
# ╠═0d24cd79-fe1b-47fb-8a78-872b3335b6c5
# ╠═ccfb6fc8-60a2-4b1d-a988-a4938e7c5b3b
# ╠═8a8e9034-f6be-4715-b481-2f62a554bbef
# ╟─1a625bfe-ddc5-4fb3-b006-6b70b086559c
# ╠═388af98a-fec2-45e2-897a-6e818d586745
# ╠═9bce461d-575e-4141-9cc8-a9e6ce449973
# ╟─92ddcaa0-5105-482a-bc90-385b53270e8b
# ╠═e5693cfe-19c3-4c35-a361-89a6560a7b21
# ╟─1905c0a8-18e8-4f67-8c37-5c2daf756bb0
# ╠═50251579-de6d-43f3-af36-dc459585ad22
# ╠═342da11c-cc36-4156-8d5a-e93d859e7a8c
# ╠═60194c76-1ae5-49af-8baf-0f3465c634e1
# ╠═badaea58-868b-4ef0-aee8-3c4583f7b3ea
# ╠═d7c398ec-4325-40ae-8d71-1aa306830131
# ╠═f97b719e-8b44-472b-b1ce-57eb5303ea02
# ╟─a28d672e-693f-44ba-9c38-9edf43bfff41
# ╠═7e61cdcc-3764-4cb8-b426-05939fc8e966
# ╟─038f2359-69ca-4227-bb12-9bec3d4a3ae5
# ╠═6c5c973f-9197-4114-9924-8e7a299b1472
# ╠═d7ba6c4c-0f80-453b-a893-648d80b0c9b3
# ╠═919c1aae-4e9b-489f-9950-17d11f59555c
# ╠═22019072-9fa5-40a3-bb22-89696e19a2d5
# ╠═410fe8b7-79a4-44b6-abfa-42f8c0e0aa60
# ╠═ffe1f6bb-e4b6-4f61-9468-2009244b607f
# ╟─520abb10-18f1-4747-8d72-b5749834d713
# ╠═b3dd0d43-dd39-4b2a-9f16-7ecb1f6646c9
# ╠═abda697c-044b-4b42-b386-aab9226b755a
# ╠═a12ccacc-e6be-44bf-a46f-d513a5d3376d
# ╟─1dabd52f-2e78-40a1-b8dc-d213fdf673a2
# ╟─e29e2ee8-1e85-4236-96ca-47962898e105
# ╟─b04ea071-adf7-4d2b-9ad9-2acb6338c321
# ╠═32daa090-3c9a-4d9f-b1a2-d969cf4e6966
# ╠═0dcbe2c1-2ed3-4911-bafe-09e76f269153
# ╠═e3c5099c-063c-4809-943f-557cf5269691
# ╠═ad02c350-0b75-49d7-8aad-253dd61e0062
# ╠═b55356ef-c4d8-4533-b90a-d2ba2adbbfc3
# ╠═db51d7d8-6e3c-46c6-bf80-47d72cdb097e
# ╠═2648949a-69c4-4981-912b-99f707b8b07d
# ╟─982814bb-a80e-4ad6-b981-a3a60691fdf4
# ╠═f89b93cd-111e-46fb-9364-03600edcaddf
# ╠═99fb1863-6d74-423e-8ea0-f8f4fa4d25e3
# ╠═25bf9601-ff3d-4d23-a011-ddb788e2b126
# ╠═976923cb-7132-415c-881d-f46485032623
# ╠═fbf795ba-0065-4e35-944e-36762dff99fd
# ╠═da863575-eeeb-44b1-a9d4-edd92ffb09ca
# ╟─e029f368-ec95-4317-8e87-2153ace4c123
# ╠═5c51eae5-c28e-4be1-8a5c-61ca8038f1c5
# ╠═cc131fbc-a2d2-42af-8160-b22e32aac512
# ╠═0c94843d-2572-4713-bbaf-44952f398e73
# ╠═a98ee210-bc95-4bbc-ac68-77724de41d62
# ╠═f65052f6-d532-426e-b4cc-a755d9955ae4
# ╠═25b0c3e1-eca9-4e09-9d06-e302f517e485
# ╠═96445695-d923-44bd-8c68-d89fd974dc13
# ╠═2f1d1532-dc3f-4c08-9576-f0b9c3a633bc
# ╟─d523f5d5-113b-44a0-a631-b05acc1a408a
# ╠═f5052cbb-d54f-4b74-a57d-959578a989f6
# ╟─47fab280-4f42-412d-ad35-adf8fa2e988f
# ╠═0a0cb7c1-99ac-41db-8061-480090fd44f6
# ╠═98bda9bb-ec29-4387-ad60-411006a9f309
# ╟─aef9601d-f05e-4995-937f-b1b9cb140e73
# ╠═5e5cb6db-def1-4ac0-9fb7-51584d91970b
# ╠═d4784e37-86a4-49d5-91bb-b68dc0ec7d24
# ╟─5cae56c7-c5ba-447b-a9ea-ef7f1ad99dfe
# ╠═5508a281-254d-4d67-ab3d-ad32b2ff67f3
# ╠═7fde6bdb-b9cc-4ca3-827f-8aaa4f03a82a
# ╟─5ee88736-733b-480a-b9e2-929408cef55f
# ╠═383d864d-310a-4e98-be68-48801a1c3a9d
# ╠═a99f953d-2df4-4f30-9ca4-d3302f56571f
# ╟─b9259bf1-e248-4ba7-bd9c-ae6094e368dd
# ╠═46ff7619-926c-445d-abf5-7bad5878169f
# ╟─c3acbb4e-1330-4c7d-9875-fc89d8a35453
# ╠═4258e19c-4277-428a-9164-3d8615520ad0
# ╠═7435b69e-a57a-44d2-8978-622f9f4236d3
# ╠═6502e5f8-56b4-4f57-8851-91b826c45bbf
# ╠═387e3100-3372-4474-86d5-a38cae652b40
# ╠═20d9076a-147c-44bc-8ab2-e49c8b335bd7
# ╠═22c34c16-cf53-41b5-8286-6a6601d6796d
# ╠═b0c61485-0ec0-4309-992e-a472b92b3e5f
# ╠═148790d4-4824-44c9-afbb-dcf318cf6321
# ╠═20b60041-fe53-424e-b482-8b5409866b49
# ╟─6f975506-2ebd-4dab-b04d-d04538a777f8
# ╠═b0a4bf6f-758a-442c-a6db-976ec4e23d0c
# ╠═90111432-eb2b-4049-932f-9b49659f3bb3
# ╟─d1ca6172-aaf5-4aa8-9eec-f398b2d153ed
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
