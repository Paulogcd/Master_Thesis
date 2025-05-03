using Pkg
Pkg.activate(".")
using Plots
using NamedArrays

begin
	using PlutoUI
	using Interpolations
	using Roots
	using LaTeXStrings
	using Plots
	# using Optim
	
end

Pkg.add("Interpolations")
Pkg.add("Roots")
Pkg.add("LaTeXStrings")
# Pkg.add("Optim")

"""
	Bellman(grid::Vector,vplus::Vector,π::Float64,yvec::Vector)

Given a grid and a next period value function `vplus`, and a probability distribution
calculate current period optimal value and actions.
"""
function Bellman(grid::Vector,vplus::Vector,π::Float64,yvec::Vector{Float64},β::Float64)
	points = length(grid)
	w = zeros(points) # temporary vector for each choice or R'
	Vt = zeros(points) # optimal value in T-1 at each state of R
	ix = 0 # optimal action index in T-1 at each state of R
	at = zeros(points) # optimal action in T-1 at each state of R

	for (ir,r) in enumerate(grid) # for all possible R-values
		# loop over all possible action choices
		for (ia,achoice) in enumerate(grid)
			if r <= achoice   # check whether that choice is feasible
				w[ia] = -Inf
			else
				rlow = r - achoice + yvec[1] # tomorrow's R if y is low
				rhigh  = r - achoice + yvec[2] # tomorrow's R if y is high
				jlow = argmin(abs.(grid .- rlow))  # index of that value in Rspace
				jhigh = argmin(abs.(grid .- rhigh))  # index of that value in Rspace
				w[ia] = sqrt(achoice) + β * ((1-π) * vplus[jlow] + (π) * vplus[jhigh] ) # value of that achoice
			end
		end
		# find best action
		Vt[ir], ix = findmax(w) # stores Value und policy (index of optimal choice)
		at[ir] = grid[ix]  # record optimal action level
	end
	return (Vt, at)
end

begin
	points = 500
	lowR = 0.01
	highR = 10.0
	# more points towards zero to make nicer plot
	Rspace = exp.(range(log(lowR), stop = log(highR), length = points))
	aT = Rspace # consume whatever is left
	VT = sqrt.(aT)  # utility of that consumption
	yvec = [1.0, 3.0]
	nperiods = 10
	β = 1.0  # irrelevant for now
	π = 0.7
end

# identical
function backwards(grid, nperiods, β, π, yvec)
	points = length(grid)
	V = zeros(nperiods,points)
	c = zeros(nperiods,points)
	V[end,:] = sqrt.(grid)  # from before: final period
	c[end,:] = collect(grid)

	for it in (nperiods-1):-1:1
		x = Bellman(grid, V[it+1,:], π, yvec, β)	
		V[it,:] = x[1]
		c[it,:] = x[2]
	end
	return (V,c)
end

V,a = backwards(Rspace, nperiods, β, π, yvec);

let
	cg = cgrad(:viridis)
    cols = cg[range(0.0,stop=1.0,length = nperiods)]
	pa = plot(Rspace, a[1,:], xlab = "R", ylab = "Action",label = L"a_1",leg = :topleft, color = cols[1])
	for it in 2:nperiods
		plot!(pa, Rspace, a[it,:], label = L"a_{%$(it)}", color = cols[it])
	end
	pv = plot(Rspace, V[1,:], xlab = "R", ylab = "Value",label = L"V_1",leg = :bottomright, color = cols[1])
	for it in 2:nperiods
		plot!(pv, Rspace, V[it,:], label = L"V_{%$(it)}", color = cols[it])
	end
	plot(pv,pa, layout = (1,2))
end

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


logistic_1(;K,a,r,t,ζ_3,h) = K./(1 .+ a .* exp.(-r.*(t.-ζ_3).*(1 .+1(h=="b"))))
probability_of_dying(age,h,w) = 1/2*()

age = 0:1:100
Plots.plot(logistic_1.(K = 1, a = 1, r = 1,t = age,ζ_3 = 50, h = "b"))

logistic_2(;K,a,r,t,ζ_4,w) = K./(1 .+ a .* exp.(-r.*(t.-ζ_4).+abs.(w)))

weather = range(start = -5, stop = 5, length = length(age)) # -5:1:5

abs.(weather)

age_range = range(start = 1, stop = 100, length = 100)
temp_range = range(start = -10, stop = 10, length = 100)
age_grid = repeat(reshape(collect(age_range), :, 1), 1, length(temp_range))
temp_grid = repeat(reshape(collect(temp_range), 1, :), size(age_grid, 1), 1)

plotly()
Plots.plot(age_grid,
			temp_grid,
			logistic_2.(K = 1, a = 1, r = 1,t = age_grid,ζ_4 = 50, w = temp_grid),
			st=:surface)

logistic_2.(K = 1, a = 1, r = 1,t = x,ζ_4 = 50, w = weather)


using Integrals

begin
	"""
	The logistic_1 function is the first intermediary death probability.
	
	`logistic_1(;K,a,r,t,E1,h) = K./(1 .+ (1+1(h=="g")).*a .* exp.(-r.*(t.-E1)))`
	
	"""
	logistic_1(;K,a,r,t,E1,h,Δh) = (K./(1 .+ (1+Δh*1(h=="g")).*a .* exp.(-r.*(t.-E1))))
end

begin 
	"""
	The logistic_2 function is the second intermediary death probability.
		
	`logistic_2(;K,a,r,t,E2,w) = K./(1 .+ a .* exp.(-r.*(t.-E2).-abs.(w)))`

	"""
	logistic_2(;K,a,r,t,E2,w) = K./(1 .+ a .* exp.(-r.*(t.-E2).-abs.(w)))
end

begin 
	"""
	ζ is the death probability function. It takes this syntax form: 

		ζ(;a1,a2,t,h,w,E1,E2,ζ1,r, Δh)
	
	"""
	ζ(;a1,a2,t,h,w,E1,E2,ζ1,r,Δh) = 
		1/2 * (ζ1*logistic_1.(K = 1, a = a1, r = r, t = t, E1 = E1, h = h, Δh = Δh) 	
			.+ 
			(1-ζ1) * logistic_2.(K=1,a = a2, r = r, t = t, E2 = E2, w = w))
end

# Redefine ζ_integral to match Integrals.jl's expected format
function ζ_integral(u, p)
    t, w = u  # Unpack integration variables
    (; h, a1, a2, E1, E2, ζ1, r, Δh) = p  # Unpack parameters
    
    # Call your original ζ function
    return ζ(; a1, a2, t, h, w, E1, E2, ζ1, r, Δh)
end

# Set up integration
tmin, tmax = 0, Inf
wmin, wmax = -Inf, Inf
a1_final = 1
a2_final = 20
E = 82
r_final = 0.1
Δh_final = 20

# Define parameters (including h)
test_parameters = (
    h = "good",  # Include h in parameters
    a1 = a1_final,
    a2 = a2_final,
    E1 = E,
    E2 = E,
    ζ1 = 0.5,
    r = r_final,
    Δh = Δh_final
)

# Create integration problem
domain = ([tmin, wmin], [tmax, wmax])
prob = IntegralProblem(ζ_integral, domain..., test_parameters)  # Note the ... for domain

# Solve
sol = solve(prob, HCubatureJL(); reltol=1e-3, abstol=1e-3)
integral_value = sol.u

a = rand(10)

b = rand(10)



c = NamedArray(zeros(10,10,10,10))

names(c)

Array{Number}(undef,length(1:10))

#[1] = ["First name"]

# ?NamedArray

# Assuming solution[:V] is your 21×201×201×21 Array{Number, 4}

V = Array{Number, 4}(undef,21,201,201,21)

V = zeros(21,201,201,21)
nothing

V = solution[:V]

# Define the dimension names
param1_names = ["Time period $i" for i in 1:21]
param2_names = ["Consumption_index_$i" for i in 1:201]
param3_names = ["Labor_supply_index_$i" for i in 1:201]
param4_names = ["Savings_next_period_index_$i" for i in 1:21]

using NamedArrays
# Create the named array
named_V = NamedArray(V, (param1_names, param2_names, param3_names, param4_names))

####################################################################################

using Pkg
Pkg.add("XLSX")
import XLSX
using CSV
using DataFrames
# Loading the data: 
mortality_table = CSV.read("working_elements/mortality_tables.csv", DataFrame)
# Data cleaning: 
mortality_table = rename(mortality_table,
["Age","Survival_men","Expectancy_men","Survival_women","Expectancy_women","Survival_both","Expectancy_both"])
mortality_table = mortality_table[Not(1),:]
mortality_table = Number(mortality_table[:,"Survival_both"])./100_000
mortality_table
Pkg.add("GLM")
using GLM
using Random 

mortality_table[:,"Survival_both"] = parse.(Float64, mortality_table[:,"Survival_both"])


model = glm(@formula(Survival_both ~ Age), mortality_table)

model = lm(@formula(Survival_both ~ Age), mortality_table)

