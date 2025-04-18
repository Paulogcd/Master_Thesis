begin 
    using Pkg
    # Pkg.add("NamedArrays")
    # Pkg.add("Interpolations")
    # Pkg.add("ProgressBars")
    # Pkg.add("JLD2")
    # Pkg.add("FileIO")
    using NamedArrays
    using Interpolations
    using ProgressBars
    using JLD2
    using FileIO
end


# Initialisation of function
begin 
"""
The `budget_surplus` function computes the budget states for certain levels of consumption, labor supply, productivity, and savings.

Its syntax is:
    
    budget_surplus(;c,l,sprime,s,z,r=((1-0.96)/0.96)) 

"""
function budget_surplus(;c::Float64,
        l::Float64,
        sprime::Float64,
        s::Float64,
        z::Float64,
        r = ((1-0.96)/0.96)::Float64)::Float64
    return (l*z + s*(1+r) - c - sprime)::Float64
end

""" 
The `ξ` function returns the disutility of work in the utility function.

Its syntax is: 
    
    ξ(w,h)

It returns: 

    (1+abs(w))*(1+1(h=="bad"))
"""
function ξ(;w::Float64,h::AbstractString)::Float64
    return ((1 + abs(w)) * (1+1(h=="bad")))::Float64
end


"""
The `utility` function is defined such that its syntax is:

    utility(;c,l,z,w,h,ρ=1.5,φ=2)

It returns:

    (abs(c)^(1-ρ))/(1-ρ) - ξ(w,h) *((abs(l)^(1+φ)/(1+φ)))


"""
function utility(;c::Float64,
                    l::Float64,
                    w::Float64,
                    h::AbstractString,
                    ρ = 1.5::Float64,
                    φ = 2::Float64)::Float64
    return ( ((abs(c))^(1-ρ)) / (1-ρ) ) - ξ(w=w,h=h) * ( ((abs(l))^(1+φ)) / (1+φ) )::Float64
end


"""
The `Bellman` function is not to be used alone, but with the `backwards` function.

    function Bellman(;s_range::AbstractRange,
                    sprime_range::AbstractRange,
                    consumption_range::AbstractRange,
                    labor_range::AbstractRange,
                    value_function_nextperiod::Array,
                    β = 0.9, 
                    z = 1,
                    ρ = 1.5,
                    φ = 2,
                    proba_survival = 0.9,
                    r = ((1-β)/β),
                    w = 0,
                    h = "good",
                    return_full_grid = true, 
                    return_budget_balance = 1)::NamedTuple

"""
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
                    r = ((1-β)/β)::Float64,
                    w = 0::Float64,
                    h = "good"::AbstractString,
                    return_full_grid = true::Bool, 
                    return_budget_balance = true::Bool)::NamedTuple

    @assert length(value_function_nextperiod) == length(s_range) "The value function of the next period has the wrong length."

    # Initialise a grid of the value function for all possible 
    # combinations of choice and state variables:
    grid_of_value_function = Array{Number}(undef,length(s_range),
                                                        length(consumption_range),
                                                        length(labor_range),
                                                        length(sprime_range))

    # choice_variables = [consumption,labor_supply,sprime]
    # state_variables = [s,z,ξ("g",0)]
    
    # Initialise empty array that will store the optimal utility and choice:
    Vstar = zeros(length(s_range))
    index_optimal_choice = Array{CartesianIndex}(undef,length(s_range))
    optimal_choice = Array{Float64}(undef,length(s_range),3)

    if return_budget_balance == true
        budget_balance = Array{Float64}(undef,length(s_range))
    end

    # for all levels of endowment
    for (index_s,s) in enumerate(s_range)
        # for all levels of consumption
        for (index_consumption,consumption) in enumerate(consumption_range) 
            # for all levels of labor
            for (index_labor,labor) in enumerate(labor_range)
                # for all levels of savings
                for (index_sprime,sprime) in enumerate(sprime_range)

                    # If the budget constraint is violated,
                    # set the value function to minus infinity : 
                    tmp = budget_surplus(c = consumption, l = labor, sprime = sprime, s = s, z = z, r = r)
                    if tmp < 0 
                        
                        grid_of_value_function[index_s,
                                        index_consumption,
                                        index_labor, 
                                        index_sprime] = -Inf

                    # If the budget constraint is not violated,
                    # set the value function to the utility plus the value 
                    # function at the next period : 
                        
                    else

                        # Use interpolated value function (evaluated at sprime)
                        continuation_value = value_function_nextperiod(sprime)
                        
                        grid_of_value_function[index_s,
                                        index_consumption,
                                        index_labor, 
                                        index_sprime] =
                            utility(c=consumption,
                                l=labor,
                                w=w,
                                h=h,
                                ρ=ρ,
                                φ=φ) + β*proba_survival*continuation_value
                        
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

        itp = linear_interpolation(s_range, Vstar)
        
        optimal_choice[index_s,:] = [consumption_range[index_optimal_choice[index_s][1]],
                                    labor_range[index_optimal_choice[index_s][2]],
                                    sprime_range[index_optimal_choice[index_s][3]]]

        budget_balance[index_s] = budget_surplus(
                        c = consumption_range[index_optimal_choice[index_s][1]],
                        l = labor_range[index_optimal_choice[index_s][2]],
                        sprime = sprime_range[index_optimal_choice[index_s][3]],
                        s = s,
                        z = z)

    end # end of s

    # Formatting the output for better readability:
    
    # Transforming the grid_of_value_function array into a Named Array:

    param1_names 			= ["s_i$i" for i in 1:length(s_range)]
    savings_value 			= ["s=$i" for i in s_range]
    consumption_value 		= ["c=$i" for i in consumption_range]
    labor_value 			= ["l=$i" for i in labor_range]
    sprime_value 			= ["l=$i" for i in sprime_range]
    
    grid_of_value_function = NamedArray(grid_of_value_function, (savings_value, consumption_value, labor_value, sprime_value))

    optimal_choice = NamedArray(optimal_choice, (param1_names, ["c","l","sprime"]))

    # Returning results:
    if return_full_grid == true & return_budget_balance == true
        return (;grid_of_value_function,Vstar,index_optimal_choice,optimal_choice,budget_balance)
    elseif return_full_grid == true & return_budget_balance == false
        return (;grid_of_value_function,Vstar,index_optimal_choice,optimal_choice)
    elseif return_full_grid == false & return_budget_balance == true
        return (;Vstar,index_optimal_choice,optimal_choice,budget_balance)
    elseif return_full_grid == false & return_budget_balance == false
        return (;Vstar,index_optimal_choice,optimal_choice)
    end
    
end

"""
    function backwards(;s_range::AbstractRange,
            sprime_range::AbstractRange,
            consumption_range::AbstractRange,
            labor_range::AbstractRange,
            nperiods::Integer,
            z = ones(nperiods)::Array,
            β = 0.9::Float64,
            r = final_r::Array,
            ρ = 1.5::Float64, 
            φ = 2::Float64,
            proba_survival = 0.9::Float64,
            w = 0::Float64,
            h = "good"::AbstractString, 
            return_full_grid = false::Bool, 
            return_budget_balance = true::Bool)::NamedTuple
"""
function backwards(;s_range::AbstractRange,
            sprime_range::AbstractRange,
            consumption_range::AbstractRange,
            labor_range::AbstractRange,
            nperiods::Integer,
            z = ones(nperiods)::Array,
            β = 0.9::Float64,
            r = final_r::Array,
            ρ = 1.5::Float64, 
            φ = 2::Float64,
            proba_survival = 0.9::Float64,
            w = 0::Float64,
            h = "good"::AbstractString, 
            return_full_grid = false::Bool, 
            return_budget_balance = true::Bool, 
            save = false::Bool)::NamedTuple

    # Initialisation: 

    # We define he name of the variables for the named arrays: 
    param1_names 			= ["t_$i" for i in 1:nperiods]
    choice_variable_name 	= ["c","l","sprime"]
                            
    savings_value 			= ["s=$i" for i in s_range]
    consumption_value 		= ["c=$i" for i in consumption_range]
    labor_value 			= ["l=$i" for i in labor_range]
    sprime_value 			= ["l=$i" for i in sprime_range]

    # From the given ranges, construct a grid of all possible values, 
    # And save its size: 
    grid_of_value_function 	= Array{Number}(undef,length(s_range),
                                                        length(consumption_range),
                                                        length(labor_range),
                                                        length(sprime_range))
    points 					= size(grid_of_value_function)

    if return_budget_balance == true
        budget_balance = Array{Number}(undef,nperiods,length(s_range))
    end
    
    # Initialize empty arrays that will:
    # contain the values of the value function (V): 
    V = zeros(nperiods,points[1],points[2],points[3],points[4])

    # the values at optimum (Vstar), 
    Vstar = zeros(nperiods,points[1])
    
    # The indices of optimal choice (index_optimal_choices),
    index_optimal_choices = Array{Array{CartesianIndex{3}}}(undef,nperiods)

    # and the values of choice variables at the optimum (optimal_choices): 
    optimal_choices 	= Array{Number}(undef,nperiods,length(sprime_range),3) # Time periods, level of initial savings, choice variables
    optimal_choices 	= NamedArray(optimal_choices,(param1_names,savings_value,choice_variable_name))

    itp_Vlast = linear_interpolation(s_range, zeros(length(s_range)), extrapolation_bc=Line())

    # First, we solve for the last period, in which the value function of next period is 0: 
    last_Bellman = Bellman(s_range = s_range::AbstractRange,
                    sprime_range = sprime_range::AbstractRange,
                    consumption_range = consumption_range::AbstractRange,
                    labor_range = labor_range::AbstractRange,
                    value_function_nextperiod = itp_Vlast,
                    β = β,
                    ρ = ρ, 
                    φ = φ,
                    r = r[nperiods],
                    proba_survival = proba_survival,
                    w = w,
                    h = h,
                    z = z[nperiods],
                    return_full_grid = true,
                    return_budget_balance = return_budget_balance)::NamedTuple

    if save == true
        # Save the last Bellman: 
        @save "results_Bellman/Bellman_$nperiods.jld2" last_Bellman
    end
    
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
    
    for index_time in ProgressBar((nperiods-1):-1:1)
        # Interpolate last period's Vstar to use in Bellman
        itp_Vnext =
            linear_interpolation(s_range,
            last_Bellman[:Vstar],
            extrapolation_bc = Line())
        
        tmp = Bellman(s_range=s_range,
                sprime_range=sprime_range,
                consumption_range=consumption_range,
                labor_range=labor_range,
                value_function_nextperiod = itp_Vnext, # Interpolated value
                β = β,
                z = z[index_time],
                ρ = ρ,
                φ = φ,
                r = r[index_time], 
                proba_survival = proba_survival, 
                w = w,
                h = h,
                return_full_grid = true,
                return_budget_balance = return_budget_balance)::NamedTuple
        
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

        if save == true
            @save "results_Bellman/Bellman_$index_time.jld2" last_Bellman
        end
        
    end # end of time loop

    # Rename in NamedArrays:
    Vstar 			= NamedArray(Vstar, (param1_names,savings_value))
    if return_budget_balance == true
        budget_balance 	= NamedArray(budget_balance, (param1_names,savings_value))
    end
    if return_full_grid == true
        V 		= NamedArray(V, (param1_names, savings_value, consumption_value, labor_value, sprime_value))
    end
    

    if return_full_grid == true && return_budget_balance == true
        return (;V,Vstar,index_optimal_choices,optimal_choices,budget_balance)
    elseif return_full_grid == true && return_budget_balance == false
        return (;V,Vstar,index_optimal_choices,optimal_choices)
    elseif return_full_grid == false && return_budget_balance == true
        return (;Vstar,index_optimal_choices,optimal_choices,budget_balance)
    else 
        return (;Vstar,index_optimal_choices,optimal_choices)
    end

end

end

begin 
    s_range 		= 0.00000000:0.1:2
	sprime_range 	= 0.00000000:0.1:2
	consumption_max = 10
	nperiods = 104
	ζ = (1 ./(1:nperiods))
	r_pi0 = (1 .- ζ) ./ζ
	# r = (1:nperiods).^2
	# r = fill(10,nperiods)
	r_min = ((1-0.96)/(0.96))
	r = r_min .+ r_pi0
	# r = zeros(nperiods)
	typical_productivity = [exp(0.1*x - 0.001*x^2) for x in 1:nperiods]
    
    @time let benchmark = backwards(s_range			= s_range,
        sprime_range		= sprime_range,
        consumption_range 	= 0:0.1:consumption_max,
        labor_range			= 0.00:0.1:1.4,
        nperiods 			= nperiods,
        r 					= r,
        z 					= typical_productivity,
        w 					= 0.00,
        h 					= "good",
        ρ 					= 1.5,
        φ 					= 2.00,
        β 					= 0.96)
    end # Mac Mini: 10.327775 seconds (271.74 M allocations: 10.794 GiB, 15.61% gc time) 

    @time let benchmark2 = backwards(s_range			= s_range,
        sprime_range		= sprime_range,
        consumption_range 	= 0:0.1:consumption_max,
        labor_range			= 0.00:0.1:1.4,
        nperiods 			= nperiods,
        r 					= r,
        z 					= typical_productivity,
        w 					= 0.00,
        h 					= "good",
        ρ 					= 1.5,
        φ 					= 2.00,
        β 					= 0.96, 
        save                = true)
    end # Mac Mini: 249.805237 seconds (2.29 G allocations: 88.475 GiB, 3.63% gc time, 0.01% compilation time)
end

