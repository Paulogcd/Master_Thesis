# include("numerical_algorithms_0.jl")

using NamedArrays

function Bellman_FOC_approximation_1(;s_range::AbstractRange,
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

function backwards_FOC_approximation_1(;s_range::AbstractRange,
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
    last_Bellman = Bellman_FOC_approximation_1(s_range 		= s_range::AbstractRange,
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
        
        tmp = Bellman_FOC_approximation_1(s_range 				= s_range,
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

