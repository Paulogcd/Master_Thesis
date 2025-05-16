using Latexify


# Full numeric: 
performance_full_numeric = @timed let 
    pure_numerical_no_interpolation = backwards_numerical(s_range = s_range_2,
                        sprime_range		= s_range_2,
                        consumption_range 	= consumption_range,
                        labor_range			= labor_range,
                        nperiods 			= T,
                        r 					= small_r,
                        z 					= 36 .* ones(T),
                        w 					= weather_path_intermediate,
                        proba_survival 		= average_proba_intermediate,
                        h 					= average_health_intermediate,
                        ρ 					= 1.50,
                        φ 					= 2.00,
                        β 					= β)
end

performance_full_numeric = performance_metrics(performance_full_numeric)

# performance_full_numeric                = [i for i in performance_full_numeric]

# FOC 1 approximation: 

approximation_FOC_1_no_interpolation = @timed let 
    approximation_FOC_1_no_interpolation = backwards_FOC_approximation_1(s_range = s_range_2,
                        sprime_range		= s_range_2,
                        consumption_range 	= consumption_range,
                        # labor_range			= labor_range,
                        nperiods 			= T,
                        r 					= small_r,
                        z 					= ones(T),
                        w 					= weather_path_intermediate,
                        proba_survival 		= average_proba_intermediate,
                        h 					= average_health_intermediate,
                        ρ 					= 1.50,
                        φ 					= 2.00,
                        β 					= 0.96)
end

approximation_FOC_1_no_interpolation = performance_metrics(approximation_FOC_1_no_interpolation)
approximation_FOC_1_no_interpolation

# approximation_FOC_1_no_interpolation    = [i for i in approximation_FOC_1_no_interpolation]
# approximation_FOC_2_no_interpolation    = [i for i in approximation_FOC_2_no_interpolation]

Algorithms = [performance_full_numeric,
    approximation_FOC_1_no_interpolation,
    approximation_FOC_2_no_interpolation]

# keys(performance_full_numeric)

Names = ["Pure Numerical Value Function Iteration", 
    "FOC approximation 1 (fixing labor supply)", 
    "FOC approximation 2 (fixing consumption)"]

Error = [i.Error for i in Algorithms]

Time = [i.Time for i in Algorithms]

Memory = [i.Memory for i in Algorithms]

result = DataFrame(Algorithm = Names,
    Error = Error, 
    Time = Time, 
    Memory = Memory)

rename!(result, ["Algorithm", "Error", "Time (in seconds)", "Memory (in Mb)"])

result[:,Not("Algorithm")] .= round.(result[:,Not("Algorithm")], digits = 4)

# result

latex_table = latexify(result, env=:table, booktabs=true, latex = false) 
println(latex_table)