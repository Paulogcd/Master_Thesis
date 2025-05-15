using Latexify

# performance_full_numeric                = [i for i in performance_full_numeric]
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