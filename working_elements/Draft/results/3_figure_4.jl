include("../../Environment/Final/sandbox_IV_simulation.jl")

using Plots
using Latexify

Plots.gr()
default(fontfamily = "Times")

fixed_temperature_1 = 0.00 # 1950
fixed_temperature_2 = 0.61 # 2018
fixed_temperature_3 = 1.50 # Previous scenario
fixed_temperature_4 = 2.00 # Good scenario at 2100
fixed_temperature_5 = 4.00 # Bad scenario at 2100

function produce_population(temperature)
    result = population_simulation(N = 10_000,
                T = 100, 
                weather_history = fill(temperature,100))
    return (;result,temperature)
end

# pop_1_1 = produce_population(fixed_temperature_1)
pop_2_1 = produce_population(fixed_temperature_2)
pop_3_1 = produce_population(fixed_temperature_3)
pop_4_1 = produce_population(fixed_temperature_4)
pop_5_1 = produce_population(fixed_temperature_5)

# keys(pop_2_1)
# keys(pop_2_1.result)
# pop_2_1.result.collective_health_history
# mean(pop_2_1.result.collective_health_history)

function produce_graph(populations)
    
    result_plot = Plots.plot(xaxis = "Time",
    yaxis = "Population")
    
    for population in populations
        temperature = unique(population.temperature)[1]
        cls 	= []
        for i in 1:length(population.result[:collective_living_history])
            tmp = population.result[:collective_living_history][i]
            push!(cls,tmp)
        end

        Plots.plot!(1:100,sum(cls[:, 1]),
            # legend = false, 
            label = "Annual Temperature = $temperature ", 
            linewidth = 5)
    end
    Plots.plot!(legend = :bottomleft, 
        size = (2400, 1600),
        legendfontsize = 24,
        guidefontsize = 28,
        tickfontsize = 20,

        bottom_margin = 100Plots.px,
        top_margin = 100Plots.px,
        left_margin = 100Plots.px, 
        color = ["green", "blue", "orange","red"])
    return(result_plot)
end

populations_1 = [pop_2_1,
    pop_3_1,
    pop_4_1,
    pop_5_1]
figure_3 = produce_graph(populations_1)

# pwd()
savefig("working_elements/Draft/output/figure_3.png")

function produce_life_expectancy_table(populations)
    
    results = DataFrame(Temperature = [],
        Life_expectancy = [])

    for population in populations 

        temperature = unique(population.temperature)[1]
        life_expectancy = population.result.life_expectancy
        to_add = DataFrame(Temperature = temperature,
        Life_expectancy = life_expectancy)
        results = vcat(to_add,results)

    end
    return(results)
end

life_expectancy_table_1 = produce_life_expectancy_table(populations_1)
rename!(life_expectancy_table_1, ["Temperature","Life expectancy"])
life_expectancy_table_1[:,:] = round.(life_expectancy_table_1[:,:], digits = 2)
latex_table = latexify(life_expectancy_table_1, env=:table, booktabs=true) 
println(latex_table)

function produce_population(;max_temperature, min_temperature)
    result = population_simulation(N = 10_000,
                T = 100, 
                weather_history = collect(range(start = min_temperature, stop = max_temperature, length = 100)))
    return (;result,max_temperature,min_temperature)
end

pop_1_2 = produce_population(min_temperature = 0.00, 
    max_temperature = 1.5)
pop_2_2 = produce_population(min_temperature = 0.5,
    max_temperature = 2)
pop_3_2 = produce_population(min_temperature = 0.5,
    max_temperature = 4)

populations_2 = [pop_1_2,
    pop_2_2,
    pop_3_2]

function produce_life_expectancy_table_range(populations)
    
    results = DataFrame(Temperature = [],
        Life_expectancy = [])

    for population in populations 

        max_temperature = (population.max_temperature)[1]
        min_temperature = (population.min_temperature)[1]

        life_expectancy = population.result.life_expectancy
        
        to_add = DataFrame(Temperature = "$min_temperature - $max_temperature",
            Life_expectancy = life_expectancy)
        
        results = vcat(to_add,results)

    end
    return(results)
end

populations_2

life_expectancy_table_2 = produce_life_expectancy_table_range(populations_2)
rename!(life_expectancy_table_2, ["Temperature","Life expectancy"])
life_expectancy_table_2[:,"Life expectancy"] = round.(life_expectancy_table_2[:,"Life expectancy"], digits = 2)
latex_table = latexify(life_expectancy_table_2, env=:table, booktabs=true) 
println(latex_table)

life_expectancy_table = vcat(life_expectancy_table_1,
    life_expectancy_table_2)
latex_table = latexify(life_expectancy_table, env=:table, booktabs=true) 
println(latex_table)
