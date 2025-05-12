begin 
    using Statistics
    using CSV
    using DataFrames
    using Plots
end

pwd() 

temperature = CSV.read("working_elements/Empirical_models/temperature.csv", DataFrame)

# Create average variables: 
# begin 
#     temperature[:,:annual_average] = (temperature[:,"Annual anomaly (1)"] .+ temperature[:,"Annual anomaly (2)"]) ./ 2
#     temperature[:,:five_annual_average] = (temperature[:,"Five years anomaly (1)"] .+ temperature[:,"Five years anomaly (2)"]) ./ 2
#     temperature[:,:ten_annual_average] = (temperature[:,"Ten years anomaly (1)"] .+ temperature[:,"Ten years anomaly (2)"]) ./ 2    
# end

# Average months, to get value per year:
# begin
#     t1 = combine(groupby(temperature, :Year),
#     [:annual_average, :five_annual_average] .=> mean .=> [:av_annual_t, :av_5_annual_t])
#     t1 = DataFrame(t1)
# 
#     t2 = combine(groupby(temperature, :Year),
#     ["Annual anomaly (1)", "Annual anomaly (2)"] .=> mean .=> ["Annual_lower_anomaly", "Annual_upper_anomaly"])
#     t2 = DataFrame(t2)
#     
#     temperature = leftjoin(t1,t2, on = :Year)
# end

tp = temperature[temperature.Year .>= 1900, :]

# Plots.plot(tp.Year, tp.av_annual_t)

mid = (tp.Annual_lower_anomaly .+
    tp.Annual_upper_anomaly) ./ 2   #the midpoints (usually representing mean values)
w = (tp.Annual_upper_anomaly .- tp.Annual_lower_anomaly) ./ 2     #the vertical deviation around the means

tp_plot =plot(tp.Year,
    tp.av_annual_t,
    ribbon = w ,
    fillalpha = 0.35,
    c = 1, lw = 2,
    legend = false, #:topleft,
    label = "Mean")

    plot!(xaxis = "Year", yaxis = "Temperature")

savefig("working_elements/Draft/output/figure_1.png")
