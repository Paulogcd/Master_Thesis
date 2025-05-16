begin 
    using Statistics
    using CSV
    using DataFrames
    using Plots
    default(fontfamily = "Times")
end

pwd() 

default(fontfamily = "Times")
temperature = CSV.read("working_elements/Empirical_models/temperature.csv", DataFrame)

tp = temperature[temperature.Year .>= 1900, :]

mid = (tp.Annual_lower_anomaly .+
    tp.Annual_upper_anomaly) ./ 2   #the midpoints (usually representing mean values)
w = (tp.Annual_upper_anomaly .- tp.Annual_lower_anomaly) ./ 2     #the vertical deviation around the means

tp_plot = Plots.plot(tp.Year,
    tp.av_annual_t,
    ribbon = w ,
    fillalpha = 0.35,
    c = 1,
    lw = 2,
    legend = false, #:topleft,
    label = "Mean", 
    
    # line_z = tp.av_annual_t,
    # cgrad = :RdYlGn_10,

    size = (2400, 1600),
    legendfontsize = 24,
    guidefontsize = 40,
    tickfontsize = 40,

    bottom_margin = 100Plots.px,
    top_margin = 100Plots.px,
    left_margin = 100Plots.px,

    fontfamily = "Times")

plot!(xaxis = "Year", yaxis = "Temperature")

savefig("working_elements/Draft/output/figure_1.png")
