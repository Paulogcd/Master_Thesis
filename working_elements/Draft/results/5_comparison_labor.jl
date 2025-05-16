
using Plots

default(fontfamily = "Times")

# plotly()
Plots.gr()

# Labor: 
function plot_labor(solutions, colors)
    
    plot = Plots.plot(xaxis = "Initial savings",
                yaxis = "Consumption")

    for (index,solution) in enumerate(solutions)

        let avT = 100
                low = avT - 79
                tmp = 0
                for t in low:avT
                    tmp = tmp .+
                        solution[:optimal_choices][t,:,"l"].array .* scaling_factor
                end
                tmp = tmp ./ (T-low)
            Plots.plot!(s_range_scaled,tmp,
                # label = "$low, $avT",
                legend = false,
                linewidth = 5,
                color = colors[index])
        end

        Plots.plot!(xaxis = "Initial savings",
            yaxis = "Labor",
            # legend = :topleft
            )
        
        Plots.plot!(
            size = (2400, 1600),
            legendfontsize = 24,
            guidefontsize = 28,
            tickfontsize = 20,

            bottom_margin = 100Plots.px,
            top_margin = 100Plots.px,
            left_margin = 100Plots.px,

            fontfamily = "Times", 
            )
    end

    return plot
end

colors = palette(:RdYlGn_10,3, rev = true)

plot_labor_scenarios = plot_labor([now_good,now_intermediate,now_bad],
    colors)

# Adding boomers:
let avT = 100
        low = avT - 79
        tmp = 0
        for t in low:avT
            tmp = tmp .+
                boomers[:optimal_choices][t,:,"l"].array .* scaling_factor
        end
        tmp = tmp ./ (T-low)
    Plots.plot!(s_range_scaled,tmp,
        # label = "$low, $avT",
        legend = false,
        linewidth = 5,
        color = "blue")
end

labels = ["Optimistic scenario", 
            "Intermediate scenario",            
            "Pessimistic scenario",
            "Historical trajectory"]

plot_labor_scenarios.series_list[1][:label] = labels[1]
plot_labor_scenarios.series_list[2][:label] = labels[2]
plot_labor_scenarios.series_list[3][:label] = labels[3]
plot_labor_scenarios.series_list[4][:label] = labels[4]
Plots.plot!(legend = :topright)

display(plot_labor_scenarios)

savefig("working_elements/Draft/output/labor_comparison.png")