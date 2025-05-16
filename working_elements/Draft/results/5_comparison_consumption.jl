
using Plots

default(fontfamily = "Times")

# plotly()
Plots.gr()

function plot_policies(solution)
    # Consumption:
    begin
        plot_consumption = Plots.plot(xaxis = "Initial savings",
                    yaxis = "Consumption")

        for avT in 40:20:100
                low = avT - 19
                tmp = 0
                for t in low:avT
                    tmp = tmp .+
                        solution[:optimal_choices][t,:,"c"].array .* scaling_factor
                end
                tmp = tmp ./ (T-low)
            Plots.plot!(s_range_scaled,tmp, label = "$low, $avT", legend = false, linewidth = 5 )
        end

        Plots.plot!(xaxis = "Initial savings",
            yaxis = "Consumption",
            legend = :topleft)
        
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
        # savefig("working_elements/Draft/output/consumption_policy.png")
    end

    # Savings:
    begin
        plot_savings = Plots.plot(xaxis = "Initial savings",
                    yaxis = "Savings")

        for avT in 40:20:100
                low = avT - 19
                tmp = 0
                for t in low:avT
                    tmp = tmp .+
                        solution[:optimal_choices][t,:,"sprime"].array .* scaling_factor
                end
                tmp = tmp ./ (T-low)
            Plots.plot!(s_range_scaled,tmp, label = "$low, $avT", legend = false, linewidth = 5 )
        end

        Plots.plot!(xaxis = "Initial savings",
            yaxis = "Savings",
            legend = :topleft)
        
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
        # savefig("working_elements/Draft/output/savings_policy.png")
    end

    # Labor:
    begin
        plot_labor = Plots.plot(xaxis="Initial savings", yaxis="Labor")

        # Get number of states
        n_states = size(pure_numerical_no_interpolation[:optimal_choices], 2)

        for avT in 40:20:100
            low = avT - 19
            tmp = zeros(n_states)  # Proper vector initialization
            
            for t in low:avT
                tmp .+= solution[:optimal_choices][t,:,"l"]
            end
            tmp .= tmp ./ (avT - low + 1)  # Correct averaging
            
            Plots.plot!(s_range_scaled, tmp, 
                    label="$low-$avT", 
                    linewidth=5)
        end

        # Plot formatting
        Plots.plot!(xaxis="Initial savings",
                yaxis="Labor",
                legend=:topright,
                size=(2400, 1600),
                legendfontsize=24,
                guidefontsize=28,
                tickfontsize=20,
                bottom_margin=100Plots.px,
                top_margin=100Plots.px,
                left_margin=100Plots.px,
                fontfamily="Times")
        
        # savefig("working_elements/Draft/output/labor_policy.png")
    end
    plots = [plot_consumption,plot_labor,plot_savings]
    return plots
end

plot_policies(boomers)[1]
plot_policies(now_bad)[1]
plot_policies(now_intermediate)[1]
plot_policies(now_good)[1]

# Consumption: 
function plot_consumption(solutions, colors)
    
    plot_consumption = Plots.plot(xaxis = "Initial savings",
                yaxis = "Consumption")

    for (index,solution) in enumerate(solutions)

        let avT = 100
                low = avT - 79
                tmp = 0
                for t in low:avT
                    tmp = tmp .+
                        solution[:optimal_choices][t,:,"c"].array .* scaling_factor
                end
                tmp = tmp ./ (T-low)
            Plots.plot!(s_range_scaled,tmp,
                # label = "$low, $avT",
                legend = false,
                linewidth = 5,
                color = colors[index])
        end

        Plots.plot!(xaxis = "Initial savings",
            yaxis = "Consumption",
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

    return plot_consumption
end

colors = palette(:RdYlGn_10,3, rev = true)

plot_consumption_scenarios = plot_consumption([now_good,now_intermediate,now_bad],
    colors)

# Adding boomers:
let avT = 100
        low = avT - 79
        tmp = 0
        for t in low:avT
            tmp = tmp .+
                boomers[:optimal_choices][t,:,"c"].array .* scaling_factor
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

plot_consumption_scenarios.series_list[1][:label] = labels[1]
plot_consumption_scenarios.series_list[2][:label] = labels[2]
plot_consumption_scenarios.series_list[3][:label] = labels[3]
plot_consumption_scenarios.series_list[4][:label] = labels[4]
Plots.plot!(legend = :topleft)

display(plot_consumption_scenarios)

savefig("working_elements/Draft/output/consumption_comparison.png")