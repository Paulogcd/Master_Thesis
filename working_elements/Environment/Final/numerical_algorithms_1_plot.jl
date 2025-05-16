include("numerical_algorithms_1.jl")

using Plots

default(fontfamily = "Times")

# plotly()
Plots.gr()

# Consumption:
begin
    p = Plots.plot(xaxis = "Initial savings",
                yaxis = "Consumption")

    for avT in 40:20:100
            low = avT - 19
            tmp = 0
            for t in low:avT
                tmp = tmp .+
                    pure_numerical_no_interpolation[:optimal_choices][t,:,"c"].array .* scaling_factor
            end
            tmp = tmp ./ (T-low)
        Plots.plot!(s_range_scaled,tmp, label = "$low, $avT", legend = false, linewidth = 5 )
    end

    Plots.plot!(xaxis = "Initial savings",
        yaxis = "Consumption",
        legend = :topleft)
    
    Plots.plot!(
		size = (2400, 1600),
		legendfontsize = 40,
		guidefontsize = 70,
		tickfontsize = 40,

		bottom_margin = 100Plots.px,
		top_margin = 100Plots.px,
		left_margin = 100Plots.px,

		fontfamily = "Times", 
		)
    savefig("working_elements/Draft/output/consumption_policy.png")
end


# Savings:
begin
    p = Plots.plot(xaxis = "Initial savings",
                yaxis = "Savings")

    for avT in 40:20:100
            low = avT - 19
            tmp = 0
            for t in low:avT
                tmp = tmp .+
                    pure_numerical_no_interpolation[:optimal_choices][t,:,"sprime"].array .* scaling_factor
            end
            tmp = tmp ./ (T-low)
        Plots.plot!(s_range_scaled,tmp, label = "$low, $avT", legend = false, linewidth = 5 )
    end

    Plots.plot!(xaxis = "Initial savings",
        yaxis = "Savings",
        legend = :topleft)
    
    Plots.plot!(
		size = (2400, 1600),
		legendfontsize = 40,
		guidefontsize = 40,
		tickfontsize = 40,

		bottom_margin = 100Plots.px,
		top_margin = 100Plots.px,
		left_margin = 100Plots.px,

		fontfamily = "Times", 
		)
    savefig("working_elements/Draft/output/savings_policy.png")
end

# Labor:
begin
    p = Plots.plot(xaxis="Initial savings", yaxis="Labor")

    # Get number of states
    n_states = size(pure_numerical_no_interpolation[:optimal_choices], 2)

    for avT in 40:20:100
        low = avT - 19
        tmp = zeros(n_states)  # Proper vector initialization
        
        for t in low:avT
            tmp .+= pure_numerical_no_interpolation[:optimal_choices][t,:,"l"]
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
               legendfontsize=40,
               guidefontsize=40,
               tickfontsize=40,
               bottom_margin=100Plots.px,
               top_margin=100Plots.px,
               left_margin=100Plots.px,
               fontfamily="Times")
    
    savefig("working_elements/Draft/output/labor_policy.png")
end