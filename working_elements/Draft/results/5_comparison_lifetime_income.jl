function lifetime_income(choices, probabilities)
    labor = choices[:,:,"l"]
    lifetime_income = zeros(100)
    lifetime_income[1] = (β * probabilities[1]) .* (labor[1,1])/(1+small_r[1])
    for t in 2:100
        lifetime_income[t] = lifetime_income[t-1] + (β^(t) * probabilities[t]) .* (labor[t,1])/(1+small_r[t])
    end
    return lifetime_income
end

lifetime_income_bad = lifetime_income(now_bad.optimal_choices,
    average_proba_bad)
lifetime_income_intermediate = lifetime_income(now_intermediate.optimal_choices,
    average_proba_intermediate)
lifetime_income_good = lifetime_income(now_good.optimal_choices,
    average_proba_good)
lifetime_income_boomer = lifetime_income(boomers.optimal_choices,
    average_proba_boomer)

lifetime_incomes = [lifetime_income_good,
    lifetime_income_intermediate,
    lifetime_income_bad,
    lifetime_income_boomer]

Loss = scaling_factor * (lifetime_incomes[1][end] - lifetime_incomes[3][end])

labels = ["Optimistic scenario", 
            "Intermediate scenario",            
            "Pessimistic scenario",
            "Historical trajectory"]

colors = palette(:RdYlGn_10,3, rev = true)

colors = vcat(colors[1:3], :blue)

income_scaled = lifetime_incomes .* scaling_factor

plot_income = Plots.plot(
    xaxis = "Year", 
    yaxis = "Income")

[Plots.plot!(income_scaled[t], color = colors[t]) for t in 1:4]

Plots.plot!(
    size = (2400, 1600),
    legendfontsize = 40,
    guidefontsize = 60,
    tickfontsize = 40,

    bottom_margin = 100Plots.px,
    top_margin = 100Plots.px,
    left_margin = 100Plots.px,

    fontfamily = "Times", 
    )

plot_income.series_list[1][:label] = labels[1]
plot_income.series_list[2][:label] = labels[2]
plot_income.series_list[3][:label] = labels[3]
plot_income.series_list[4][:label] = labels[4]

plot_income.series_list[1][:color] = colors[1]
plot_income.series_list[2][:color] = colors[2]
plot_income.series_list[3][:color] = colors[3]
plot_income.series_list[4][:color] = colors[4]

display(plot_income)

savefig("working_elements/Draft/output/lifetime_income_comparison.png")