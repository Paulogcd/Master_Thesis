average_proba_boomer
average_proba_good            
average_proba_intermediate    
average_proba_bad             

Plots.plot(1:100,average_proba_boomer, label = "Historical trajectory"                    , linewidth = 5)
Plots.plot!(average_proba_good, label = "Optimistic scenario"                     , linewidth = 5)
Plots.plot!(average_proba_intermediate, label = "Intermediate scenario"     , linewidth = 5)
Plots.plot!(average_proba_bad, label = "Pessimistic scenario"                       , linewidth = 5)

Plots.plot!(xaxis = "Time", yaxis = "Probability")

Plots.plot!(legend = :bottomleft, 
    size = (2400, 1600),
    legendfontsize = 40,
    guidefontsize = 60,
    tickfontsize = 40,

    bottom_margin = 100Plots.px,
    top_margin = 100Plots.px,
    left_margin = 100Plots.px, 
    color = ["green", "blue", "orange","red"])

savefig("working_elements/Draft/output/demographic_comparison.png")
