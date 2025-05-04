using Pkg
Pkg.activate(".")
using PlutoSliderServer

pwd()

### Framework models: 
# Framework model 1: 
PlutoSliderServer.export_notebook("working_elements/Framework_models/Framework_model_1.jl"; ) # Export it in HTML 
run(`mv working_elements/Framework_models/Framework_model_1.html website/resources/Framework_models/Framework_model_1.html`) # Move it to the resources folder
# Does work ! 
run(`mv working_elements/Framework_models/Framework_model_1plot1.png website/resources/Framework_models/Framework_model_1plot1.png`) # Move the first illustrative plot
run(`mv working_elements/Framework_models/Framework_model_1plot2.png website/resources/Framework_models/Framework_model_1plot2.png`) # Move the second illustrative plot

# Framework model 2: 
PlutoSliderServer.export_notebook("working_elements/Framework_models/Framework_model_2.jl"; )
run(`mv working_elements/Framework_models/Framework_model_2.html website/resources/Framework_models/Framework_model_2.html`)
run(`mv working_elements/Framework_models/Framework_model_2_plot_1.html website/resources/Framework_models/Framework_model_2_plot_1.html`) # Move the first illustrative plot
run(`mv working_elements/Framework_models/Framework_model_2_plot_2.html website/resources/Framework_models/Framework_model_2_plot_2.html`) # Move the second illustrative plot

# Framework model 3:
PlutoSliderServer.export_notebook("working_elements/Framework_models/Framework_model_3.jl"; )
run(`mv working_elements/Framework_models/Framework_model_3.html website/resources/Framework_models/Framework_model_3.html`)

# Other: 
# Demographics notebook:
PlutoSliderServer.export_notebook("working_elements/Other/Demographics.jl"; )
run(`mv working_elements/Other/Demographics.html website/resources/Demographics.html`)

# Visualisation notebook : 
PlutoSliderServer.export_notebook("working_elements/Other/Visualisation.jl"; )
run(`mv working_elements/Other/Visualisation.html website/resources/Visualisation.html`)

# Empirical models: 

# Empirical model 1: 
PlutoSliderServer.export_notebook("working_elements/Empirical_models/Empirical_model_1.jl"; )
run(`mv working_elements/Empirical_models/Empirical_model_1.html website/resources/Empirical_model_1.html`)

run(`figlet Notebook rendering done !`)
