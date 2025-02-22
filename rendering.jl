using Pkg
Pkg.activate("")
using PlutoSliderServer

# Framework model 1 : 
PlutoSliderServer.export_notebook("working_elements/Framework_model_1.jl"; ) # Export it in HTML 
run(`mv working_elements/Framework_model_1.html website/resources/Framework_model_1.html`) # Move it to the resources folder
# Does not work. Why...
run(`mv working_elements/Framework_model_1plot1.png website/resources/Framework_model_1plot1.png`) # Move the first illustrative plot
run(`mv working_elements/Framework_model_1plot2.png website/resources/Framework_model_1plot2.png`) # Move the second illustrative plot

# Framework model 2 : 
PlutoSliderServer.export_notebook("working_elements/Framework_model_2.jl"; )
run(`mv working_elements/Framework_model_2.html website/resources/Framework_model_2.html`)

# Visualisation notebook : 
PlutoSliderServer.export_notebook("working_elements/Visualisation.jl"; )
run(`mv working_elements/Visualisation.html website/resources/Visualisation.html`)

run(`figlet Notebook rendering done !`)
