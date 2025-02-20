using Pkg
Pkg.activate("")
using PlutoSliderServer

# Export Notebooks in html : 
PlutoSliderServer.export_notebook("working_elements/Framework_model_1.jl"; )
PlutoSliderServer.export_notebook("working_elements/Framework_model_2.jl"; )
PlutoSliderServer.export_notebook("working_elements/Visualisation.jl"; )

# Move them to the website directory : 
run(`mv working_elements/Framework_model_1.html website/resources/Framework_model_1.html`)
run(`mv working_elements/Framework_model_2.html website/resources/Framework_model_2.html`)
run(`mv working_elements/Visualisation.html website/resources/Visualisation.html`)

