using Pkg
Pkg.activate("")
using PlutoSliderServer

# Export Notebooks in html : 
PlutoSliderServer.export_notebook("working_elements/Tractable_model_1.jl"; )
PlutoSliderServer.export_notebook("working_elements/Tractable_model_2.jl"; )

# Move them to the website directory : 
run(`mv working_elements/Tractable_model_1.html website/resources/Tractable_model_1.html`)
run(`mv working_elements/Tractable_model_2.html website/resources/Tractable_model_2.html`)

