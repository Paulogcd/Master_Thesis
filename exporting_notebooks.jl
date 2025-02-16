
using Pkg
Pkg.activate("")
using PlutoSliderServer

# To export Notebook : 
PlutoSliderServer.export_notebook("working_elements/Tractable_model_1.jl"; )
run(`mv working_elements/Tractable_model_1.html website/resources/Tractable_model_1.html`)

