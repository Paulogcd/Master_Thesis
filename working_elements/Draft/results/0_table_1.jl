begin 
    using Plots
    using DataFrames
    using Latexify
    using StatsPlots
    using Formatting
    default(fontfamily = "Times")
end

# Files
begin
    include("../../../working_elements/Environment/Climate/df.jl")
    include("../../../working_elements/Environment/GDP/df.jl")
    include("../../../working_elements/Environment/HRS_data/df.jl")
    df = df_final
end

# Merging
begin 
    # Join df with gdp (leftjoin keeps all rows in df, even if no GDP data exists)
    combined = leftjoin(df, gdp, on=:Year)

    # Then join with temperature data (again, keeping all rows from the previous result)
    combined = leftjoin(combined, temperature[!, [:Year, :av_annual_t]], on=:Year)

    df = combined
    # df
end

begin 
	"""
	The clean_health function takes out rows with Health values that correspond to NAs in the HRS data.
	"""
    clean_health = function(DF::DataFrame,COLUMN::AbstractString)
        DF = DF[DF[:,COLUMN] .!= -8, :]
        DF = DF[DF[:,COLUMN] .!= 8, :]
        DF = DF[DF[:,COLUMN] .!= 9, :]
        return DF
    end
end

begin 
    DF = copy(df)
    DF = dropmissing!(DF)

    DF = clean_health(DF,"Health")

    rename!(DF, Dict("av_annual_t" => "Temperature"))

    DF = DF[DF.Year .<= 2018,:]

    d = describe(DF)

    #for col in names(d) 
    #    d[!,col] = convert(String, d[!,col])
    # end

    # parse.(String, df[:, 1:end])

    d = d[Not(1),:]
    d = d[:,Not(6,7)]
    d
end 

begin 
    # Plots.histogram(DF.Age)
    
    N = zeros(length(unique(DF.Year)))
    N_1 = zeros(length(unique(DF.Year)))
    N_2 = zeros(length(unique(DF.Year)))
    N_3 = zeros(length(unique(DF.Year)))
    N_4 = zeros(length(unique(DF.Year)))
    N_5 = zeros(length(unique(DF.Year)))
    
    Years = unique(DF.Year)
    Years = string.(Years)
    push!(Years)

    Ns = [N,
        N_1,
        N_2, 
        N_3, 
        N_4, 
        N_5]

    for (index,year) in enumerate(unique(DF.Year))
        N[index] = nrow(DF[DF.Year .== year,:])
        N_1[index] = nrow(DF[DF.Year .== year .&& DF.Health .== 1,:])
        N_2[index] = nrow(DF[DF.Year .== year .&& DF.Health .== 2,:])
        N_3[index] = nrow(DF[DF.Year .== year .&& DF.Health .== 3,:])
        N_4[index] = nrow(DF[DF.Year .== year .&& DF.Health .== 4,:])
        N_5[index] = nrow(DF[DF.Year .== year .&& DF.Health .== 5,:])
    end
    
    results = DataFrame(Year = Years, 
        N1 = N_1, 
        N2 = N_2, 
        N3 = N_3, 
        N4 = N_4, 
        N5 = N_5, 
        Total = N)
    results_2 = results[:,Not(:Total,:Year)]
    results_2 = Matrix(results_2)

    # values = Matrix(results[:, [:N1, :N2, :N3, :N4, :N5]])

    # Colors: 
    colorsi = collect(palette(:RdYlGn_10, 5, rev = true))
    colors_2d = reshape(colorsi, 1, :)

    # Stacked barplot
    StatsPlots.groupedbar(
        string.(results.Year),  # x-axis labels as strings
        results_2,
        # yticks = (0:1000:10_000, [string(tick) for tick in 0:1000:25_000]),
        label = ["Excellent" "Very Good" "Good" "Fair" "Poor"],
        color = colors_2d,
        # color = ["#1b9e77", "#d95f02", "#7570b3"],
        bar_position = :stack,
        xlabel = "Year",
        ylabel = "Count",
        legend = :outerright,
        yformatter = :plain,
        # color = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"],
        # size = (1600, 800),
        size = (2400, 1600),
        legendfontsize = 35,
        guidefontsize = 35,
        tickfontsize = 35,
        # bar_width=0.7, 
        bottom_margin = 100Plots.px,
        top_margin = 100Plots.px,
        left_margin = 100Plots.px,
        fontfamily = "Times"
    )
end
    
savefig("working_elements/Draft/output/histogram_1.png")

