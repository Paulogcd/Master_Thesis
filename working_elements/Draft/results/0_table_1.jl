begin 
    using Plots
    using DataFrames
    using Latexify
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

latexify(d; env = :table, booktabs = true, latex = false, fmt="%.2f" ) |> print
