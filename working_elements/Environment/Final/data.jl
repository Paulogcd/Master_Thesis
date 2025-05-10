# Packages
begin 
    using Plots
    using DataFrames
    # using GLM
    # using OrdinalMultinomialModels
    # using MLJLinearModels, MLJ, MLJModels
end

# Files
begin
    include("../Climate/df.jl")
    include("../GDP/df.jl")
    include("../HRS_data/df.jl")
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

# Cleaning data and coding health = 6 for dead individuals
begin 
    clean_health = function(DF::DataFrame,COLUMN::AbstractString)
        DF = DF[DF[:,COLUMN] .!= -8, :]
        DF = DF[DF[:,COLUMN] .!= 8, :]
        DF = DF[DF[:,COLUMN] .!= 9, :]
        return DF
    end
    df[df.Status .== 0,:Health] .= 6
    
    df

    df = dropmissing!(df)
    df = clean_health(df,"Health")

    describe(df)
end

# Creating file: 
begin 
    CSV.write("working_elements/Empirical_models/data.csv", df)
    CSV.write("working_elements/Empirical_models/temperature.csv", temperature)
    CSV.write("working_elements/Empirical_models/gdp.csv", gdp)
end