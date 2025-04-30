# This file produces a "temperature" dataframe. 
begin 
    using Statistics
    using CSV
    using DataFrames
end

# Loading: 
begin 
    file_path = "/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/climate/Complete_TAVG_complete.txt"

    temperature = CSV.read(file_path, DataFrame; header=false, skipto=36, delim=' ', ignorerepeated=true)
    newnames = ["Year", "Month", 
            "Monthly anomaly (1)",
            "Monthly anomaly (2)",
            "Annual anomaly (1)",
            "Annual anomaly (2)",
            "Five years anomaly (1)",
            "Five years anomaly (2)",
            "Ten years anomaly (1)",
            "Ten years anomaly (2)",
            "Twenty years anomaly (1)",
            "Twenty years anomaly (2)"]
    rename!(temperature,newnames)
end

# Create average variables: 
begin 
    temperature[:,:annual_average] = (temperature[:,"Annual anomaly (1)"] .+ temperature[:,"Annual anomaly (2)"]) ./ 2
    temperature[:,:five_annual_average] = (temperature[:,"Five years anomaly (1)"] .+ temperature[:,"Five years anomaly (2)"]) ./ 2
    temperature[:,:ten_annual_average] = (temperature[:,"Ten years anomaly (1)"] .+ temperature[:,"Ten years anomaly (2)"]) ./ 2    
end

# Average months, to get value per year:
begin
    temperature = combine(groupby(temperature, :Year), [:annual_average, :five_annual_average] .=> mean .=> [:av_annual_t, :av_5_annual_t])
    temperature = DataFrame(temperature)
end

# describe(temperature)

# temperature = temperature[!,[:annual_average, Year]]