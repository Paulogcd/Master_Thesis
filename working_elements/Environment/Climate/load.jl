using Statistics
using DelimitedFiles
using CSV
using DataFrames
using Plots

# Fahrenheit to celsius: 
celsius(x::Number) = 5/9 * (x::Number-32)

# Berkeley data: 
# https://berkeleyearth.org/data/
# https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Complete_TAVG_complete.txt

month_temperature = (2.56, 3.19,  5.29,  8.29, 11.28, 13.43, 14.31, 13.84, 12.05,  9.20,  6.05,  3.60)
annual_average = mean(month_temperature)

file_path = "/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/climate/Complete_TAVG_complete.txt"
data = read(file_path, String)

# In Bash: 
# FILE="/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/climate/Complete_TAVG_complete.txt"
# head -n 35 $FILE 

head(data)
readdlm(data)[1]


CSV.read(data, DataFrame)




df = CSV.read(file_path, DataFrame; header=false, skipto=36, delim=' ', ignorerepeated=true)

summary(df)
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
rename!(df,newnames)

df[:,:annual_average] = (df[:,"Annual anomaly (1)"] .+ df[:,"Annual anomaly (2)"]) ./ 2
df[:,:five_annual_average] = (df[:,"Five years anomaly (1)"] .+ df[:,"Five years anomaly (2)"]) ./ 2
df[:,:ten_annual_average] = (df[:,"Ten years anomaly (1)"] .+ df[:,"Ten years anomaly (2)"]) ./ 2

a = Plots.plot(df[:,:Year],df[:,:annual_average])
b = Plots.plot(df[:,:Year],df[:,:five_annual_average])
c = Plots.plot(df[:,:Year],df[:,:ten_annual_average])
# Wow. Crazy stuff.

a
b
c

# Next, intervals and dispersion: 

# So these are absolute values of confidence intervals, 
# but we could (maybe) use them as a proxy for temperature dispersion?

df[:,:range_1yav] = abs.(df[:,"Annual anomaly (2)"] .- df[:,"Annual anomaly (1)"])
df[:,:range_5yav] = abs.(df[:,"Five years anomaly (2)"] .- df[:,"Five years anomaly (1)"])
df[:,:range_10yav] = abs.(df[:,"Ten years anomaly (2)"] .- df[:,"Ten years anomaly (1)"])

Plots.plot(df[:,:Year],df[:,:range_1yav])
Plots.plot(df[:,:Year],df[:,:range_5yav])
Plots.plot(df[:,:Year],df[:,:range_10yav])




# %                  Monthly          Annual          Five-year        Ten-year        Twenty-year
# % Year, Month,  Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.,   Anomaly, Unc.