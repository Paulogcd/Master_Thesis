# https://fred.stlouisfed.org/series/GDP
gdp = CSV.read("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/GDP/GDP.csv", DataFrame)
# Units:  Billions of Dollars, Seasonally Adjusted Annual Rate
# Time: per year, average of the quarterly gdp.
# Plots.plot(gdp.observation_date, gdp.GDP)
gdp.Year = 1947:2024
# gdp
gdp = gdp[:,["GDP","Year"]]
# gdp
