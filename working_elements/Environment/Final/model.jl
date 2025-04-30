begin 
    using Plots
    using DataFrames
    using GLM
end

# Checking load of data:
begin 
    temperature
    gdp
    df = df_final
end

# Merging:
begin 
    # Join df with gdp (leftjoin keeps all rows in df, even if no GDP data exists)
    combined = leftjoin(df, gdp, on=:Year)

    # Then join with temperature data (again, keeping all rows from the previous result)
    combined = leftjoin(combined, temperature[!, [:Year, :av_annual_t]], on=:Year)

    df = combined
    df
end

# Checking:
begin 
    test = DataFrame(Year = df.Year, GDP = df.GDP, Temperature = df.av_annual_t)
    test = unique(test)
    test
    Plots.plot(test.Year,test.GDP)
end

# df

# Plotting: 
begin
    model_health_age_temperature_gdp = GLM.glm(@formula(Status ~ Age + Health + av_annual_t + GDP), df, Bernoulli(), LogitLink())

    Poor        = DataFrame(Age = 1:110, Health = fill(5,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)
    Fair        = DataFrame(Age = 1:110, Health = fill(4,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)
    Good        = DataFrame(Age = 1:110, Health = fill(3,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)
    VeryGood    = DataFrame(Age = 1:110, Health = fill(2,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)
    Excellent   = DataFrame(Age = 1:110, Health = fill(1,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)



    pv = predict(model_health_age,Poor)
    fv = predict(model_health_age,Fair)
    gv = predict(model_health_age,Good)
    vgv = predict(model_health_age,VeryGood)
    ev = predict(model_health_age,Excellent)


    Plots.plot(pv, label = "Poor")
    Plots.plot!(fv, label = "Fair")
    Plots.plot!(gv, label = "Good")
    Plots.plot!(vgv, label = "Very good")
    Plots.plot!(ev, label = "Excellent")
    Plots.plot!(xaxis = "Age", yaxis = "Survival Probability")

    model_health_age_temperature_gdp
end