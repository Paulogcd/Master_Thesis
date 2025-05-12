begin 
    using CSV
    using DataFrames
    using GLM
end

df = CSV.read("working_elements/Empirical_models/data.csv", DataFrame)
gdp = CSV.read("working_elements/Empirical_models/gdp.csv", DataFrame)

df = df[df.Year .>= 2018,:]

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
	GDP_2010 = gdp[gdp.Year .== 2010, :GDP][1]
	GDP_2012 = gdp[gdp.Year .== 2012, :GDP][1]
	GDP_2014 = gdp[gdp.Year .== 2014, :GDP][1]
	GDP_2016 = gdp[gdp.Year .== 2016, :GDP][1]
	GDP_2018 = gdp[gdp.Year .== 2018, :GDP][1]
	GDP_2020 = gdp[gdp.Year .== 2020, :GDP][1]
	GDP_2022 = gdp[gdp.Year .== 2022, :GDP][1]
end

begin 
	#temperature_2022 = Float64.(temperature[temperature.Year .== 2022, :av_annual_t][1])
	temperature_2020 =
		Float64(temperature[temperature.Year .== 2020, :av_annual_t][1])
	temperature_2018 =
		Float64(temperature[temperature.Year .== 2018, :av_annual_t][1])
	temperature_2016 =
		Float64(temperature[temperature.Year .== 2016, :av_annual_t][1])
	temperature_2014 =
		Float64(temperature[temperature.Year .== 2014, :av_annual_t][1])
end

DF = dropmissing!(df)
DF = clean_health(df,"Health")

age_range = range(minimum(df.Age), maximum(df.Age))

begin 
	Plots.gr()

	# fixed_temperature = 0.616167 # Temperature deviation of 2018
	# fixed_temperature = 0.61 # Temperature deviation of 2018
	
	model_health_age_temperature_gdp =
		GLM.glm(@formula(Status ~
						 	Age +
							Health
                            # Health_1
							# av_annual_t
							# Age * Health * av_annual_t
							# Health * av_annual_t +
							# Age * av_annual_t
							# When including GDP, the solver does not converge
							# log(GDP) + 
							# Age * log(GDP)
							# Year +
							# Year * Age + 
							# Age * Health
						),
				DF, Bernoulli(), LogitLink())

    Poor        =
		DataFrame(Age = age_range,
				  Health = fill(5,length(age_range)),
				  av_annual_t = fill(temperature_2018,length(age_range)),
				  GDP = fill(GDP_2018, length(age_range)),
				  Year = fill(2018,length(age_range)))
	
    Fair        = 
		DataFrame(Age = age_range,
				  Health = fill(4,length(age_range)),
				  av_annual_t = fill(temperature_2018,length(age_range)),
				  GDP = fill(GDP_2018,length(age_range)),
				  Year = fill(2018,length(age_range)))
	
    Good        = DataFrame(Age = age_range,
							Health = fill(3,length(age_range)),
							av_annual_t = fill(temperature_2018,length(age_range)),
							GDP = GDP_2018,
							Year = 2018)
	
    VeryGood    = DataFrame(Age = age_range,
							Health = fill(2,length(age_range)),
							av_annual_t = fill(temperature_2018,length(age_range)),
							GDP = GDP_2018,
						   Year = 2018)
	
    Excellent   = DataFrame(Age = age_range,
	 						Health = fill(1,length(age_range)),
	 						av_annual_t = fill(temperature_2018,length(age_range)),
	 						GDP = GDP_2018,
	 					   Year = 2018)
	 
    pv  = GLM.predict(model_health_age_temperature_gdp,Poor)
    fv  = GLM.predict(model_health_age_temperature_gdp,Fair)
    gv  = GLM.predict(model_health_age_temperature_gdp,Good)
    vgv = GLM.predict(model_health_age_temperature_gdp,VeryGood)
    ev  = GLM.predict(model_health_age_temperature_gdp,Excellent)

    Plots.plot(pv, label = "Poor")
    Plots.plot!(fv, label = "Fair")
    Plots.plot!(gv, label = "Good")
    Plots.plot!(vgv, label = "Very good")
    Plots.plot!(ev, label = "Excellent")
    Plots.plot!(xaxis = "Age",
				yaxis = "Probability")
				# title = "Survival probability as a function of age")
	Plots.plot!(legend = :bottomleft)
end

savefig("working_elements/Draft/output/figure_2.png")
