begin 
    using Plots
    using DataFrames
    using GLM
    using OrdinalMultinomialModels
    using MLJLinearModels, MLJ, MLJModels
end

begin
    include("../Climate/df.jl")
    include("../GDP/df.jl")
    include("../HRS_data/df.jl")
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
    # test
    Plots.plot(test.Year,test.GDP)
end

# Survival regression
begin
    model_health_age_temperature_gdp = GLM.glm(@formula(Status ~ Age + Health + av_annual_t + GDP), df, Bernoulli(), LogitLink())

    Poor        = DataFrame(Age = 1:110, Health = fill(5,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)
    Fair        = DataFrame(Age = 1:110, Health = fill(4,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)
    Good        = DataFrame(Age = 1:110, Health = fill(3,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)
    VeryGood    = DataFrame(Age = 1:110, Health = fill(2,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)
    Excellent   = DataFrame(Age = 1:110, Health = fill(1,110), av_annual_t = fill(0.723292,110), GDP = 21354.1)

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
    Plots.plot!(xaxis = "Age", yaxis = "Survival Probability")

    model_health_age_temperature_gdp
end

# Assessing role of health in 2020 on health 2022
begin

    clean_health = function(DF::DataFrame)
        DF = DF[DF[:,:Health] .!= -8, :]
        DF = DF[DF[:,:Health] .!= 8, :]
        DF = DF[DF[:,:Health] .!= 9, :]
        return DF
    end

    df_2022 = df_20_22[df_20_22[:,:Year] .== 2022,:]
    df_2022 = clean_health(df_2022)
    rename!(df_2022, Dict("Health" => "Health_2022"))

    df_2020 = df_20_22[df_20_22[:,:Year] .== 2020,:]
    df_2020 = clean_health(df_2020)
    rename!(df_2020, Dict("Health" => "Health_2020"))

    df_2018 = df_18_20[df_18_20[:,:Year] .== 2018,:]
    df_2018 = clean_health(df_2018)
    rename!(df_2018, Dict("Health" => "Health_2018"))

    df_2016 = df_16_18[df_16_18[:,:Year] .== 2016,:]
    df_2016 = clean_health(df_2016)
    rename!(df_2016, Dict("Health" => "Health_2016"))

    df1 = innerjoin(df_2022, df_2020, on=:ID, makeunique=true)
    df2 = innerjoin(df1,df_2018, on = :ID, makeunique=true)
    df3 = leftjoin(df2,temperature, on = :Year)

    # Formatting: 
    y = df3.Health_2022
    # X = df3.Health_1
    y = coerce(y, Multiclass)
    X = select(df3, [:Health_2020]) # This is mandatory

    model = MultinomialClassifier(penalty=:none) # Initialise a non-trained model.
    mach = machine(model, X, y) # Initialise a machine
    fit!(mach)

    probabilities = MLJ.predict(mach, X)

    # Transition matrix: 
    states = levels(X.Health_2020)  # or use unique(X)
    n_states = length(states)
    transition_matrix = zeros(n_states, n_states)

    for (i, from_state) in enumerate(states)
        idx = findall(X.Health_2020 .== from_state)
        if !isempty(idx)
            # For all observations where X == from_state, get predicted probabilities
            probs = probabilities[idx]
            for (j, to_state) in enumerate(states)
                # Average probability of transitioning to to_state
                transition_matrix[i, j] = mean(pdf.(probs, to_state))
            end
        end
    end

    transition_matrix

    # DataFrame for readability:
    transition_df = DataFrame(transition_matrix, Symbol.(states))
    # transition_df.rowindex = states
    transition_df

    sum(transition_df[1,:])
    sum(transition_df[2,:])
    sum(transition_df[3,:])
    sum(transition_df[4,:])
    sum(transition_df[5,:])

end


# Assessing role of previous health AND TEMPERATURE on health next period
begin
    
    # Formatting: 
    # df3
    df3.Health_2022         = categorical(df3.Health_2022)
    df3.Health_2020         = categorical(df3.Health_2020)
    df3.av_annual_t         = Float64.(df3.av_annual_t)
    y                       = coerce(df3.Health_2022, Multiclass)
    X                       = select(df3, [:Health_2020, :av_annual_t])


    # One-hot encode the categorical predictor
    # This allows to encode the X variables
    HotEncoder      = @load OneHotEncoder pkg=MLJModels
    encoder         = HotEncoder()
    mach_encoder    = machine(encoder, X)
    fit!(mach_encoder)
    
    X_encoded       = MLJ.transform(mach_encoder, X)

    # Create and fit the model
    model           = MultinomialClassifier(penalty=:none)
    mach            = machine(model, X_encoded, y)
    MLJ.fit!(mach)

    probabilities = MLJ.predict(mach,X_encoded)

    # Trying to plot it: 

    av_annual_range = range(minimum(df3.av_annual_t)-0.01, maximum(df3.av_annual_t)+0.01, length=10)

    # 2. Get all Health_1 categories
    health_2020_categories = levels(df3.Health_2020)
    health_2022_categories = levels(df3.Health_2022)

    # health_2020 = health_2020_categories[1]

    # 3. Create a plot for each Health_1 category
    for health_2020 in health_2020_categories
        # Create a DataFrame with fixed Health_1 and varying av_annual_t
        plot_data = DataFrame(
            Health_2020 = fill(health_2020, length(av_annual_range)),
            av_annual_t = av_annual_range
        )
        
        # CRITICAL: Align categorical levels
        plot_data.Health_2020 = categorical(plot_data.Health_2020)
        levels!(plot_data.Health_2020, health_2020_categories)  # Force same levels

        plot_data_encoded = MLJ.transform(mach_encoder, plot_data) #? 
        
        # Get predicted probabilities
        probs = MLJ.predict(mach, plot_data_encoded) #? 
        
        prob_matrix = Matrix{Float64}(undef, length(probs), length(health_2020_categories))
        for (i,p) in enumerate(probs)
            prob_matrix[i,:] = [p.prob_given_ref[l] for l in health_2020_categories]
        end

        # Create the plot
        p = plot(title = "Transition from Health_2020 = $health_2020",
                xlabel = "av_annual_t",
                ylabel = "Probability",
                legend = :topright)
        
        # Add a line for each Health category
        for (i, health) in enumerate(health_2020_categories)
            plot!(av_annual_range, prob_matrix[:,i],
                label="To $health", linewidth=2)
        end
        
        display(p)
    end

end

# Trying to generalise to any year (to get the temperature variation valid)
# begin 
    # Goal: ID, h_{t}, h_{t-1},t_{t}

    # 2022 - 2020
    dff1 = DataFrame(ID = df_20_22.ID, 
                    Health_t = df_20_22.Health, 
                    Age_t = df_20_22.Age)
                    # Missing: Health_t_1, Temperature
    
    dff1 = innerjoin(dff1, df_2020, on=:ID, makeunique=true)
    dff1 = select!(dff1,Not([:Age,:Status]))
    rename!(dff1, Dict("Health_2020" => "Health_t_1"))

    # 2020 - 2018
    dff2 = DataFrame(ID = df_18_20.ID, 
                    Health_t = df_18_20.Health, 
                    Age_t = df_18_20.Age)
    dff2 = innerjoin(dff2, df_2018, on=:ID, makeunique=true)
    dff2 = select!(dff2,Not([:Age,:Status]))
    rename!(dff2, Dict("Health_2018" => "Health_t_1"))

    # 2018 - 2016
    dff3 = DataFrame(ID = df_16_18.ID, 
                        Health_t = df_16_18.Health, 
                        Age_t = df_16_18.Age)
    dff3 = innerjoin(dff3, df_2016, on=:ID, makeunique=true)
    dff3 = select!(dff3,Not([:Age,:Status]))
    rename!(dff3, Dict("Health_2016" => "Health_t_1"))

    # Altogether: 
    dff = vcat(dff1,dff2,dff3)
    # Adding temperature: 
    dff = leftjoin(dff,temperature, on = :Year)

    describe(dff)

    clean_health_2 = function(DF::DataFrame,COLONNE::AbstractString)
        DF = DF[DF[:,COLONNE] .!= -8, :]
        DF = DF[DF[:,COLONNE] .!= 8, :]
        DF = DF[DF[:,COLONNE] .!= 9, :]
        return DF
    end

    dff = clean_health_2(dff, "Health_t")
    describe(dff)
    dff = dropmissing!(dff) # 3 observations dropped
    # describe(dff) # Checking: it's cleaned.

    # Formatting: 
    dff.Health_t            = categorical(dff.Health_t)
    dff.Health_t_1          = categorical(dff.Health_t_1)
    dff.av_annual_t         = Float64.(dff.av_annual_t)
    dff.Age_t               = Float64.(dff.Age_t)
    y                       = coerce(dff.Health_t, Multiclass)
    X                       = select(dff, [:Health_t_1, :av_annual_t, :Age_t])


    # One-hot encode the categorical predictor
    # This allows to encode the X variables
    HotEncoder      = @load OneHotEncoder pkg=MLJModels
    encoder         = HotEncoder()
    mach_encoder    = machine(encoder, X)
    fit!(mach_encoder)
    
    X_encoded       = MLJ.transform(mach_encoder, X)

    # Create and fit the model
    model           = MultinomialClassifier(penalty=:none)
    mach            = machine(model, X_encoded, y)
    MLJ.fit!(mach)

    probabilities   = MLJ.predict(mach, X_encoded)

    # Trying to plot it: 

    av_annual_range = range(minimum(dff.av_annual_t)-1, maximum(dff.av_annual_t)+1, length=100)

    # 2. Get all Health_1 categories
    health_t_1_categories = levels(dff.Health_t_1)
    health_t_categories = levels(dff.Health_t)

    # health_2020 = health_2020_categories[1]
    # health_t_1 = health_t_1_categories[1]

    # Here, we need an age range: 
    age_range = range(minimum(dff.Age_t), maximum(dff.Age_t))

    # Setting the backend to plotly: 
    plotly()

    # 3. Create a plot for each Health_1 category
    for health_t_1 in health_t_1_categories
        # Create a DataFrame with fixed Health_1 and varying av_annual_t
        plot_data = DataFrame(
            Health_t_1   = fill(health_t_1, length(av_annual_range) * length(age_range)),
            av_annual_t  = repeat(av_annual_range, length(age_range)),
            Age_t        = repeat(age_range, inner=length(av_annual_range))
        )
        
        # CRITICAL: Align categorical levels
        plot_data.Health_t_1 = categorical(plot_data.Health_t_1)
        levels!(plot_data.Health_t_1, health_t_1_categories)  # Force same levels

        plot_data_encoded = MLJ.transform(mach_encoder, plot_data) #? 
        
        # Get predicted probabilities
        probs = MLJ.predict(mach, plot_data_encoded) #? 

        # Reshape probabilities into a matrix (av_annual_t × Age_t × Health)
        prob_matrix = reshape([p.prob_given_ref[l] for p in probs, l in health_t_1_categories],
            (length(av_annual_range), length(age_range), length(health_t_1_categories)))

        # Create 3D plot
        p = Plots.surface(title = "Transition from Health_t_1 = $health_t_1",
            xlabel = "av_annual_t", 
            ylabel = "Age_t",
            zlabel = "Probability",
            # ylims = (minimum(age_range), maximum(age_range)),
            legend = :topright)
        
        # Plot a surface for each target health category
        for (i, health) in enumerate(health_t_1_categories)
            surface!(av_annual_range, age_range, prob_matrix[:, :, i]',
            label = "To $health", alpha = 0.7)
        end

        heatmap(age_range, av_annual_range, prob_matrix[:, :, 1],
        title="2D Slice Check", xlabel="Age", ylabel="av_annual")
        

        display(p) # Looks weird: the surface stops at y = 30 (age = 30)
    end

    size(prob_matrix)
    length(av_annual_range)
    length(age_range)
    length(health_t_1_categories)


# end

temperature