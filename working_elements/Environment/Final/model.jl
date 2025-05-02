begin 
    using Plots
    using DataFrames
    using GLM
    # using Econometrics
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
    test
    Plots.plot(test.Year,test.GDP)
end

# Running survival regression
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

# Assessing role of previous health on health next period
begin

    clean_health = function(DF::DataFrame)
        DF = DF[DF[:,:Health] .!= -8, :]
        DF = DF[DF[:,:Health] .!= 8, :]
        DF = DF[DF[:,:Health] .!= 9, :]
        return DF
    end

    
    df_2022 = df_20_22[df_20_22[:,:Year] .== 2022,:]
    df_2022 = clean_health(df_2022)

    df_2020 = df_20_22[df_20_22[:,:Year] .== 2020,:]
    df_2020 = clean_health(df_2020)

    df_2018 = df_18_20[df_18_20[:,:Year] .== 2018,:]
    df_2018 = clean_health(df_2018)

    df1 = innerjoin(df_2022, df_2020, on=:ID, makeunique=true)
    df2 = innerjoin(df1,df_2018, on = :ID, makeunique=true)
    df3 = leftjoin(df2,temperature, on = :Year)

    # Formatting: 
    y = df3.Health
    # X = df3.Health_1
    y = coerce(y, Multiclass)
    X = select(df3, [:Health_1]) # This is mandatory

    model = MultinomialClassifier(penalty=:none) # Initialise a non-trained model.
    mach = machine(model, X, y) # Initialise a machine
    fit!(mach)

    probabilities = MLJ.predict(mach, X)

    # Transition matrix: 
    states = levels(X.Health_1)  # or use unique(X)
    n_states = length(states)
    transition_matrix = zeros(n_states, n_states)

    for (i, from_state) in enumerate(states)
        idx = findall(X.Health_1 .== from_state)
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

    # Optionally, convert to DataFrame for readability
    transition_df = DataFrame(transition_matrix, Symbol.(states))
    transition_df.rowindex = states
    transition_df

end


# Assessing role of previous health on health next period
# begin
    
    # Formatting: 
    # df3
    df3.Health = categorical(df3.Health)
    df3.Health_1 = categorical(df3.Health_1)
    df3.av_annual_t = Float64.(df3.av_annual_t)
    df3
    
    y = coerce(df3.Health, Multiclass)
    X = select(df3, [:Health_1, :av_annual_t])

    # One-hot encode the categorical predictor
    HotEncoder = @load OneHotEncoder pkg=MLJModels
    encoder = HotEncoder()
    mach_encoder = machine(encoder, X)
    fit!(mach_encoder)
    # MLJ.predict(mach_encoder, X)
    X_encoded = MLJ.transform(mach_encoder, X)

    # Create and fit the model
    model = MultinomialClassifier(penalty=:none)
    mach = machine(model, X_encoded, y)
    MLJ.fit!(mach)

    mach

    # Get the fitted model parameters
fitted_params_A = fitted_params(mach)
coefficients = fitted_params_A.coefs
intercept = fitted_params_A.intercept

# Create a DataFrame for visualization
using DataFrames
coef_df = DataFrame(
    feature = repeat([:intercept; names(X_encoded)], outer=length(levels(y))),
    class = repeat(levels(y), inner=size(X_encoded, 2)+1),
    coefficient = vcat(intercept[:], coefficients[:])
)

# Display the coefficients
display(coef_df)