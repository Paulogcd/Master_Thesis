using DataFrames, GLM, CategoricalArrays

df = DataFrame(
    Health = [3, 2, 4, 5, 1, 3, 2, 4, 5, 1],  # Categorical outcome (1-5)
    Health_1 = [2, 1, 3, 4, 1, 3, 2, 4, 5, 2]  # Predictor (1-5)
)

df.Health = categorical(df.Health)  # Convert to categorical

# Create binary models for each Health level (2-5) vs 1 (reference)
models = Dict()

df
for level in 2:5
    df[!, Symbol("Health_$level")] = df.Health .== level
    models[level] = glm(
        @formula(Symbol("Health_$level") ~ Health_1), 
        df, 
        Binomial(), 
        LogitLink()
    )
end