
# Packages:
begin 
    using GLM
    using Plots
    using DataFrames
end

# Data: 
begin 
    Y                   = Float64.(rand(Binomial(), 100))
    X1                  = rand(100)
    data                = DataFrame(Y = Y, X1 = X1)
    new_data            = DataFrame(X1 = rand(100))
end

# Regression:
begin 
    logistic_model      = GLM.glm(@formula(Y ~ X1), data, Bernoulli(), LogitLink())
    predicted_values    = GLM.predict(logistic_model, new_data)
    Plots.plot(X1, predicted_values)
end

##########################################

# Package: 
begin 
    using Econometrics
end

# Data: 
begin 
    Y           = rand(1:5, 100)
    X1          = rand(1:5,100)
    data        = DataFrame(Y = Y, X1 = X1)
    new_data    = DataFrame(X1 = rand(1:5,100)) 
end

# Regression:
begin 
    multinomial_logistic        = Econometrics.fit(EconometricModel,
                                    @formula(Y ~ X1 ),
                                    data)
    Econometrics.predict(multinomial_logistic) # Yields the estimated vaues for the exact vector of values on which the regression was performed.
    # Econometrics.predict(multinomial_logistic, new_data) # Method error 
end

##################

begin 
    using OrdinalMultinomialModels
    data = DataFrame(Y = df3.Health, X1 = df3.Health_1)

    # data.Freq
    freq_df = combine(groupby(data, [:Y, :X1]), nrow => :Freq)

    # Left-join the frequencies back to the original dataframe
    data_with_freq = leftjoin(data, freq_df, on = [:Y, :X1])

    model = polr(@formula(Y ~ X1), data_with_freq, wts = data_with_freq[!,:Freq])
    
    # OrdinalMultinomialModels.predict(model, data_with_freq, kind=:probs)
    # predict(model, data_with_freq, kind=:probs)


    # using OrdinalMultinomialModels, RDatasets

    housing = dataset("MASS", "housing");

    housing
    data_with_freq

    houseplr1   = polr(@formula(Sat ~ Infl + Type + Cont), housing,
                   LogitLink(), wts = housing[!, :Freq])

    model       = polr(@formula(Y ~ X1), data_with_freq,
                    LogitLink(), wts = data_with_freq[!,:Freq])

    
    OrdinalMultinomialModels.predict(houseplr1, housing, kind=:probs)
    OrdinalMultinomialModels.predict(model, data_with_freq, kind=:probs)
end


methods(predict)

###################################################

using MLJLinearModels, MLJ

# Load dataset
df3

y = df3.Health
X = df3.Health_1

# Convert target to categorical if needed
y = coerce(y, Multiclass)
X = select(df3, :Health_1) # This is mandatory

# Define the multinomial logistic regression model
model = MultinomialClassifier(penalty=:none)  # no regularization; options: :l2, :l1, :elasticnet

# Wrap in MLJ machine
mach = machine(model, X, y)

# Fit the model
fit!(mach)

Xnew = DataFrame(Health_1 = rand(1:5,10))

# Predict on training data (probabilities)
# probabilities = predict(mach, X)

# Predict classes (most probable class)
predicted_classes = predict_mode(mach, X)

# Predict probabilities on new data
# The MLJ is mandatory
probabilities = MLJ.predict(mach, Xnew)  # returns probabilistic predictions
# probabilities = predict(mach, X)

data
probabilities[2]
probabilities[3]

probabilities[3] .== probabilities[2]

probabilities = MLJ.predict(mach, X)

## Markov Matrix ## 
using DataFrames

states = levels(data.X1)  # or use unique(X)
n_states = length(states)
transition_matrix = zeros(n_states, n_states)

for (i, from_state) in enumerate(states)
    idx = findall(data.X1 .== from_state)
    if !isempty(idx)
        # For all observations where X == from_state, get predicted probabilities
        probs = probabilities[idx]
        for (j, to_state) in enumerate(states)
            # Average probability of transitioning to to_state
            transition_matrix[i, j] = mean(pdf.(probs, to_state))
        end
    end
end

# Optionally, convert to DataFrame for readability
transition_df = DataFrame(transition_matrix, Symbol.(states))
transition_df.rowindex = states
transition_df
##


instance_index = 1  # Choose the instance you want to plot

# distribution = probabilities[instance_index]
# distribution = probabilities[end]
classes = levels(y)  # Get the class labels
probs = [pdf(distribution, c) for c in classes]  # Probabilities for each class

# Create a bar plot
plot(classes, probs,
    seriestype=:bar,
    title="Probability Distribution for Instance $instance_index",
    xlabel="Class",
    ylabel="Probability",
    xticks=(classes, string.(classes)), # Display class labels on x-axis
    legend=false)

num_instances = min(5, length(probabilities))  # Plot the first 5 instances or fewer
plots_array = []

for i in 1:num_instances
    distribution = probabilities[i]
    classes = levels(y)
    probs = [pdf(distribution, c) for c in classes]

    p = plot(classes, probs,
        seriestype=:bar,
        title="Instance $i",
        xlabel="Class",
        ylabel="Probability",
        xticks=(classes, string.(classes)),
        legend=false)

    push!(plots_array, p)
end

# Combine the plots into a single layout
plot(plots_array..., layout=(num_instances, 1), size=(600, 200*num_instances)) # Adjust size as needed
