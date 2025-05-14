# First, run data.jl: 
include("data.jl")

using GLM
using OrdinalMultinomialModels
using HypothesisTests
using RegressionTables


df.HP = df.Blood_Pressure .+ 
    df.Lung_Disease .+ 
    df.Hearth_Condition .+ 
    df.Stroke

df = dropmissing!(df)

rename!(df, Dict("av_annual_t" => "Temperature"))

df.HP = convert.(Float64, df.HP)
df.Age = convert.(Float64, df.Age)
df.Temperature = convert.(Float64, df.Temperature)

describe(df)

# IV regression: 

IV_formula = @formula(HP ~
    Age + Temperature + Age * Temperature)

REG1 = lm(IV_formula, df)

df.HP_predicted = predict(REG1, df)

# Health transition regression:

#Data: 
DF = copy(df)
df_2022 = DF[DF[:,:Year] .== 2022,:]	
df_2020 = DF[DF[:,:Year] .== 2020,:]
df_2018 = DF[DF[:,:Year] .== 2018,:]
df_2016 = DF[DF[:,:Year] .== 2016,:]
df_2014 = DF[DF[:,:Year] .== 2014,:]
df_2012 = DF[DF[:,:Year] .== 2012,:]
df_2010 = DF[DF[:,:Year] .== 2010,:]
df_2008 = DF[DF[:,:Year] .== 2008,:]
df_2006 = DF[DF[:,:Year] .== 2006,:]

df_2022_2020 = leftjoin(df_2022,df_2020, on = :ID, makeunique=true)
df_2020_2018 = leftjoin(df_2020,df_2018, on = :ID, makeunique=true)
df_2018_2016 = leftjoin(df_2018,df_2016, on = :ID, makeunique=true)
df_2016_2014 = leftjoin(df_2016,df_2014, on = :ID, makeunique=true)
df_2014_2012 = leftjoin(df_2014,df_2012, on = :ID, makeunique=true)
df_2012_2010 = leftjoin(df_2012,df_2010, on = :ID, makeunique=true)
df_2010_2008 = leftjoin(df_2010,df_2008, on = :ID, makeunique=true)

DF = vcat(df_2022_2020,
        df_2020_2018,
        df_2018_2016,
        df_2016_2014, 
        df_2014_2012,
        df_2012_2010,
        df_2010_2008)

DF = dropmissing!(DF)

describe(DF)
# Regression:
equ8bis_formula = @formula(Health ~
    Health_1 + HP_predicted + #  + Age + 
    Health_1 * HP_predicted)#  * Age)

DF.Health = Float64.(DF.Health)
DF.Health_1 = Float64.(DF.Health_1)

DF = dropmissing!(DF)

REG2 =
    OrdinalMultinomialModels.polr(equ8bis_formula,
        DF, 
        LogitLink())

# Survival probability: 

survival_equation = @formula(Status ~
    Health + HP_predicted + # Age +
    Health * HP_predicted) # * Age)

REG3 = GLM.glm(survival_equation,
    DF,
    Bernoulli(),
    LogitLink())

# Print Regression Tables
latex_output = regtable(REG1; 
    stats=[:nobs, :r2],             # Show #observations and R²
    render = LatexTable()
)

println(latex_output)


# coefs = coef(REG2)
# stderrs = stderror(REG2)
# pvalues = pvalue(REG2)
# fieldnames(typeof(REG2.model))

# coef(REG2)
# stderror(REG2)
REG2

fields(typeof(1))


latex_output = regtable(REG2; 
    stats=[:nobs, :r2],             # Show #observations and R²
    render = LatexTable()
)

println(latex_output)

# GLM.loglikelihood(REG3)
# GLM.loglikelihood(REG1)

