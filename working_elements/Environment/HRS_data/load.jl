using ReadStatTables
using StatsBase

# This file is dedicated to the loading of the needed files from the HRS.
# We need the 'c' section, for physical condition, and the exit files, 'pr' section.

# Main surveys:
data_c_2022 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h22core/h22sta/H22C_R.dta")
data_pr_2022 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h22core/h22sta/H22PR_R.dta")
data_c_2020 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h20sta/H20C_R.dta")
data_pr_2020 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h20sta/H20PR_R.dta")
# data_c_2018 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h18sta/H18C_R.dta")

# Exit surveys:
# We have to be extra careful here, because the names of the variables change!
exit_2022 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x22exit/x22sta/X22pr_r.dta")
exit_2022[:XSX004] # Month born
exit_2022[:XSX067_R] # Year born
exit_2020 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x20sta/X20PR_R.dta")
exit_2020[:XRX004] # Month born
exit_2020[:XRX067_R] # Year born
exit_2018 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x18sta/X18PR_R.dta")
exit_2018[:XQX004] # Month born
exit_2018[:XQX067_R] # Year born

# We are now going to create the full IDs:

# Main surveys:
ppl_2022 = string.(data_c_2022[:HHID],data_c_2022[:PN])
ppl_2020 = string.(data_c_2020[:HHID],data_c_2020[:PN])
ppl_2018 = string.(data_c_2018[:hhid],data_c_2018[:pn])

ppl_18_22 = vcat(ppl_2018,ppl_2020,ppl_2022)
unique(ppl_18_22)

compare(ppl_2022,ppl_2020)
compare(ppl_2022,ppl_2018)

# Exit surveys:
d_2022 = string.(exit_2022[:hhid],exit_2022[:pn])
d_2020 = string.(exit_2020[:HHID],exit_2020[:PN])
d_2018 = string.(exit_2018[:HHID],exit_2018[:PN])

compare(d_2022,d_2020) # 0, which is good.
compare(d_2022,d_2018) # 0, which is good.

# Now, the goal is to have a dataframe with 2 variables: age, and status (alive or dead)

# Let's begin with age: 
# Main surveys

# 2022: 
data_pr_2022[:SX004_R] # Month born
data_pr_2022[:SX067_R ] # Year born
age_2022 = 2022 .- data_pr_2022[:SX067_R ]
# describe(age_2022)
# Plots.plot(sort(age_2022))

# 2020: 
data_pr_2020[:RX004_R] # Month born
data_pr_2020[:RX067_R] # Year born
age_2020 = 2020 .- data_pr_2020[:RX067_R]

# Create a dataframe with the age and status (living) of individuals: 
age_20_22_a = vcat(age_2020,age_2022)
status_20_22_a = ones(length(age_20_22))
df_20_22_a = DataFrame(Age = age_20_22_a, Status = status_20_22_a)

# In the exit surveys, we need the date of death:
age_of_death_2022 = 2022 .- exit_2022[:XSX067_R]
age_of_death_2020 = 2020 .- exit_2020[:XRX067_R]
age_20_22_d = vcat(age_of_death_2020,age_of_death_2022)
status_20_22_d = zeros(length(age_20_22_d))
df_20_22_d = DataFrame(Age = age_20_22_d, Status = status_20_22_d)

main_df = vcat(df_20_22_a,df_20_22_d)



# Now, the model: 

# We have some missing data (5 lines), we drop it: 
main_df = dropmissing(main_df)

# We need Float64 for the GLM package: 
main_df[!,:Age] .= Float64.(main_df[!,:Age])
main_df[!,:Status] .= Int8.(main_df[!,:Status])

sort!(main_df)

model = GLM.glm(@formula(Status ~ Age), main_df, Bernoulli(), LogitLink())

# Now, to visualise predicted data: 
Age_df = DataFrame(Age = 1:110)
Plots.plot(predict(model,Age_df))
Plots.plot!(xaxis = "Age", yaxis = "Probability of surviving", legend = false)

# INCLUDING HEALTH:

# 2022:

N_2022_a = length(ppl_2022)
ppl_2022_a = string.(data_c_2022[:HHID],data_c_2022[:PN])
age_2022_a = 2022 .- data_pr_2022[:SX067_R]
status_a_22 = ones(N_2022_a)
year_a_22 = fill(2022,N_2022_a)
health_rate_2022 = data_c_2022[:SC001]
df_h_22 = DataFrame(ID = ppl_2022_a, 
        Age = age_2022_a, 
        Status = status_a_22, 
        Year = year_a_22, 
        Health = health_rate_2022)

compare(df_h_22[!,:Health],[-8,8,9]) # There are 20 NA-like in heatlh self report

df_h_22[!,:Health] .== -8
df_h_22[!,:Health] .== 8
df_h_22[!,:Health] .== 9

selector = df_h_22[!,:Health] .!= -8 .&& df_h_22[!,:Health] .!= 8 .&& df_h_22[!,:Health] .!= 9
# df_h_22[df_h_22[!,:Health] .!= -8 .&& df_h_22[!,:Health] .!= 8 .&& df_h_22[!,:Health] .!= 9,:]
# -8 means NA in web-interview, 8 means dont know, 9 means refused to answer
df_h_22_a = df_h_22[selector,:] # Now, without the NA in health self-report.

# 2020: 

ppl_2020_a = string.(data_c_2020[:HHID],data_c_2020[:PN])
N_2020_a = length(ppl_2020_a)
age_2020_a = 2020 .- data_pr_2020[:RX067_R]
status_a_20 = ones(N_2020_a)
year_a_20 = fill(2020,N_2020_a)
health_rate_2020 = data_c_2020[:RC001]
df_h_20_a = DataFrame(ID = ppl_2020_a, 
        Age = age_2020_a, 
        Status = status_a_20, 
        Year = year_a_20, 
        Health = health_rate_2020)

# One way to deal with the absence of self-report
# about the health of someone who died would be to insert manual a 5, 
# meaning "poor" health state, 
# but this would totally bias the regression.
# We could also try to take the previous declaration, and repeat the same value.

# Insert a manual 5: 
ppl_2022_d = string.(exit_2022[:hhid],exit_2022[:pn])
N_2022_d = length(ppl_2022_d)
age_of_death_2022 = 2022 .- exit_2022[:XSX067_R]
status_d_22 = zeros(N_2022_d)
year_d_22 = fill(2022,N_2022_d)
health_rate_2022_d = fill(5,N_2022_d)
df_h_22_d = DataFrame(ID = ppl_2022_d, 
        Age = age_of_death_2022, 
        Status = status_d_22, 
        Year = year_d_22, 
        Health = health_rate_2022_d)

# Trying to retrieve past health declaration: 

# Ppl in the 2020 survey that are in exit 2022 (with health for 2020)
fdf1 = filter(row -> row.ID in ppl_2022_d, df_h_20_a)

# We take the exit 2022 data, and are going to replace the Health by the previous one:
fdf2 = filter(row -> row.ID in fdf1.ID, df_h_22_d)
fdf2.Health = fdf1.Health
# compare(ppl_2022_d,ppl_2020_a) # 823 # To check

# Now, we can merge fdf1 with the main survey of 2022:
# fdf1
df_h_22 = vcat(fdf2,df_h_22_a)

# Running the logistic regression: 

model_health_age = GLM.glm(@formula(Status ~ Age + Health), df_h_22, Bernoulli(), LogitLink())

Poor = DataFrame(Age = 1:110, Health = fill(5,110))
Fair = DataFrame(Age = 1:110, Health = fill(4,110))
Good = DataFrame(Age = 1:110, Health = fill(3,110))
VeryGood = DataFrame(Age = 1:110, Health = fill(2,110))
Excellent = DataFrame(Age = 1:110, Health = fill(1,110))



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



# We now are interested in the income of individuals. 
# These data are observed in the 'Q' section.
data_q_2022 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h22core/h22sta/H22Q_H.dta")

data_q_2022[:SQ016] # Self employment
data_q_2022[:SQ020] # Wage and salary
data_q_2022[:SQ223_1] # From pension
data_q_2022[:SQ038]
data_q_2022[:SQ037]
data_q_2022[:SQ069]
data_q_2022[:SQ079]
data_q_2022[:SQ088]


test = data_q_2022[:SQ087] .- data_q_2022[:SQ086]

countmap(test)

dropmissing!(test)

describe(data_q_2022[:SQ087])

x = dropmissing(data_q_2022[:SQ087])

countmap(data_q_2022[:SQ087])
countmap(data_q_2022[:SQ086])

unique(test)


unique(data_q_2022[:SQ088])

unique(data_q_2022[:SQ079])

unique(data_q_2022[:SQ069])

unique(data_q_2022[:SQ037])
describe(data_q_2022[:SQ037])

describe(data_q_2022[:SQ038])
unique(data_q_2022[:SQ038])
describe(data_q_2022[:SQ223_1])
unique(data_q_2022[:SQ223_1])

# It seems delicate to control for "income" with these Data. 
# Instead, if we use the gdp ?

# Other data: 
exit_2018 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x18sta/X18PR_R.dta")

# Manually loading the data of other years: 
data_pr_2016    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/H16PR_R.dta")
data_c_2016     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/H16C_R.dta")
exit_2016       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/X16PR_R.dta")
