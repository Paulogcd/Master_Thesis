# The goal of this file is to construct a dataframe such that: 
# id, year, age, health, status
# Using as many years surveyed as possible. 
# It produces a final dataframe, used in the model.jl file, 
# names "final_df"

# Loading libraries: 
begin 
    using ReadStatTables
    using CSV
    using DataFrames
end

# 2022-2020: 

# Loading data:
begin
# Living:
    # For physical health:
    data_c_2022     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h22core/h22sta/H22C_R.dta")
    data_c_2022     = DataFrame(data_c_2022)
    data_c_2020     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h20sta/H20C_R.dta")
    data_c_2020     = DataFrame(data_c_2020)
    # For age:
    data_pr_2022    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h22core/h22sta/H22PR_R.dta")
    data_pr_2022    = DataFrame(data_pr_2022)
    data_pr_2020    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h20sta/H20PR_R.dta")
    data_pr_2020    = DataFrame(data_pr_2020)
# Dead:
    exit_2022       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x22exit/x22sta/X22pr_r.dta")
    exit_2022    = DataFrame(exit_2022)
end

# Defining IDs and Age:
begin
    # Defining IDs:
    ID_2022_a   = string.(data_c_2022[!,:HHID],data_c_2022[!,:PN])
    ID_2022_d   = string.(exit_2022[!,:hhid],exit_2022[!,:pn])
    ID_2020_a   = string.(data_c_2020[!,:HHID],data_c_2020[!,:PN])

    # Age:
    age_2022_a              = 2022 .- data_pr_2022[!,:SX067_R]
    age_of_death_2022       = 2022 .- exit_2022[!,:XSX067_R]
    age_2020_a              = 2020 .- data_pr_2020[!,:RX067_R]
end

# Defining Health status: 
begin 
    # Living:
    health_rate_2022_a                  = data_c_2022[!,:SC001]
    df_2022_a = DataFrame(ID = ID_2022_a, Year = fill(2022,length(ID_2022_a)),
            Age = age_2022_a, Health = health_rate_2022_a,
            Status = ones(length(ID_2022_a)))
    # Dead:
    # ID_ppl_alive_in_2020_dead_in_2022   = intersect(ID_2020_a,ID_2022_d)
    # data_c_2020.ID                      = ID_2020_a
    
    df_2022_d = DataFrame(ID = ID_2022_d, Year = fill(2022, length(ID_2022_d)), 
            Age = age_of_death_2022, Health = fill(8,length(ID_2022_d)), # 8 is the value for "don't know/NA" in the HRS dataset
            Status = zeros(length(ID_2022_d)))
    df_2020_a = DataFrame(ID = ID_2020_a, Year = fill(2020, length(ID_2020_a)), 
            Age = age_2020_a, Health = data_c_2020[!,:RC001], 
            Status = ones(length(ID_2020_a)))

    # We take the exit 2022 data, and are going to replace the Health by the previous one:
    fdf1 = filter(row -> row.ID in ID_2022_d, df_2020_a)

    # We take the exit 2022 data, and are going to replace the Health by the previous one:
    df_2022_d = filter(row -> row.ID in fdf1.ID, df_2022_d)
    df_2022_d.Health = fdf1.Health
    # df_2022_d

    df_20_22 = vcat(df_2020_a,df_2022_a,df_2022_d)
end

# 2020-2018:

# Loading data:
begin
    # Living:
        # For physical health:
        data_c_2018 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h18sta/H18C_R.dta")
        data_c_2018 = DataFrame(data_c_2018)
        # For age:
        data_pr_2018 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h18sta/h18pr_r.dta")
        data_pr_2018 = DataFrame(data_pr_2018)
    # Dead:
        exit_2020 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x20sta/X20PR_R.dta")
        exit_2020 = DataFrame(exit_2020)
end
    
# Defining IDs and Age:
begin
    # Defining IDs:
    ID_2018_a   = string.(data_c_2018[!,:hhid],data_c_2018[!,:pn])
    ID_2020_d   = string.(exit_2020[!,:HHID],exit_2020[!,:PN])

    # Age:
    age_2018_a              = 2018 .- data_pr_2018[!,:QX067_R]
    age_of_death_2020       = 2020 .- exit_2020[!,:XRX067_R]
end
    
# Defining Health status: 
begin 
    # Living:
    health_rate_2018_a                  = data_c_2018[!,:QC001]
    # df_2022_a = DataFrame(ID = ID_2022_a, Year = fill(2022,length(ID_2022_a)),
    #        Age = age_2022_a, Health = health_rate_2022_a,
    #        Status = ones(length(ID_2022_a)))
    # Dead:
    # ID_ppl_alive_in_2020_dead_in_2022   = intersect(ID_2020_a,ID_2022_d)
    # data_c_2020.ID                      = ID_2020_a
    
    df_2020_d = DataFrame(ID = ID_2020_d,
                            Year = fill(2020, length(ID_2020_d)), 
                            Age = age_of_death_2020,
                            Health = fill(8,length(ID_2020_d)), # 8 is the value for "don't know/NA" in the HRS dataset
                            Status = zeros(length(ID_2020_d)))

    df_2018_a = DataFrame(ID = ID_2018_a,
                            Year = fill(2018, length(ID_2018_a)), 
                            Age = age_2018_a,
                            Health = health_rate_2018_a, 
                            Status = ones(length(ID_2018_a)))

    # Ppl in the 2018 survey that are in exit 2020 (with health for 2018)
    # fdf1 = filter(row -> row.ID in ID_2022_d, df_2018_a)
    fdf1 = filter(row -> row.ID in ID_2020_d, df_2018_a)
    fdf1

    # We take the exit 2020 data, and are going to replace the Health by the previous one:
    fdf2 = filter(row -> row.ID in fdf1.ID, df_2020_d)
    fdf2.Health = fdf1.Health
    fdf2

    df_2020_d = fdf2
    
    df_18_20 = vcat(df_2018_a,df_2020_a,df_2020_d)
end

# df_final = vcat(df_18_20,df_20_22)

# 2018-2016

data_pr_2016    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/H16PR_R.dta")
data_c_2016     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/H16C_R.dta")
exit_2016       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/X16PR_R.dta")
exit_2018       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x18sta/X18PR_R.dta")

# Loading data:
begin
    # Living:
        # For physical health:
        data_c_2016     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/H16C_R.dta")
        data_c_2016     = DataFrame(data_c_2016)
        # For age:
        data_pr_2016    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/H16PR_R.dta")
        data_pr_2016    = DataFrame(data_pr_2016)
    # Dead:
    exit_2018           = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x18sta/X18PR_R.dta")
    exit_2018           = DataFrame(exit_2018)
end
    
# Defining IDs and Age:
begin
    # Defining IDs:
    ID_2016_a   = string.(data_c_2016[!,:HHID],data_c_2016[!,:PN])
    ID_2018_d   = string.(exit_2018[!,:HHID],exit_2018[!,:PN])

    # Age:
    age_2016_a              = 2016 .- data_pr_2016[!,:PX067_R]
    age_of_death_2018       = 2018 .- exit_2018[!,:XQX067_R]
end
    
# Defining Health status: 
begin 
    # Living:
    health_rate_2016_a                  = data_c_2016[!,:PC001]
    # df_2022_a = DataFrame(ID = ID_2022_a, Year = fill(2022,length(ID_2022_a)),
    #        Age = age_2022_a, Health = health_rate_2022_a,
    #        Status = ones(length(ID_2022_a)))
    # Dead:
    # ID_ppl_alive_in_2020_dead_in_2022   = intersect(ID_2020_a,ID_2022_d)
    # data_c_2020.ID                      = ID_2020_a
    
    df_2018_d = DataFrame(ID = ID_2018_d,
                            Year = fill(2018, length(ID_2018_d)), 
                            Age = age_of_death_2018,
                            Health = fill(8,length(ID_2018_d)), # 8 is the value for "don't know/NA" in the HRS dataset
                            Status = zeros(length(ID_2018_d)))

    df_2016_a = DataFrame(ID = ID_2016_a,
                            Year = fill(2016, length(ID_2016_a)), 
                            Age = age_2016_a,
                            Health = health_rate_2016_a, 
                            Status = ones(length(ID_2016_a)))

    # Ppl in the 2018 survey that are in exit 2020 (with health for 2018)
    # fdf1 = filter(row -> row.ID in ID_2022_d, df_2018_a)
    fdf1 = filter(row -> row.ID in ID_2018_d, df_2016_a)
    fdf1

    # We take the exit 2020 data, and are going to replace the Health by the previous one:
    fdf2 = filter(row -> row.ID in fdf1.ID, df_2018_d)
    fdf2.Health = fdf1.Health
    fdf2

    df_2018_d = fdf2
    
    df_16_18 = vcat(df_2016_a,df_2018_a,df_2018_d)
end

df_final = vcat(df_16_18,df_18_20,df_20_22)
