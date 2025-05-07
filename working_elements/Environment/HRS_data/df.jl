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
begin 
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
        exit_2022       = DataFrame(exit_2022)
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
        df_20_22 = unique(df_20_22)
    end
end

# 2020-2018:
begin 
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
        
        df_18_20 = vcat(df_2018_a,df_2020_d)
        df_18_20 = unique(df_18_20)
    end
end



# 2018-2016
begin 
    data_pr_2016    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/H16PR_R.dta")
    data_c_2016     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/H16C_R.dta")
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
        
        df_16_18 = vcat(df_2016_a,df_2018_d)
        df_16_18 = unique(df_16_18)
    end
end

# df_final = vcat(df_16_18,df_18_20,df_20_22)

# 2016-2014
begin 

    data_pr_2014    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h14_man/H14PR_R.dta")
    data_c_2014     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h14_man/H14C_R.dta")
    exit_2014       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h14_man/X14PR_R.dta")
    exit_2016       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/X16PR_R.dta")

    # Loading data:
    begin
        # Living:
            # For physical health:
            data_c_2014     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h14_man/H14C_R.dta")
            data_c_2014     = DataFrame(data_c_2014)
            # For age:
            data_pr_2014    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h14_man/H14PR_R.dta")
            data_pr_2014    = DataFrame(data_pr_2014)
        # Dead:
        exit_2016       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h16_man/X16PR_R.dta")
        exit_2016           = DataFrame(exit_2016)
    end
        
    # Defining IDs and Age:
    begin
        # Defining IDs:
        ID_2014_a   = string.(data_c_2014[!,:HHID],data_c_2014[!,:PN])
        ID_2016_d   = string.(exit_2016[!,:HHID],exit_2016[!,:PN])

        # Age:
        age_2014_a              = 2014 .- data_pr_2014[:,:OX067_R]
        age_of_death_2016       = 2016 .- exit_2016[!,:ZX067_R]
    end
        
    # Defining Health status: 
    begin 
        # Living:
        health_rate_2014_a                  = data_c_2014[!,:OC001]
        
        df_2016_d = DataFrame(ID = ID_2016_d,
                                Year = fill(2016, length(ID_2016_d)), 
                                Age = age_of_death_2016,
                                Health = fill(8,length(ID_2016_d)), # 8 is the value for "don't know/NA" in the HRS dataset
                                Status = zeros(length(ID_2016_d)))

        df_2014_a = DataFrame(ID = ID_2014_a,
                                Year = fill(2014, length(ID_2014_a)), 
                                Age = age_2014_a,
                                Health = health_rate_2014_a, 
                                Status = ones(length(ID_2014_a)))

        # Ppl in the 2018 survey that are in exit 2020 (with health for 2018)
        # fdf1 = filter(row -> row.ID in ID_2022_d, df_2018_a)
        fdf1 = filter(row -> row.ID in ID_2016_d, df_2014_a)
        fdf1

        # We take the exit 2020 data, and are going to replace the Health by the previous one:
        fdf2 = filter(row -> row.ID in fdf1.ID, df_2016_d)
        fdf2.Health = fdf1.Health
        fdf2

        df_2016_d = fdf2
        
        df_14_16 = vcat(df_2014_a,df_2016_d)
        df_14_16 = unique(df_14_16)
    end
end

# df_final = vcat(df_14_16,df_16_18,df_18_20,df_20_22)
# df_final = unique(df_final)

# 2014-2012:
begin 
    data_pr_2012    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h12_man/H12PR_R.dta")
    data_c_2012     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h12_man/H12C_R.dta")
    exit_2012       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h12_man/X12PR_R.dta")
    exit_2014       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h14_man/X14PR_R.dta")

    # Loading data:
    begin
        # Living:
            # For physical health:
            data_c_2012     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h12_man/H12C_R.dta")
            data_c_2012     = DataFrame(data_c_2012)
            # For age:
            data_pr_2012    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h12_man/H12PR_R.dta")
            data_pr_2012    = DataFrame(data_pr_2012)
        # Dead:
        exit_2014           = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h14_man/X14PR_R.dta")
        exit_2014           = DataFrame(exit_2014)
    end
        
    # Defining IDs and Age:
    begin
        # Defining IDs:
        ID_2012_a   = string.(data_c_2012[!,:HHID],data_c_2012[!,:PN])
        ID_2014_d   = string.(exit_2014[!,:HHID],exit_2014[!,:PN])

        # Age:
        age_2012_a              = 2012 .- data_pr_2012[:,:NX067_R]
        age_of_death_2014       = 2014 .- exit_2014[!,:YX067_R]
    end
        
    # Defining Health status: 
    begin 
        # Living:
        health_rate_2012_a                  = data_c_2012[!,:NC001]
        
        df_2014_d = DataFrame(ID = ID_2014_d,
                                Year = fill(2014, length(ID_2014_d)), 
                                Age = age_of_death_2014,
                                Health = fill(8,length(ID_2014_d)), # 8 is the value for "don't know/NA" in the HRS dataset
                                Status = zeros(length(ID_2014_d)))

        df_2012_a = DataFrame(ID = ID_2012_a,
                                Year = fill(2012, length(ID_2012_a)), 
                                Age = age_2012_a,
                                Health = health_rate_2012_a, 
                                Status = ones(length(ID_2012_a)))

        # Ppl in the 2018 survey that are in exit 2020 (with health for 2018)
        # fdf1 = filter(row -> row.ID in ID_2022_d, df_2018_a)
        fdf1 = filter(row -> row.ID in ID_2014_d, df_2012_a)
        fdf1

        # We take the exit 2020 data, and are going to replace the Health by the previous one:
        fdf2 = filter(row -> row.ID in fdf1.ID, df_2014_d)
        fdf2.Health = fdf1.Health
        fdf2

        df_2014_d = fdf2
        
        df_12_14 = vcat(df_2012_a,df_2014_d)
        df_12_14 = unique(df_12_14)
    end
end

# 2012-2010:
begin 
    data_pr_2010    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h10_man/H10PR_R.dta")
    data_c_2010     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h10_man/H10C_R.dta")
    exit_2012       = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h12_man/X12PR_R.dta")

    # Loading data:
    begin
        # Living:
            # For physical health:
            data_c_2010     = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h10_man/H10C_R.dta")
            data_c_2010     = DataFrame(data_c_2010)
            # For age:
            data_pr_2010    = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h10_man/H10PR_R.dta")
            data_pr_2010    = DataFrame(data_pr_2010)
        # Dead:
        exit_2012           = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h12_man/X12PR_R.dta")
        exit_2012           = DataFrame(exit_2012)
    end
        
    # Defining IDs and Age:
    begin
        # Defining IDs:
        ID_2010_a   = string.(data_c_2010[!,:HHID],data_c_2010[!,:PN])
        ID_2012_d   = string.(exit_2012[!,:HHID],exit_2012[!,:PN])

        # Age:
        age_2010_a              = 2010 .- data_pr_2010[:,:MX067_R]
        age_of_death_2012       = 2012 .- exit_2012[!,:XX067_R]
    end
        
    # Defining Health status: 
    begin
        # Living:
        health_rate_2010_a                  = data_c_2010[!,:MC001]
        
        df_2012_d = DataFrame(ID = ID_2012_d,
                                Year = fill(2012, length(ID_2012_d)), 
                                Age = age_of_death_2012,
                                Health = fill(8,length(ID_2012_d)), # 8 is the value for "don't know/NA" in the HRS dataset
                                Status = zeros(length(ID_2012_d)))

        df_2010_a = DataFrame(ID        = ID_2010_a,
                                Year    = fill(2010, length(ID_2010_a)), 
                                Age     = age_2010_a,
                                Health  = health_rate_2010_a, 
                                Status  = ones(length(ID_2010_a)))

        # Ppl in the 2018 survey that are in exit 2020 (with health for 2018)
        # fdf1 = filter(row -> row.ID in ID_2022_d, df_2018_a)
        fdf1 = filter(row -> row.ID in ID_2012_d, df_2010_a)
        fdf1

        # We take the exit 2020 data, and are going to replace the Health by the previous one:
        fdf2 = filter(row -> row.ID in fdf1.ID, df_2012_d)
        fdf2.Health = fdf1.Health
        fdf2

        df_2012_d = fdf2
        
        df_10_12 = vcat(df_2010_a,df_2012_d)
        df_10_12 = unique(df_10_12)
        # describe(df_10_12)
    end
end

# df_final = vcat(df_10_12,df_12_14,df_14_16,df_16_18,df_18_20,df_20_22)

# 2012-2010:
begin 
    data_pr_2008    = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h08_man/H08PR_R.dta")
    data_c_2008     = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h08_man/H08C_R.dta")
    exit_2010       = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h10_man/X10PR_R.dta")

    # Loading data:
    begin
        # Living:
            # For physical health:
            data_c_2008     = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h08_man/H08C_R.dta")
            data_c_2008     = DataFrame(data_c_2008)
            # For age:
            data_pr_2008    = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h08_man/H08PR_R.dta")
            data_pr_2008    = DataFrame(data_pr_2008)
        # Dead:
        exit_2010           = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h10_man/X10PR_R.dta")
        exit_2010           = DataFrame(exit_2010)
    end
        
    # Defining IDs and Age:
    begin
        # Defining IDs:
        ID_2008_a   = string.(data_c_2008[!,:HHID],data_c_2008[!,:PN])
        ID_2010_d   = string.(exit_2010[!,:HHID],exit_2010[!,:PN])

        # Age:
        age_2008_a              = 2008 .- data_pr_2008[:,:LX067_R]
        age_of_death_2010       = 2010 .- exit_2010[!,:WZ068_R]
    end
        
    # Defining Health status: 
    begin
        # Living:
        health_rate_2008_a                  = data_c_2008[!,:LC001]
        
        df_2010_d = DataFrame(ID = ID_2010_d,
                                Year = fill(2010, length(ID_2010_d)), 
                                Age = age_of_death_2010,
                                Health = fill(8,length(ID_2010_d)), # 8 is the value for "don't know/NA" in the HRS dataset
                                Status = zeros(length(ID_2010_d)))

        df_2008_a = DataFrame(ID        = ID_2008_a,
                                Year    = fill(2008, length(ID_2008_a)), 
                                Age     = age_2008_a,
                                Health  = health_rate_2008_a, 
                                Status  = ones(length(ID_2008_a)))

        # Ppl in the 2018 survey that are in exit 2020 (with health for 2018)
        # fdf1 = filter(row -> row.ID in ID_2022_d, df_2018_a)
        fdf1 = filter(row -> row.ID in ID_2010_d, df_2008_a)
        fdf1

        # We take the exit 2020 data, and are going to replace the Health by the previous one:
        fdf2 = filter(row -> row.ID in fdf1.ID, df_2010_d)
        fdf2.Health = fdf1.Health
        fdf2

        df_2010_d = fdf2
        
        df_08_10 = vcat(df_2008_a,df_2010_d)
        df_08_10 = unique(df_08_10)
    end
end

# df_final = vcat(df_08_10,df_10_12,df_12_14,df_14_16,df_16_18,df_18_20,df_20_22)

# df_final = unique(df_final)

# 2008-2006:
begin 
    data_pr_2006    = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h06_man/H06PR_R.dta")
    data_c_2006     = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h06_man/H06C_R.dta")
    exit_2008       = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h08_man/X08PR_R.dta")

    # Loading data:
    begin
        # Living:
            # For physical health:
            data_c_2006     = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h06_man/H06C_R.dta")
            data_c_2006     = DataFrame(data_c_2006)
            # For age:
            data_pr_2006    = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h06_man/H06PR_R.dta")
            data_pr_2006    = DataFrame(data_pr_2006)
        # Dead:
        exit_2008           = ReadStatTables.readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h08_man/X08PR_R.dta")
        exit_2008           = DataFrame(exit_2008)
    end
        
    # Defining IDs and Age:
    begin
        # Defining IDs:
        ID_2006_a   = string.(data_c_2006[!,:HHID],data_c_2006[!,:PN])
        ID_2008_d   = string.(exit_2008[!,:HHID],exit_2008[!,:PN])

        # Age:
        age_2006_a              = 2006 .- data_pr_2006[:,:KX067_R ]
        age_of_death_2008       = 2008 .- exit_2008[!,:VZ068_R] 
    end
        
    # Defining Health status: 
    begin
        # Living:
        health_rate_2006_a                  = data_c_2006[!,:KC001]
        
        df_2008_d = DataFrame(ID = ID_2008_d,
                                Year = fill(2008, length(ID_2008_d)), 
                                Age = age_of_death_2008,
                                Health = fill(8,length(ID_2008_d)), # 8 is the value for "don't know/NA" in the HRS dataset
                                Status = zeros(length(ID_2008_d)))

        df_2006_a = DataFrame(ID        = ID_2006_a,
                                Year    = fill(2006, length(ID_2006_a)), 
                                Age     = age_2006_a,
                                Health  = health_rate_2006_a, 
                                Status  = ones(length(ID_2006_a)))

        # Ppl in the 2018 survey that are in exit 2020 (with health for 2018)
        # fdf1 = filter(row -> row.ID in ID_2022_d, df_2018_a)
        fdf1 = filter(row -> row.ID in ID_2008_d, df_2006_a)
        fdf1

        # We take the exit 2020 data, and are going to replace the Health by the previous one:
        fdf2 = filter(row -> row.ID in fdf1.ID, df_2008_d)
        fdf2.Health = fdf1.Health
        fdf2

        df_2008_d = fdf2
        
        df_06_08 = vcat(df_2006_a,df_2008_d)
        df_06_08 = unique(df_06_08)
    end
end

df_final = vcat(df_06_08,df_08_10,df_10_12,df_12_14,df_14_16,df_16_18,df_18_20,df_20_22)

df_final = unique(df_final)