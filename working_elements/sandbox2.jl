using Pkg
using DataFrames
using ReadStatTables
using Statistics

# For information about the HRS data: 
# https://hrsdata.isr.umich.edu/data-products/public-survey-data?_gl=1*17r2dv7*_ga*MTE1ODEyMDA0NC4xNzQ1ODUyMjk1*_ga_FF28MW3MW2*MTc0NTg1MjI5NS4xLjAuMTc0NTg1MjI5NS4wLjAuMA..

# Loading the data: 
# data_pr = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h22core/h22sta/H22PR_R.dta")
# data_c = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h22core/h22sta/H22C_R.dta")
# data_b = readstat("/Users/paulogcd/Documents/Data_Master_Thesis/h22core/h22sta/H22B_R.dta")

data_c_2022 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h22core/h22sta/H22C_R.dta")
data_c_2020 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h20sta/H20C_R.dta")
data_c_2018 = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/h18sta/H18C_R.dta")

# How to identify someone in one year?
# We need both the personal number (nb), but also the household id (hhid).
# ppl_living_2022 = unique(data_c_2022[:PN]) # 15 !
# id = HHID + PN

ppl_living_2020 = string.(data_c_2020[:HHID],data_c_2020[:PN])
ppl_living_2022 = string.(data_c_2022[:HHID],data_c_2022[:PN])
# N_2022 = 15,856 (R), this match the data!

unique(ppl_living_2020)
unique(ppl_living_2022)

# Now, if we want to identify someone across surveys, 
# we need the previous ID.

# all_ppl = vcat(ppl_living_2022,ppl_living_2020) # won't work?
# unique(all_ppl)
# For the 2022 data, we can do: 
# id = HHID + PN + RSUBHH + SSUBHH 
# The name of the last two variables may change depending on the year !

ppl_living_2022_long_id = string.(data_c_2022[:HHID],data_c_2022[:PN],data_c_2022[:RSUBHH],data_c_2022[:SSUBHH])
# data_c_2022[:HHID] # 6 numbers
# data_c_2022[:PN] # 3 numbers
# data_c_2022[:RSUBHH] # 1 number
# data_c_2022[:SSUBHH] # 1 number
# 11 numbers in total.
n = parse.(Int128, data_c_2022[:HHID]) # To see them in numbers.
n = DataFrame(Column = n)
describe(n)


# If we want people who died, we need the exit data!
exit2022_pr = readstat("/Users/paulogcd/Library/Mobile Documents/com~apple~CloudDocs/Documents/Sciences_Po/Master/Data_Master_Thesis/x22exit/x22sta/X22pr_r.dta")

# To identify them, we could do:
ppl_died_2022_id = string.(exit2022_pr[:hhid],exit2022_pr[:pn])
unique(ppl_died_2022_id)

# To find people that were in previous wave but died: 
i = 0
for dead in ppl_died_2022_id 
    dead in ppl_living_2020 ? i = i+1 : i = i 
end
i

function compare(A::Vector,B::Vector)
    i = 0 
    for a in A 
        a in B ? i = i+1 : i = i 
    end
    return(i)
end

compare(ppl_living_2020,ppl_living_2022) # 12168 # people in common between these two waves.

length(unique(ppl_living_2020))

# But we need to identify their previous id, to be sure:
ppl_died_2022_id = string.(exit2022_pr[:hhid],exit2022_pr[:pn],exit2022_pr[:XSSUBHH])
unique(ppl_died_2022_id)

exit2022_pr[:XSX004]
exit2022_pr[:XSX067_R]