### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 0d381a45-b8a7-4ce0-a904-b238fd26902d
begin
	using Plots
	using PlutoUI
	using Distributions
	using PlotlyJS
	using Plotly
	using DataFrames
	using Dates
	using Random
	# using WebIO
end

# ╔═╡ d532165b-d442-4114-a3b9-f866a281c67f
md"# Framework Model 1

Author: Paulo GUGELMO CAVALHEIRO DIAS

Date: $(monthname(today())) $(day(today())), $(year(today()))

This model aims to set the basis for a further modelisation of the relationship between health and the economic choices of the agents.
"

# ╔═╡ ab992a5b-6a2c-4457-a651-c6818df8f5e2
begin
	Random.seed!(1234)
	TableOfContents()
end

# ╔═╡ 0ef87638-e32f-11ef-1ac0-094df082b41c
md"# Summary

1. Health 
2. Weather
3. Productivity
4. Survival
5. Maximization Programm 
6. Other
"

# ╔═╡ 4c2a0eb5-6bf6-4b8b-9c1d-51107af68743
md"# 1. Health"

# ╔═╡ 1d30ee07-9993-4cdf-bea0-f318b5461079
md"
Health can be either bad or good :

$$h\in\{good, bad\}=\{g,b\}$$

Let us suppose the following transition matrix : 

$$\pi=\begin{pmatrix} p(g|g) & p(b|g) \\ p(g|b) & p(b|b) \end{pmatrix}=\begin{pmatrix} \alpha_1 & 1-\alpha_1 \\ \alpha_2 & 1-\alpha_2 \end{pmatrix}$$

"

# ╔═╡ 33467e51-029b-4fc5-a889-ee8a6e108a7e
# Good health parameter 1 : 
@bind α_1 Slider(0:0.001:1, default=0.85)

# ╔═╡ 95d6c246-2773-4141-b53a-428364936b8a
# Good health parameter 2 : 
@bind α_2 Slider(0:0.001:1, default=0.35)

# ╔═╡ f2dcf131-4c96-4259-85eb-c78f2ea1b700
(α_1,α_2)

# ╔═╡ 9cce5faf-f860-494d-8de7-87f0c698e1b7
# Probability of being in good health next period : 
health_probability(h) = h == "g" ? α_1 : α_2

# ╔═╡ ea3e3f1f-6e07-470f-bd6c-bb7e12bb9ed8
md" 
Let us also define the following indicator functions : 
"

# ╔═╡ 22220a91-4f5a-4fdc-bd84-f0d79f911034
begin
	indicator_function_bad_health(h) = h == "b" ? 1 : 0 
	indicator_function_weather_deviation(w) = w == "d" ? 1 : 0 
end

# ╔═╡ a65ab9ac-f91c-4c6c-a162-34a3fc3ced13
md"
# 2. Weather

Let us assume that the weather can either follow its normal route, or deviate : 

$$w \in\{normal, deviation\}=\{n,d\}$$

Let us suppose that it has probability $\alpha_3$ of being normal, and $1-\alpha_3$ to deviate.

"

# ╔═╡ 8eff2abf-254e-4336-b2b5-7d46a75279f3
# Probability of weather being normal : 
@bind α_3 Slider(0:0.001:1, default=0.5)

# ╔═╡ dc128964-6f99-4695-8c39-4b688b6041d8
begin
	weather(p) = rand(Binomial(1,p)) == 1 ? "n" : "d"
end

# ╔═╡ 13259939-4338-44c0-904f-2d2a73a25879
md" 
# 3. Productivity 

Let us define productivity as $z$, such as : 

$$z = f(h,w,age)+ X$$

With $f(.)$ being a determinisic function of health, weather, and age, and $X$ being stochastic : 

$$f(h,w,\text{age})=\text{age}^{\alpha_4}-\alpha_5\cdot\text{age}\cdot\mathbb{1}\{h=b\}-\alpha_6\cdot\text{age}\cdot\mathbb{1}\{w=d\}$$

$$X\sim \mathcal{N}(0,1)$$

"

# ╔═╡ 4dec6012-eb5c-4db2-bb95-e67e29ef4532
# Concavity of the function : 
@bind α_4 Slider(0:0.001:1, default=0.5)

# ╔═╡ 156417da-bd77-425a-a41d-2f5ef31c339c
# Health penalty :
@bind α_5 Slider(0:0.001:0.1, default=0.06)

# ╔═╡ 4748644a-e1ef-4e1d-9e43-b0e566a74d0a
# Bad weather penalty :
@bind α_6 Slider(0:0.001:0.1, default=0.04)

# ╔═╡ 5439a417-bde0-4255-b310-09140dfe2eda
begin
	skill(h,w,age) = age^(α_4)-age*α_5*indicator_function_bad_health(h)-age*α_6*indicator_function_weather_deviation(w)
	productivity(h,w,age) = skill(h,w,age) + rand(Normal(0,0.3))
end

# ╔═╡ 4d6ec0eb-f023-451f-80ba-14b6fb3f1e24
(α_4,α_5,α_6)

# ╔═╡ 7c1b788b-9d94-4d3a-93ce-385bb8a3d3ae
# Setting the Plots backend : 
# plotlyjs()
# gr()

# ╔═╡ edf27a0c-9e66-46f1-918b-b404e1b6dc7a
begin
	gr()
	Plots.plot(18:100, skill.("g","n",18:100), label = "Good health in normal weather.")
	Plots.plot!(18:100, skill.("b","n",18:100), label = "Bad health in normal weather.")
	Plots.plot!(18:100, skill.("g","d",18:100), label = "Good health in deviation weather.")
	Plots.plot!(18:100, skill.("b","d",18:100), label = "Bad health in deviation weather.")
	Plots.plot!(title = "Deterministic skill of workers in \n function of age and weather.")
end

# ╔═╡ a79e4de7-ad33-4043-a416-2ce3952bbafa
begin
	gr()
	Plots.plot(18:100, productivity.("g","n",18:100), label = "Good health in normal weather.")
	Plots.plot!(18:100, productivity.("b","n",18:100), label = "Bad health in normal weather.")
	Plots.plot!(18:100, productivity.("g","d",18:100), label = "Good health in deviation weather.")
	Plots.plot!(18:100, productivity.("b","d",18:100), label = "Bad health in deviation weather.")
	Plots.plot!(title = "Semi-sochastic productivity of workers in \n function of age and weather.")
end

# ╔═╡ 6b7607e3-ced8-4549-9287-e4fc0d578908
md"

# 4. Survival

Let us suppose that the agents have a probability of dying at time $t$, defined as : 

$$\zeta_{t} \equiv \zeta_{t}(\text{age}_{t},h_{t}) = f(\Lambda(n(\text{age}_{t}))+\alpha_{11}*\mathbb{1}\{h=b\}$$

With : 

$$n(\text{x}) = \frac{(\text{x}-\alpha_7)}{\alpha_8}$$

$$\Lambda(\text{x}) = \frac{\alpha_9}{\text{exp}^{-\alpha_{10}\cdot\text{x}}}$$

$$f(x) = 
\begin{cases} 
  1 & \text{if } x>1 \\
  x & \text{if } x\in [0,1] \\
  0 & \text{if } x<0
\end{cases}$$

"

# ╔═╡ 4f550514-f478-4cce-9cea-2f846c668bcf
# Effect of age 'mean' normalisation :
@bind α_7 Slider(0:0.01:100, default=98)

# ╔═╡ c54d7f45-46dd-4cc8-ac82-141427fccc3f
# Effect of age 'sd' normalisation : 
@bind α_8 Slider(0:0.01:100, default=4)

# ╔═╡ 3ad533a5-3f3c-46d3-9229-208679e7f01a
# General survival rate in logistic function : 
@bind α_9 Slider(0.00:0.01:1.22, default=0.9)

# ╔═╡ 00629ad3-fe8e-403a-8d94-35a74a263447
# Last effect of age in logistic function : 
@bind α_10 Slider(0:0.01:1, default=0.5)

# ╔═╡ c2f427aa-9e39-4957-982a-6e863f4a4853
# Effect of bad health (coefficient of indicator function) :
@bind α_11 Slider(0:0.01:0.9, default=0.01)

# ╔═╡ 3d11cbe4-e503-43a7-8bba-6d0a590ef980
(α_7,α_8,α_9,α_10,α_11)

# ╔═╡ 1a578c44-650f-4b5c-8c5a-bbc47ad3a904
begin
	gr()
	normalize(age) = (age-α_7)/α_8
	Λ(age) = α_9/(exp(-α_10*age))
	function contain!(x::Number)
	    if x > 1 
	        return 1
	    elseif x < 0
	        return 0
	    else
	        return x
	    end
	end
	# ζ the probabiliy of dying :
	ζ(age,h) = contain!(Λ(normalize(age))+α_11*indicator_function_bad_health(h))
	
	Plots.plot(1:100, ones(100).-ζ.(1:100,"g"), label = "good health")
	Plots.plot!(ones(100)-ζ.(1:100,"b"), label = "bad health")
	Plots.plot!(title = "Probability of surviving for good and bad health.")
end

# ╔═╡ a7170b37-86d2-4e89-92a1-ef3c19793776
# Firs period with probability 1 of dying : 
ζ(99,"b"), ζ(99,"g")

# ╔═╡ 41e9ad1c-248a-4fd4-8c81-c037f213171b
md" With this survival function defined through the probabiliy of dying, we can now run some simulations. First, we can run a simulation for one person : "

# ╔═╡ 4a88de81-0e41-4889-a45b-d303fa6dbfcf
begin

	"""
	The function `sim()` simulates the life of an individual. 
	It returns a tuple containing: 

	1. Their age of death, 
	2. Their living status history (1 when living, 0 when being dead),
	3. Their health status history ("g" when good, "b" when bad).
	"""
	function sim()
		living_history = zeros(100)
		health_history = Vector{String}(undef,100)

		# previous_health = "g"
	
		for t in 1:100
	    
		   if t == 1
		       global previous_health = "g"
		   end
	    
		    # The age : 
		    age = t
		    
		    # The weather ("n" or "d") :
		    weather_t = weather(α_3)
		    
		    # The health status : 
		    # probability of being in good health : 
		    pgh = probability_good_health = health_probability(previous_health)
		    health_t = rand(Binomial(1,pgh)) == 1 ? "g" : "b"
			health_history[t] = health_t
			global previous_health = health_t

		
		    # The living status : 
		    pd = probability_dying = ζ(age,health_t)
		    living_status = rand(Binomial(1,1-pd))
		    global living_history[t] = living_status

			# Plots.plot(1:100,living_history)

			# When death comes : 
			if living_status == 0
				# print("Agent died at ", t)
				results = (age,living_history,health_history)
				return(results)
				break
			else
			end
			
		end
	end
end

# ╔═╡ 3f7790ab-248d-43f6-9b8c-5fe50e034cc3
begin
	# Plots.plot(1:100,living_history)
	single_results = sim()
	# typeof(single_results)
	# single_results[3]
	# Plots.plot(single_results[2])
	# Plots.title!("Living status")
	# single_results
end

# ╔═╡ ccc5887d-f8b9-4851-9299-043272b9336a
md"We can now run the simulation for a population of individuals :"

# ╔═╡ 24de739d-d053-4e41-83dd-0e076f2fe471
begin
	"""
	Th function `popsim(I)` runs a simulation for `I` individuals. 
	
	It returns a 3 dimensions tuple with, for each individual : 

	- Their age of death 
	- Their living status history 
	- Their health history 
	"""
	function popsim(I)
		
		# Initialize arrays with proper dimensions
		AOD = age_of_death = Array{Number}(undef, 0)
		LH = living_history = Array{Vector{Float64}}(undef, 0)
		HH = health_history = Array{Vector{String}}(undef, 0)
		
		# Run simulation
		for i in 1:I
		    tmp = sim()
		    push!(AOD, tmp[1])      # First element of tuple
		    push!(LH, tmp[2])       # Second element of tuple
		    push!(HH, tmp[3])       # Third element of tuple
		end

		# Check dimensions and lengths
		# println("Dimensions of AOD:", ndims(AOD))
		# println("Length of AOD:", length(AOD))
		# println("Dimensions of LH:", ndims(LH))
		# println("Length of LH:", length(LH))
		# println("Dimensions of HH:", ndims(HH))
		# println("Length of HH:", length(HH))
	
		println("Life expectancy in this population: ", mean(AOD))
		
		results = (;age_of_death,living_history,health_history)
		return(results)
	end
end

# ╔═╡ 7804dcba-6adb-4d3b-9328-32a734acfacb
pop_results = popsim(200)

# ╔═╡ 561b8929-c453-4a75-a432-f29031b663ac
begin
	# Syntax : 
	# pop_results[2] # We acces the living status history of all agents
	# pop_results[2][i] # We acces the living status history of individual i
	# pop_results[2][i][t] # We acces the living status of individual i at period t
	# pop_results[2][1][1]
	# length(pop_results[2]) # Number of individuals
end

# ╔═╡ 5c105e50-d1b8-4af3-b510-588c1dbbc40e
begin
	plotlyjs()
	pop_results[2]
	pop_living_status = sum(pop_results[2][:])
	Plots.plot(pop_living_status, label = "Current living population")
	Plots.xlabel!("Time")
	Plots.ylabel!("Population")
	Plots.title!("Evolution of population over time.")
end

# ╔═╡ f73b41a5-01f5-4694-b34b-a4648d18fc91
md"# 5. Maximisation programs"

# ╔═╡ 3100a50b-f4f7-4cf9-8c74-edb2b5579779
md"## In one period

Considering a one period problem, we have : 

Utility function : 

$$u(c,l) = \frac{c^{1-\rho}}{1-\rho} - l\cdot \xi$$

Budget constraint : 

$$c \le l\cdot z$$

The Lagrangien is : 

$$\mathcal{L}(c,l,\lambda) = \frac{c^{1-\rho}}{1-\rho} - l\cdot \xi + \lambda \cdot(l\cdot z - c)$$

The FOC are : 

$$\begin{align*}
\begin{cases}
\frac{\partial \mathcal{L}}{\partial c} = c^{-\rho}-\lambda = 0 \\
\frac{\partial \mathcal{L}}{\partial l} = -\xi + \lambda\cdot z = 0
\end{cases}
&  \iff              &
\begin{cases}
c^{-\rho}=\lambda \\
\xi=\lambda\cdot z
\end{cases}\\
\end{align*}$$

$$\implies$$

$$c^{-\rho} \cdot z = \xi$$
$$\iff$$
$$c^{*}=\left(\frac{\xi}{z}\right)^{-\frac{1}{\rho}}=\left(\frac{z}{\xi}\right)^{\frac{1}{\rho}}$$

The first FOC implies that $\lambda>0$, meaning that the budget constraint binds. Therefore, we get :

$$\begin{align*}
c^{*}=l^{*}\cdot z \iff \left(\frac{z}{\xi}\right)^{\frac{1}{\rho}} = l^{*}\cdot z
\end{align*}$$

$$\iff l^{*}=z^{\frac{1-\rho}{\rho}}\cdot\xi^{-\frac{1}{\rho}}$$
"

# ╔═╡ 49d95aaa-b5cf-469c-b235-c1eb1ce776b8
md"In this one period context, the agent works so that the marginal benefit of consumption is equal to the marginal disutility of working."

# ╔═╡ 2d79fea9-af26-4ef9-b938-e40a868f5995
md" ## In multiple periods

Now, let us assume that the agents live several periods and can transmit savings from one period to the next.

First, we can define the utility function of the agent : 

$$u(c,l,w,h)=\frac{c^{1-\rho}}{1-\rho}-\phi(l,w,h)$$

With $$\phi$$ being the disutility function of working, such that : 

$$\phi(l,w,h) = \phi_{l}\cdot l+\phi_{w}\cdot l\cdot\mathbb{1}\{w=d\}+\phi_{h}\cdot l\cdot\mathbb{1}\{h=b\}$$

$$\iff$$

$$\phi(l,w,h) = l \cdot (\phi_{l}+\phi_{w}\cdot\mathbb{1}\{w=d\}+\phi_{h}\cdot\mathbb{1}\{h=b\})$$

For notation purposes, let us define:

$$\xi\equiv (\phi_{l}+\phi_{w}\cdot\mathbb{1}\{w=d\}+\phi_{h}\cdot\mathbb{1}\{h=b\})$$

We can thus rewrite: 

$$\phi(l,w,h) = l\cdot\xi$$

And therefore : 

$$u(c,l,w,h) = \frac{c^{1-\rho}}{1-\rho}-l\cdot\xi$$


"

# ╔═╡ 81ab2269-9a8d-4626-9f1f-4a193f4e7fb3
md"
### Sequential form

Let us take a sequential form approach.

Assuming the agent maximises over $T$ periods from their 18-th years, the discounted total utility the agent tries to maximise is : 

$$U = \max_{\{c_t,l_t,s_{t+1}\}_{t=18}^{T}}\quad\sum^{T}_{t=18}\beta^{t-18} \cdot \mathbb{E}\left[\frac{c_{t}^{1-\rho}}{1-\rho}-l_{t}\cdot\xi_{t}\right]$$

They are subject to a budget constraint such that : 

$$c_{t}+s_{t+1}\le l_{t}\cdot z_{t}+s_{t}$$

The sum of current consumption and savings transmitted to the next period must be less than the sum of the current labor income and the initial wealth, from the savings of last period.

Let us also impose that the agents cannot borrow, such that : 

$$s_{t}\geq 0\quad \forall t \in [\![18,T]\!]$$

We can therefore write the Lagrangien, such that:

$$\begin{split}
\mathcal{L}(c_t,l_t,s_{t+1},\lambda_{t},\gamma_{t})=&\sum^{T}_{t=18}\beta^{t-18} \cdot \mathbb{E}[\frac{c_{t}^{1-\rho}}{1-\rho}-l_{t}\cdot\xi_{t}\\+&\lambda_{t}\cdot(l_{t}\cdot z_{t}+s_{t}-c_{t}-s_{t+1})\\+&\gamma_{t}(s_{t}-0)]
\end{split}$$
"

# ╔═╡ b7e9a54c-a813-4864-bcf7-8437c4c6d69b
md"The F.O.C.s are : 

$$\newcommand{\pder}[2]{\frac{\partial#1}{\partial#2}}$$

$$
\begin{cases}
\pder{\mathcal{L}}{c_{t}} = \beta^{t-18}\cdot\left(c_{t}^{-\rho}-\lambda_{t}\right) = 0\\
\pder{\mathcal{L}}{l_{t}} = \beta^{t-18}\cdot\left(-\xi_{t}+\lambda_{t}\cdot z_{t}\right) = 0\\
\pder{\mathcal{L}}{s_{t+1}} = \beta^{t-18}\cdot(-\lambda_{t}) + \beta^{t-17}\cdot\mathbb{E}\left[\lambda_{t+1}+\gamma_{t+1}\right] = 0\\
\end{cases}$$
$$\iff$$
$$\begin{cases}
c_{t}^{-\rho} = \lambda_{t} \\
c_{t}^{-\rho}\cdot z_{t} = \xi_{t} \\
c_{t}^{-\rho} = \beta\cdot \mathbb{E}\left[c_{t+1}^{-\rho}+\gamma_{t+1}\right]
\end{cases}$$

At the optimum : 

- The marginal increase of the budget is equal to the marginal benefit of consuming one more unit, i.e. $c_{t}^{-\rho}$ utility.
- The marginal benefit of working is equal to the marginal disutility of working. 
- The marginal benefit of consuming at a period is equal to the discounted expected marginal benefit of consuming in the next period (assuming that $\gamma_t = 0, \forall t$).

"

# ╔═╡ 16f8acf3-c913-4a06-85b6-3111e18a7494
md"
From the second equation, we have: 

$$c^{-\rho}_{t}\cdot z_t = \xi_{t} \iff c_{t}^{*}=\left(\frac{z_t}{\xi_t}\right)^{\frac{1}{\rho}}$$


Since the budget constraint binds, we have: 

$$\left(\frac{z_t}{\xi_t}\right)^{\frac{1}{\rho}}+s_{t+1}=l_{t}\cdot z_{t}+s_{t}$$
$$\iff$$
$$l_{t}\cdot z_{t}=\left(\frac{z_t}{\xi_t}\right)^{\frac{1}{\rho}}+s_{t+1}-s_{t}$$

Let us define:

$$\Delta s_{t+1}\equiv s_{t+1}-s_{t}$$

$$l_{t}=z_{t}^{-1}\left[\left(\frac{z_t}{\xi_t}\right)^{\frac{1}{\rho}}+\Delta s_{t+1}\right]$$
$$\iff$$
$$l_{t}^{*}=z_{t}^{\frac{1-\rho}{\rho}}\cdot\xi_{t}^{-\frac{1}{\rho}}+z_{t}^{-1}\Delta s_{t+1}$$


"

# ╔═╡ 182d2fe5-14f8-4262-b8cf-672f8a1c6f9b
md"

### Recursive form

We can let our Bellman equation be such as: 

$$\begin{split}
V(h,w,\text{age},s,z) = \max_{c,l,s'}\{u(c,h,l,w) + \beta\cdot\mathbb{E}_{w,h,ζ}\left[V(h',w',\text{age}',s',z')\right]\} \quad (1)
\end{split}$$

Let us define : $$\mathbb{S}\equiv (h,w,\text{age},s,z)$$. 

We can now write $$(1)$$ such as :

$$\begin{split}
V(\mathbb{S}) = \max_{c,l,s'}\{u(c,h,l,w) + \beta\cdot\mathbb{E}_{w,h,\zeta}\left[V(\mathbb{S}')\right]\}
\quad \quad (1)
\end{split}$$

subject to : 

$$c + s' \leq l\cdot z + s\quad (2) $$ 

With $s$ being the savings transferred to period $t$ such that: 

$$s\geq0 \quad \quad \quad \quad \quad \quad (3)$$

The equation $(2)$ being the budget constraint, and $(3)$ translating the fact that agents do not borrow in this model. The equation $(4)$ is the productivity of the agent.

"

# ╔═╡ 76a99e4f-1287-4fb6-9873-1ffc5db9d88e
md"
Assuming that $(2)$ binds, we can write : 

$$c+s'=l\cdot z + s \iff c = l\cdot z +s- s'$$

And therefore rewrite the maximisation program : 

$$\begin{split}
V(\mathbb{S}) = \max_{l,s'}\{u(l\cdot z +s- s',h,l,w) + \beta\cdot\mathbb{E}_{w,h,ζ}\left[V(\mathbb{S}')\right]\} \quad (1)
\end{split}$$

"

# ╔═╡ 496514b8-bcaa-4c06-844e-082eb4dfdbca
md"
Writing explicitly the utility function, we have: 

$$\newcommand{\disutility}{l \cdot (\phi_{l}+\phi_{w}\cdot\mathbb{1}\{w=d\}+\phi_{h}\cdot\mathbb{1}\{h=b\})}$$
$$\newcommand{\utility}{\frac{(l\cdot z +s- s')^{1-\rho}}{1-\rho}-\disutility }$$

$$\begin{split}
V(\mathbb{S}) = & \max_{l,s'}\Bigl\{\frac{(l\cdot z +s- s')^{1-\rho}}{1-\rho}-l\cdot\xi+\beta\cdot\mathbb{E}_{w,h,ζ}\left[V(\mathbb{S}')\right]\Bigl\} \quad (1)
\end{split}$$

"

# ╔═╡ 4219f0fe-bd6a-45c2-9ae9-2113a65d9e9e
md"
The First Order Conditions (F.O.C.) are : 

$$\frac{\partial V(\mathbb{S})}{\partial l} = 0 \quad (\text{F.O.C. } 1)$$
$$\iff$$
$$\frac{\partial u}{\partial c}\cdot\frac{\partial c}{\partial l}+\frac{\partial u}{\partial l}+\beta\cdot\mathbb{E}[\frac{\partial V(\mathbb{S}')}{\partial l}]=0$$
$$\iff$$
$$z\cdot c^{-\rho} - \xi + \beta\cdot\mathbb{E}[\frac{\partial V(\mathbb{S}')}{\partial l}]=0$$
$$\iff$$
$$z\cdot c^{-\rho} + \beta\cdot\mathbb{E}[\frac{\partial V(\mathbb{S}')}{\partial l}]=\xi \quad (\text{F.O.C. } 1)$$

In words, the marginal utility of working now plus the discounted marginal value of working in the future now is equal to the disutility of working now.



$$
	\begin{matrix}
	\text{marginal utility} \\ \text{of working now}
	\end{matrix} + 
	\begin{matrix} 
	\text{discounted expected future marginal} \\ \text{value of working now}
	\end{matrix} = \text{disutility of working now}
$$



Now, optimizing with respect to the savings decision : 

$$\frac{\partial V(\mathbb{S})}{\partial s'} = 0 \quad (\text{F.O.C. } 2)$$
$$\iff$$

$$\frac{\partial u}{\partial c}\cdot\frac{\partial c}{\partial s'}+ \beta \cdot \mathbb{E}[\frac{\partial V(\mathbb{S'})}{\partial s'}] = 0$$

$$\iff$$

$$c^{-\rho}\cdot (-1)+ \beta \cdot \mathbb{E}[\frac{\partial V(\mathbb{S'})}{\partial s'}] = 0$$
$$\iff$$
$$\beta \cdot \mathbb{E}[\frac{\partial V(\mathbb{S'})}{\partial s'}] = c^{-\rho}\quad (\text{F.O.C. } 2)$$
"

# ╔═╡ 2acf58c3-3227-4ee6-a72e-1422accb1b98
md"
Now, for the first FOC : 

$$\dots \text{Unsolved ? }$$

From the Budget Constraint, we have : 

$$c + s' = l\cdot z + s$$

Isolating the savings, we obtain : 

$$s'= l\cdot z + s-c$$

Iterating the budget constraint, we get : 

$$c' + s'' = l'\cdot z' + s'$$

Plugging the isolated expression of $s'$, we get : 

$$c' + s'' = l'\cdot z' + l\cdot z + s-c$$

$$\iff$$
$$c' = l'\cdot z' + l\cdot z + s-c-s''$$

Plugging it into the Bellman equation, we obtain : 

$$\frac{\partial V(\mathbb{S}')}{\partial l} = \frac{\partial}{\partial l}\cdot \max_{l',s''}\{u(l'z'+lz+s-c-s'',h',l',w')+\beta\cdot\mathbb{E}[\frac{\partial V(\mathbb{S''})}{\partial l}]\}$$

$$\iff$$

$$\frac{\partial V(\mathbb{S}')}{\partial l} = z\cdot c^{-\rho}+\beta\cdot\mathbb{E}[\frac{V(\mathbb{S}'')}{\partial l}]$$

$$\dots$$

$$z\cdot c^{-\rho} + \beta\cdot\mathbb{E}[z\cdot c^{-\rho}+\beta\cdot\mathbb{E}[\frac{V(\mathbb{S}'')}{\partial l}]=\xi \quad (\text{F.O.C. } 1)$$


To obtain a similar result compared to the sequential form, we could assume that:

$$\frac{\partial V(\mathbb{S}')}{\partial l}=0$$

We then obtain:

$$z\cdot c^{-\rho} =\xi \quad (\text{F.O.C. } 1)$$

"

# ╔═╡ 2711fa92-9877-4441-8456-9c6c40e3fcad
md"
Now, for the second FOC :

$$\frac{\partial V(\mathbb{S})}{\partial s} = \frac{\partial u}{\partial c}\cdot \frac{\partial c}{\partial s} + \beta \cdot \mathbb{E}[\frac{\partial V(\mathbb{S}')}{\partial s}]$$

$$\frac{\partial V(\mathbb{S})}{\partial s} = \frac{\partial u}{\partial c}\cdot \frac{\partial c}{\partial s} = \frac{\partial u}{\partial c} = c^{-\rho}$$
$$\iff$$
$$\frac{\partial V(\mathbb{S}')}{\partial s'} = {c'}^{-\rho}$$

Therefore, the second FOC is : 

$$\beta \cdot \mathbb{E}[{c'}^{-\rho}] = c^{-\rho}\quad (\text{F.O.C. } 2)$$

This is the Euler equation.

"

# ╔═╡ e38e36a8-f410-4888-8fa9-eb907cacf127
md"## Attempt of visualisation : no savings

Let us now try to simulate the life of an agent that does not save.

We are going to assume that they consume: 

$$c^{*}_{t}=\left(\frac{z_t}{\xi_t}\right)^{\frac{1}{\rho}}$$

And that they work such that:

$$l^{*}_{t}=z_{t}^{-1}\left(c^{*}_{t}\right)=z_{t}^{-1}\cdot\left(\frac{z_t}{\xi_t}\right)^{\frac{1}{\rho}}$$
"

# ╔═╡ ffe89f7c-e31d-4f07-8d01-c3de52feaed9
md"The utility function and all its parameters are defined in Julia in the Appendix of the notebook."

# ╔═╡ 7ecb1a5a-dc5c-4319-947f-adf4b2a766f0
begin
	"""
	The function `choice_sim_nosavings(p)` simulates the choices of an individual that does not save.

	The parameter `p` is the probability of the weather begin in a deviation state.
	
	It returns a tuple containing: 

	1. Their age of death, 
	2. Their living status history (1 when living, 0 when being dead),
	3. Their health status history ("g" when good, "b" when bad).
	4. Their utility for each period. 
	5. Their consumption for each period. 
	6. Their labor supply for each period.
	7. Their productivity for each period.
	"""
	function choice_sim_nosavings(p)
		labor_history = zeros(100)
		living_history = zeros(100)
		utility_history = zeros(100)
		consumption_history = zeros(100)
		productivity_history = zeros(100)
		
		health_history = Vector{String}(undef,100)
		
		health_t_1 = String("g")
	
		for t in 1:100
	    
			# if t == 1
			# 	health_t_1 = "g"
			# end
	    
		    # The age : 
		    age = t
		    
		    # The weather ("n" or "d") :
		    weather_t = weather(p)
		    
		    # The health status : 
		    # probability of being in good health : 
		    pgh = probability_good_health = health_probability(health_t_1)
		    health_t = rand(Binomial(1,pgh)) == 1 ? "g" : "b"
			health_history[t] = health_t
			health_t_1 = health_t
		
		    # The living status : 
		    pd = probability_dying = ζ(age,health_t)
		    living_status = rand(Binomial(1,1-pd))
		    living_history[t] = living_status

			# Choices : 

			# Productivity (exogenous)
			productivity_t = productivity(health_t,weather_t,age)
			productivity_history[t] = productivity_t

			# Disutiliy : 
			ξ_t = ξ(weather_t,health_t)
			
			# Consumption : 
			# I detail the steps due to rounding error yielding complex numbers
			# The error is due to float operations yielding c1 as negative. 
			# I therefore use the abs() function to avoid it. 
			c1 = productivity_t/ξ_t
			c2 = 1/ρ
			consumption_t = abs(c1)^c2
				# consumption_t = ((productivity_t)/(ξ_t))^(1/ρ)
				# println("Error, value of parameters : ")
				# println("ξ = ", ξ_t)
				# println("z = ", productivity_t)
				# println("ρ = ", ρ)
				# return(;ξ_t,productivity_t,ρ)
				# break
			consumption_history[t] = consumption_t

			# Labor : 
			labor_t = consumption_t/productivity_t
			labor_history[t] = labor_t

			# Utility : 
			u_t = u(consumption_t,health_t,labor_t,weather_t)
			utility_history[t] = u_t

			# When death comes : 
			if living_status == 0
				# print("Agent died at ", t)
				results = (;age,
					living_history,
					health_history,
					utility_history,
					consumption_history,
					labor_history,
					productivity_history)
				return(results)
				break
			else
			end
			
		end
	end
end

# ╔═╡ 9b81ad5f-5f30-438a-8761-120193034e67
choice_single = choice_sim_nosavings(0.5)

# ╔═╡ aa2aa419-398f-4e36-aef6-4c7fb68e30bb
begin
	plotlyjs()
	# Optimal consumption :
	plot_51 = Plots.plot(18:100,
		choice_single[5][18:100],
		legend = false,
		# xaxis = "Age",
		yaxis = "Consumption")

	# Utility :
	plot_52 = Plots.plot(18:100,
		choice_single[4][18:100],
		legend = false,
		# xaxis = "Age",
		yaxis = "Utility")

	# Optimal work supply : 
	plot_53 = Plots.plot(18:100,
		choice_single[6][18:100],
		legend = false,
		xaxis = "Age",
		yaxis = "Work supply")

	# Productivity :
	plot_54 = Plots.plot(18:100,
		choice_single[7][18:100],
		legend = false,
		xaxis = "Age",
		yaxis = "Productivity")
	
	Plots.plot(plot_51, plot_52, plot_53, plot_54)
end

# ╔═╡ 6eeab653-8f96-48db-abaf-e9e9f4932a5a
begin
	plotlyjs()
	Plots.plot(18:100,choice_single[5][18:100], label = "Consumption")
	Plots.plot!(18:100,choice_single[4][18:100], label = "Utility")
	Plots.plot!(18:100,choice_single[6][18:100], label = "Work supply")
	Plots.plot!(18:100,choice_single[7][18:100], label = "Productivity")
	# Labor income is consumption here, since there is no savings. 
	Plots.plot!(xaxis = "Age")
end

# ╔═╡ fe9b7ac6-305a-4fdf-9c21-b5fda8fa46d5
md"### Aggregate simulation


Let us now try to simulate the choices and utility of several agents, without savings. We have to be more careful this time: contrary to a population simulation focusing on survival, the consumption is affected by weather directly through productivity. In this sense, we need to have a common weather history for a given population in order to compare different populations. We need to reprogram a population simulation, and not just loop over the `choice_sim_nosavings()` function.
"

# ╔═╡ cf36d6c7-e4d4-418c-9fce-40479ff55898
begin
	"""
	The function `choice_sim_nosavings_population(I::Integer,p::Number)` simulates the consumption and labor supply choices of `I` individuals within a world with a probabiliy `p` of the weather deviating.

	It returns an array of size `I`, with each element containing: 

	1. Their age of death,

	2. Their living status history (1 when living, 0 when being dead),
	
	3. Their health status history ("g" when good, "b" when bad).
	
	4. Their utility for each period.
	
	5. Their consumption for each period.
	
	6. Their labor supply for each period.
	
	7. Their productivity for each period.
	"""
	function choice_sim_nosavings_population(I::Integer,p::Number)

		Population = Array{NamedTuple}(undef,I)

		# Weather history common to the whole population 
		weather_history = Array{String}(undef,100)
		for t in 1:100
			weather_history[t] = weather(p)
		end

		# Now we simulate all the individuals:
		for i in 1:I

			# Initialisation of the vectors: 
			labor_history = zeros(100)
			living_history = zeros(100)
			utility_history = zeros(100)
			consumption_history = zeros(100)
			productivity_history = zeros(100)
			
			health_history = Vector{String}(undef,100)

			# the initial health status is good: 
			health_t_1 = String("g")

			# They begin their choice at 18 years old:
			for t in 18:100
				
			    # The age : 
			    age = t
			    
			    # The weather comes from the common weather history :
			    weather_t = weather_history[t]
			    
			    # The health status : 
			    # probability of being in good health : 
			    pgh = probability_good_health = health_probability(health_t_1)
			    health_t = rand(Binomial(1,pgh)) == 1 ? "g" : "b"
				health_history[t] = health_t
				health_t_1 = health_t
			
			    # The living status : 
			    pd = probability_dying = ζ(age,health_t)
			    living_status = rand(Binomial(1,1-pd))
			    living_history[t] = living_status
	
				# Choices : 
	
				# Productivity (exogenous)
				productivity_t = productivity(health_t,weather_t,age)
				productivity_history[t] = productivity_t
	
				# Disutiliy : 
				ξ_t = ξ(weather_t,health_t)
				
				# Consumption : 
				# I detail the steps due to rounding error yielding complex numbers
				# The error is due to float operations yielding c1 as negative. 
				# I therefore use the abs() function to avoid it. 
				c1 = productivity_t/ξ_t
				c2 = 1/ρ
				consumption_t = abs(c1)^c2
				consumption_history[t] = consumption_t
	
				# Labor : 
				# Recall that we are in a no savings world.
				labor_t = consumption_t/productivity_t
				labor_history[t] = labor_t
	
				# Utility : 
				u_t = u(consumption_t,health_t,labor_t,weather_t)
				utility_history[t] = u_t
	
				# When death comes : 
				if living_status == 0
					# print("Agent died at ", t)
					results_of_i = (;age,
						living_history,
						health_history,
						utility_history,
						consumption_history,
						labor_history,
						productivity_history)
					Population[i] = results_of_i
					break
				end
			end # End of time for individual i
		end # End of individuals 
		results = (;weather_history,Population)
		return results
	end # End of function 
end	# End of block

# ╔═╡ 235c6844-1f7d-46b7-bb72-5fdd0a43b8eb
md"Within such a very simplisic model, we can run the simulations for different probability of weather deviation, and compare the outcomes in terms of aggregate utility. Here is an example with the aggregate uility of a society with a probability of he weather staying normal of 0.5."

# ╔═╡ c531f1c8-906a-434d-ad8f-3438ddb8a99d
begin
	# The syntax is such that : 
	
	# pop_choices_5[1] # will yield the weather history.
	
	# pop_choices_5[2] # will yield individuals data
	# pop_choices_5[2][i] # will yield individuals i data
	# pop_choices_5[2][1][:living_history] # will yield individuals i data
	# pop_choices_5[2][1][:age] # will yield individuals i data of age of death
	# length(pop_choices_5[2]) # will yield the number of individuals
	# typeof(pop_choices[1,:])

	# typeof(pop_choices[1,:]) # Is a vector of one tuple
	# typeof(pop_choices[1,:][1]) # Is a tuple of 7 elements, coming from choice_sim_nosavings()
	
	# pop_choices[i,:][1] # will yield the same data of the i-th individual
	# pop_choices[1,:][1][d] # will yield the d-th data of the i-th individual
	# pop_choices[1,:][1][1] # age of death
	# pop_choices[1,:][1][2] # living history
	# pop_choices[1,:][1][3] # health history
	# etc...
end

# ╔═╡ 36716607-70b2-444f-88bb-6f8db70f542e
begin
	plotlyjs()

	# To sum up the utilities :
	function aggregating(I::Integer,p::Number)

		x = choice_sim_nosavings_population(I,p)
		
		# Get the number of time periods
		number_of_periods = length(x[2][1][:living_history])
		
		# Initialize an array to store sums
		sum_of_utilities = zeros(number_of_periods)
			
		# Sum utilities across individuals for each time period
		for t in 1:number_of_periods
		    for i in 1:length(x[2]) # for all individuals	
		        sum_of_utilities[t] += x[2][i][:utility_history][t]
		    end
		end	
		
		return sum_of_utilities
	end

	pop_choices_5 = aggregating(2000,0.5)

	Plots.plot(1:100,pop_choices_5[1:100], legend = false)
	Plots.plot!(xaxis = "Periods", yaxis = "Aggregate utility")
	Plots.plot!(title = "Aggregate utility over time with p = 0.5")
end

# ╔═╡ 57681a33-f184-405d-8a27-c370a9bcb0ed
md"Now, comparing across societies, with different probabilities of weather being normal, we get:"

# ╔═╡ 834e8cba-3424-4722-b4b8-3e80cab92240
begin
	# Generating data:
	simulations = Array{Vector{Float64}}(undef, 11)
	for (index,probability) in zip(1:1:11,0.00:0.10:1.00)
		simulations[index] = aggregating(2000,probability)
	end

	# Plotting : 
	plotlyjs()
	plot_all = Plots.plot(1:100,simulations[1],label = "0.1")
	for (index,probability) in zip(1:1:11,0.00:0.10:1.00)
		Plots.plot!(simulations[index], label = probability)
	end
	plot_all
end

# ╔═╡ 01b81ae5-eba8-450a-8519-bddb29476628
md"With only the extreme values : " 

# ╔═╡ e7293adc-8fea-4b9b-a688-83bb02292268
begin 
	plotlyjs()
	pop_choices_full_normal = aggregating(2000,1)
	pop_choices_full_deviation = aggregating(2000,0)
	exported_plot_1 = Plots.plot(1:100,pop_choices_full_normal, label = "Perfectly normal weather")
	Plots.plot!(pop_choices_full_deviation, label = "Totally deviating weather")
	Plots.plot!(xaxis = "Period", yaxis = "Aggregated utility")
	Plots.savefig("Framework_model_1plot1.png")
	exported_plot_1
end

# ╔═╡ 91a32ee5-cf11-45c7-bef7-f3747610a2a8
md"If we sum up all periods, to get the aggregated intertemporal utility, we obtain:"

# ╔═╡ 91595a6c-39da-4190-8e6c-de462f6b1f60
begin
	Probabilities = [0.00:0.10:1.00]
	Sum_Utilities = [sum(simulations[1]), 
					sum(simulations[2]),
					sum(simulations[3]),
					sum(simulations[4]),
					sum(simulations[5]),
					sum(simulations[6]),
					sum(simulations[7]),
					sum(simulations[8]),
					sum(simulations[9]),
					sum(simulations[10]),
					sum(simulations[11])]
	exported_plot_2 = Plots.bar(Probabilities,Sum_Utilities, legend = false)
	Plots.plot!(xaxis = "Probabiliy of weather being normal",
	 			yaxis = "Aggregated intertemporal utility")
	Plots.savefig("Framework_model_1plot2.png")
	exported_plot_2
end

# ╔═╡ 33086f8a-3ce2-4e28-ba36-49b233b8c692
md"We touch here the core of what my maser thesis will try to reach. Run a model with several meteorological parameters conditioning the probabiliy of a weather deviation, and seeing how it affects the economy.

In this simplified model, the weather is binary and does not affect the probability of dying. The productivity is only affected by the weather through a constant effect. Later, the weather will be modelised more in details. "

# ╔═╡ 6f0c1138-61a2-48df-8c82-09de7533e28f
md"# 6. Appendix

## Parameters

Here can be found the utility function and its corresponding parameters of risk aversion and different penalty coefficients associated with labor disutility. 

Changing them will affect the rest of the above simulations using the utility function. 
"

# ╔═╡ 23ecebf6-22ff-43a0-97a2-62718f5ebab5
# Risk aversion :
@bind ρ Slider(0.00:0.01:1, default=0.9)

# ╔═╡ 1bcb39cf-0a2b-453e-845a-e22caa1bd23b
# Working penalty :
@bind ϕ_l Slider(0.00:0.01:1, default=0.9)

# ╔═╡ 27ec39f4-aca3-41da-a6ed-b6bb4a6b1035
# Working with weather deviation penalty :
@bind ϕ_w Slider(0.00:0.01:1, default=0.7)

# ╔═╡ e76cb128-d03e-465b-bbb4-9abee96d9249
# Working in bad health penalty :
@bind ϕ_h Slider(0.00:0.01:1, default=0.9)

# ╔═╡ f1d6cd8a-1588-4b2a-b7d4-20dbc8b7dfe5
begin
	# Indicator function of working : 
	indicator_function_work(l) = l > 0 ? 1 : 0

	# Disutility function : 
	ξ(w,h) = ϕ_l + ϕ_w * indicator_function_weather_deviation(w) + ϕ_h * indicator_function_bad_health(h)
	
	ϕ(h,l,w) = l*ξ(w,h)	

	# Utility function : 
	u(c,h,l,w) = (c^(1-ρ))/(1-ρ) - ϕ(h,l,w)
end

# ╔═╡ 1df2fc95-9b98-4e7f-95ef-42974f554c6a
md"
## One-individual discrepancies "

# ╔═╡ df9a5201-c5ff-4b32-a913-746bae1c691a
md"In this last section, I include a visualisaion of the utility function for individuals differently affected by weather and health status. This allows to understand at a one-individual-level where the discrepancies observed in the aggregated version come from."

# ╔═╡ a9e401d2-91ed-49b3-a4e7-318bc2119306
md"Plotting the utility for good health and bad health agents for a given value of labor supply, we get : "

# ╔═╡ 264be045-c2fa-4cff-ab0a-3d8e79675875
# Amount of work :
@bind l Slider(0.00:0.01:10, default=0.5)

# ╔═╡ aa596cad-807d-4317-a83e-4988cc6c4dcb
md"When we vary the level of work, through the slider of the variable `l`, we see that individuals will obtain very different utility levels in function of these condiions."

# ╔═╡ 2496b1d6-5738-4568-bf3d-c92468d7e900
begin 
	plotlyjs()
	Plots.plot(u.(1:100,"g",l,"n"), label = "Good health, normal weather")
	Plots.plot!(u.(1:100,"b",l,"n"), label = "Bad health, normal weather")
	Plots.plot!(u.(1:100,"g",l,"d"), label = "Good health, deviation weather")
	Plots.plot!(u.(1:100,"b",l,"d"), label = "Bad health, deviation weather")
	Plots.title!("Utility levels of agents for the same amount of work.")
	Plots.xlabel!("Consumption")
	Plots.ylabel!("Utility")
	Plots.plot!(legend=:bottomright)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Plotly = "58dd65bb-95f3-509e-9936-c39a10fdeae7"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
DataFrames = "~1.7.0"
Distributions = "~0.25.117"
Plotly = "~0.4.1"
PlotlyJS = "~0.18.15"
Plots = "~1.40.9"
PlutoUI = "~0.7.61"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "2df9e42c3f8e8b72868f8b54ab7bac43961c9aae"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[deps.Blink]]
deps = ["Base64", "BinDeps", "Distributed", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Reexport", "Sockets", "WebIO", "WebSockets"]
git-tree-sha1 = "08d0b679fd7caa49e2bca9214b131289e19808c0"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "03aa5d44647eaec98e1920635cdfed5d5560a8b9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.117"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "0ff136326605f8e06e9bcf085a356ab312eef18a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.13"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "9cb62849057df859575fc1dda1e91b82f8609709"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.13+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "2bd56245074fab4015b9174f24ceba8293209053"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.27"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd714447457c660382fe634710fb56eb255ee42e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.6"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "3b2db451a872b20519ebb0cec759d3d81a1c6bcb"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.20"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "Pkg", "Sockets", "WebSockets"]
git-tree-sha1 = "82dfb2cead9895e10ee1b0ca37a01088456c4364"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "0.7.6"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a9697f1d06cc3eb3fb3ad49cc67f2cfabaac31ea"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plotly]]
deps = ["Base64", "DelimitedFiles", "HTTP", "JSON", "PlotlyJS", "Reexport"]
git-tree-sha1 = "044a9194ae38a50cbdb34a05dc63bf68e4db95df"
uuid = "58dd65bb-95f3-509e-9936-c39a10fdeae7"
version = "0.4.1"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "90af5c9238c1b3b25421f1fdfffd1e8fca7a7133"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.20"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "PlotlyKaleido", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "e415b25fdec06e57590a7d5ac8e0cf662fa317e2"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.15"

    [deps.PlotlyJS.extensions]
    CSVExt = "CSV"
    DataFramesExt = ["DataFrames", "CSV"]
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyJS.weakdeps]
    CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlotlyKaleido]]
deps = ["Artifacts", "Base64", "JSON", "Kaleido_jll"]
git-tree-sha1 = "ba551e47d7eac212864fdfea3bd07f30202b4a5b"
uuid = "f2990250-8cf9-495f-b13a-cce12b45703c"
version = "2.2.6"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "dae01f8c2e069a683d3a6e17bbae5070ab94786f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.9"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "0eef0765186f7452e52236fa42ca8c9b3c11c6e3"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.21"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "f91a602e25fe6b89afc93cf02a4ae18ee9384ce3"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.5.9"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "ee6f41aac16f6c9a8cab34e2f7a200418b1cc1e3"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56c6604ec8b2d82cc4cfe01aa03b00426aac7e1f"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+1"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "055a96774f383318750a1a5e10fd4151f04c29c5"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.46+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─d532165b-d442-4114-a3b9-f866a281c67f
# ╟─ab992a5b-6a2c-4457-a651-c6818df8f5e2
# ╟─0ef87638-e32f-11ef-1ac0-094df082b41c
# ╠═0d381a45-b8a7-4ce0-a904-b238fd26902d
# ╟─4c2a0eb5-6bf6-4b8b-9c1d-51107af68743
# ╟─1d30ee07-9993-4cdf-bea0-f318b5461079
# ╠═33467e51-029b-4fc5-a889-ee8a6e108a7e
# ╠═95d6c246-2773-4141-b53a-428364936b8a
# ╠═f2dcf131-4c96-4259-85eb-c78f2ea1b700
# ╠═9cce5faf-f860-494d-8de7-87f0c698e1b7
# ╟─ea3e3f1f-6e07-470f-bd6c-bb7e12bb9ed8
# ╠═22220a91-4f5a-4fdc-bd84-f0d79f911034
# ╟─a65ab9ac-f91c-4c6c-a162-34a3fc3ced13
# ╠═8eff2abf-254e-4336-b2b5-7d46a75279f3
# ╠═dc128964-6f99-4695-8c39-4b688b6041d8
# ╟─13259939-4338-44c0-904f-2d2a73a25879
# ╠═5439a417-bde0-4255-b310-09140dfe2eda
# ╠═4d6ec0eb-f023-451f-80ba-14b6fb3f1e24
# ╠═4dec6012-eb5c-4db2-bb95-e67e29ef4532
# ╠═156417da-bd77-425a-a41d-2f5ef31c339c
# ╠═4748644a-e1ef-4e1d-9e43-b0e566a74d0a
# ╠═7c1b788b-9d94-4d3a-93ce-385bb8a3d3ae
# ╟─edf27a0c-9e66-46f1-918b-b404e1b6dc7a
# ╟─a79e4de7-ad33-4043-a416-2ce3952bbafa
# ╟─6b7607e3-ced8-4549-9287-e4fc0d578908
# ╠═4f550514-f478-4cce-9cea-2f846c668bcf
# ╠═c54d7f45-46dd-4cc8-ac82-141427fccc3f
# ╠═3ad533a5-3f3c-46d3-9229-208679e7f01a
# ╠═00629ad3-fe8e-403a-8d94-35a74a263447
# ╠═c2f427aa-9e39-4957-982a-6e863f4a4853
# ╠═3d11cbe4-e503-43a7-8bba-6d0a590ef980
# ╠═a7170b37-86d2-4e89-92a1-ef3c19793776
# ╠═1a578c44-650f-4b5c-8c5a-bbc47ad3a904
# ╟─41e9ad1c-248a-4fd4-8c81-c037f213171b
# ╠═4a88de81-0e41-4889-a45b-d303fa6dbfcf
# ╠═3f7790ab-248d-43f6-9b8c-5fe50e034cc3
# ╟─ccc5887d-f8b9-4851-9299-043272b9336a
# ╠═24de739d-d053-4e41-83dd-0e076f2fe471
# ╠═7804dcba-6adb-4d3b-9328-32a734acfacb
# ╟─561b8929-c453-4a75-a432-f29031b663ac
# ╠═5c105e50-d1b8-4af3-b510-588c1dbbc40e
# ╟─f73b41a5-01f5-4694-b34b-a4648d18fc91
# ╟─3100a50b-f4f7-4cf9-8c74-edb2b5579779
# ╟─49d95aaa-b5cf-469c-b235-c1eb1ce776b8
# ╠═2d79fea9-af26-4ef9-b938-e40a868f5995
# ╟─81ab2269-9a8d-4626-9f1f-4a193f4e7fb3
# ╟─b7e9a54c-a813-4864-bcf7-8437c4c6d69b
# ╟─16f8acf3-c913-4a06-85b6-3111e18a7494
# ╠═182d2fe5-14f8-4262-b8cf-672f8a1c6f9b
# ╟─76a99e4f-1287-4fb6-9873-1ffc5db9d88e
# ╠═496514b8-bcaa-4c06-844e-082eb4dfdbca
# ╠═4219f0fe-bd6a-45c2-9ae9-2113a65d9e9e
# ╟─2acf58c3-3227-4ee6-a72e-1422accb1b98
# ╠═2711fa92-9877-4441-8456-9c6c40e3fcad
# ╟─e38e36a8-f410-4888-8fa9-eb907cacf127
# ╟─ffe89f7c-e31d-4f07-8d01-c3de52feaed9
# ╠═7ecb1a5a-dc5c-4319-947f-adf4b2a766f0
# ╠═9b81ad5f-5f30-438a-8761-120193034e67
# ╟─aa2aa419-398f-4e36-aef6-4c7fb68e30bb
# ╠═6eeab653-8f96-48db-abaf-e9e9f4932a5a
# ╟─fe9b7ac6-305a-4fdf-9c21-b5fda8fa46d5
# ╠═cf36d6c7-e4d4-418c-9fce-40479ff55898
# ╟─235c6844-1f7d-46b7-bb72-5fdd0a43b8eb
# ╟─c531f1c8-906a-434d-ad8f-3438ddb8a99d
# ╠═36716607-70b2-444f-88bb-6f8db70f542e
# ╟─57681a33-f184-405d-8a27-c370a9bcb0ed
# ╠═834e8cba-3424-4722-b4b8-3e80cab92240
# ╟─01b81ae5-eba8-450a-8519-bddb29476628
# ╟─e7293adc-8fea-4b9b-a688-83bb02292268
# ╟─91a32ee5-cf11-45c7-bef7-f3747610a2a8
# ╟─91595a6c-39da-4190-8e6c-de462f6b1f60
# ╟─33086f8a-3ce2-4e28-ba36-49b233b8c692
# ╟─6f0c1138-61a2-48df-8c82-09de7533e28f
# ╠═f1d6cd8a-1588-4b2a-b7d4-20dbc8b7dfe5
# ╠═23ecebf6-22ff-43a0-97a2-62718f5ebab5
# ╠═1bcb39cf-0a2b-453e-845a-e22caa1bd23b
# ╠═27ec39f4-aca3-41da-a6ed-b6bb4a6b1035
# ╠═e76cb128-d03e-465b-bbb4-9abee96d9249
# ╟─1df2fc95-9b98-4e7f-95ef-42974f554c6a
# ╟─df9a5201-c5ff-4b32-a913-746bae1c691a
# ╟─a9e401d2-91ed-49b3-a4e7-318bc2119306
# ╠═264be045-c2fa-4cff-ab0a-3d8e79675875
# ╟─aa596cad-807d-4317-a83e-4988cc6c4dcb
# ╟─2496b1d6-5738-4568-bf3d-c92468d7e900
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
