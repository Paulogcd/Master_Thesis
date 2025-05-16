now_good = backwards_numerical(s_range = s_range_2,
                        sprime_range		= s_range_2,
                        consumption_range 	= consumption_range,
                        labor_range			= labor_range,
                        nperiods 			= T,
                        r 					= small_r,
                        z 					= z,
                        w 					= weather_path_good,
                        proba_survival 		= average_proba_good,
                        h 					= average_health_good,
                        ρ 					= 1.50,
                        φ 					= 2.00,
                        β 					= β)

now_bad = backwards_numerical(s_range = s_range_2,
                        sprime_range		= s_range_2,
                        consumption_range 	= consumption_range,
                        labor_range			= labor_range,
                        nperiods 			= T,
                        r 					= small_r,
                        z 					= z,
                        w 					= weather_path_bad,
                        proba_survival 		= average_proba_bad,
                        h 					= average_health_bad,
                        ρ 					= 1.50,
                        φ 					= 2.00,
                        β 					= β)

# Lifetime income and scaling:
# lifetime_income_normalized = compute_lifetime_income(
#     boomers.optimal_choices,
#     s_range_2,
#     z,
#     average_proba_intermediate,
#     small_r,
#     0.00)

lifetime_income_normalized

lifetime_bad = compute_lifetime_income(
    now_bad.optimal_choices,
    s_range_2,
    z,
    average_proba_intermediate,
    small_r,
    0.00)

scaling_factor = 1_450_000 / lifetime_income_normalized
s_range_scaled = s_range_2 .* scaling_factor
consumption_range_scaled = range(0.0, stop=5.0*scaling_factor, length=200)  # 200 points
z_scaled = z .* scaling_factor  # Rescale wages

