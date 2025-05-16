using NamedArrays

boomers = backwards_numerical(s_range = s_range_2,
                        sprime_range		= s_range_2,
                        consumption_range 	= consumption_range,
                        labor_range			= labor_range,
                        nperiods 			= T,
                        r 					= small_r,
                        z 					= z,
                        w 					= weather_path_boomer,
                        proba_survival 		= average_proba_boomer,
                        h 					= average_health_boomer,
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

now_intermediate = backwards_numerical(s_range = s_range_2,
                        sprime_range		= s_range_2,
                        consumption_range 	= consumption_range,
                        labor_range			= labor_range,
                        nperiods 			= T,
                        r 					= small_r,
                        z 					= z,
                        w 					= weather_path_intermediate,
                        proba_survival 		= average_proba_intermediate,
                        h 					= average_health_intermediate,
                        ρ 					= 1.50,
                        φ 					= 2.00,
                        β 					= β)

now_good = backwards_numerical(s_range      = s_range_2,
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


