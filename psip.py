import os
from pyomo.environ import *
import pha

def define_arguments(argparser):
    argparser.add_argument('--psip-relax', action='store_true', default=False, 
        help="Relax PSIP plans (retiring AES and building certain technologies) to find a more optimal strategy.")

def define_components(m):
    ###################
    # resource rules to match HECO's 2016-04-01 PSIP
    ##################
    
    # decide whether to enforce the PSIP preferred plan
    # if an environment variable is set, that takes precedence 
    # (e.g., on a cluster to override options.txt)
    psip_env_var = os.environ.get('USE_PSIP_PLAN')
    if psip_env_var is None:
        # no environment variable; use the --psip-relax flag
        psip = m.options.psip_relax
    elif psip_env_var.lower() in ["1", "true", "y", "yes", "on"]:
        psip = True
    elif psip_env_var.lower() in ["0", "false", "n", "no", "off"]:
        psip = False
    else:
        raise ValueError('Unrecognized value for environment variable USE_PSIP_PLAN={} (should be 0 or 1)'.format(psip_env_var))

    if psip:
        print "Using PSIP construction plan."
    else:
        print "Relaxing PSIP construction plan."
    
    # force conversion to LNG in 2021
    # force use of containerized LNG
    # don't allow addition of anything other than those specified here
    # force retirement of AES in 2021
    
    # targets for individual generation technologies
    # (year, technology, MW added)
    # TODO: allow either CentralFixedPV or CentralTrackingPV for utility-scale solar
    # (not urgent now, since CentralFixedPV is not currently in the model)

    # projects that are definitely being built (we assume near-term are underway
    # and military projects are being built for their own reasons)
    technology_targets_definite = [ 
        (2016, 'CentralTrackingPV', 27.6),  # Waianae Solar by Eurus Energy America
        (2018, 'IC_Schofield', 54.0),
        (2018, 'IC_Barge', 100.0),         # JBPHH plant
        (2021, 'IC_MCBH', 27.0), 
        # Distributed PV from Figure J-19
        (2016, 'DistPV',  443.993468266547 - 210), # net of 210 MW of pre-existing DistPV
        (2017, 'DistPV',  92.751756737742),
        (2018, 'DistPV',  27.278236032368),
        (2019, 'DistPV',  26.188129564885),
        (2020, 'DistPV',  21.8245069017911),
        (2021, 'DistPV',  15.27427771741),
        (2022, 'DistPV',  12.0039583149589),
        (2023, 'DistPV',  10.910655054315),
        (2024, 'DistPV',  10.913851847475),
        (2025, 'DistPV',  10.910655054316),
        (2026, 'DistPV',  9.82054858683205),
        (2027, 'DistPV',  10.910655054316),
        (2028, 'DistPV',  10.910655054315),
        (2029, 'DistPV',  14.1873680430859),
        (2030, 'DistPV',  9.82054858683205),
        (2031, 'DistPV',  10.913851847475),
        (2032, 'DistPV',  9.82054858683193),
        (2033, 'DistPV',  14.1841712499261),
        (2034, 'DistPV',  7.64033565186492),
        (2035, 'DistPV',  13.094064782442),
        (2036, 'DistPV',  9.82054858683205),
        (2037, 'DistPV',  10.9202454337949),
        (2038, 'DistPV',  9.66989970917803),
        (2039, 'DistPV',  12.1514103994531),
        (2040, 'DistPV',  12.2397218104919),
        (2041, 'DistPV',  11.7673956211361),
        (2042, 'DistPV',  10.9106550543149),
        (2043, 'DistPV',  9.82054858683205),
        (2044, 'DistPV',  15.27747451057),
        (2045, 'DistPV',  10.291675978754),
    ]
    # projects proposed in PSIP but which may not be built if a better plan is found
    technology_targets_psip = [     
        (2018, 'OnshoreWind', 24),      # NPM wind
        (2018, 'CentralTrackingPV', 109.6),  # replacement for canceled SunEdison projects
        (2018, 'OnshoreWind', 10),      # CBRE wind
        (2018, 'CentralTrackingPV', 15),  # CBRE PV
        (2020, 'OnshoreWind', 30),
        (2020, 'CentralTrackingPV', 60),
        (2021, 'CC_383', 383.0),
        (2030, 'CentralTrackingPV', 100),
        (2030, 'OffshoreWind', 200),
        (2040, 'CentralTrackingPV', 200),
        (2040, 'OffshoreWind', 200),
        (2045, 'CentralTrackingPV', 300),
        (2045, 'OffshoreWind', 400),
    ]
    
    if psip:
        technology_targets = technology_targets_definite + technology_targets_psip
    else:
        technology_targets = technology_targets_definite

    def technology_target_init(m, per, tech):
        """Find the amount of each technology that is targeted to be built by the end of each period."""
        start = 2000 if per == m.PERIODS.first() else per
        end = per + m.period_length_years[per]
        target = sum(
            mw for (tyear, ttech, mw) in technology_targets
                if ttech == tech and start <= tyear and tyear < end
        )
        return target
    m.technology_target = Param(m.PERIODS, m.GENERATION_TECHNOLOGIES, initialize=technology_target_init)

    # with PSIP: BuildProj is zero except for technology_targets 
    #     (sum during each period or before first period)
    # without PSIP: BuildProj is >= definite targets
    def Enforce_Technology_Target_rule(m, per, tech):
        """Enforce targets for each technology; exact target for PSIP cases, minimum target for non-PSIP."""
        build = sum(
            m.BuildProj[proj, per] 
                for proj in m.PROJECTS 
                    if m.proj_gen_tech[proj] == tech and (proj, per) in m.PROJECT_BUILDYEARS
        )
        target = m.technology_target[per, tech]
        if type(build) is int and build == 0:    # no matching projects found
            if target == 0:
                return Constraint.Skip
            else:
                print(
                    "WARNING: target was set for {} in {}, but no matching projects are available. "
                    "Model will be infeasible.".format(tech, per)
                )
                return Constraint.Infeasible
        elif psip:
            return (build == target)
        else:
            return (build >= target)
    m.Enforce_Technology_Target = Constraint(
        m.PERIODS, m.GENERATION_TECHNOLOGIES, rule=Enforce_Technology_Target_rule
    )

    aes_proj = 'Oahu_AES_GEN1'
    aes_size = 180
    aes_bld_year = 1992
    m.AES_OPERABLE_PERIODS = Set(initialize = lambda m:
        m.PROJECT_BUILDS_OPERATIONAL_PERIODS[aes_proj, aes_bld_year]
    )
    m.OperateAES = Var(m.AES_OPERABLE_PERIODS, within=Binary)
    m.Enforce_AES_Deactivate = Constraint(m.TIMEPOINTS, rule=lambda m, tp:
        Constraint.Skip if (aes_proj, tp) not in m.PROJ_DISPATCH_POINTS
        else (m.DispatchProj[aes_proj, tp] <= m.OperateAES[m.tp_period[tp]] * aes_size)
    )
    m.AESDeactivateFixedCost = Expression(m.PERIODS, rule=lambda m, per: 
        0.0 if per not in m.AES_OPERABLE_PERIODS
        else - m.OperateAES[per] * aes_size * m.proj_fixed_om[aes_proj, aes_bld_year]
    )
    m.cost_components_annual.append('AESDeactivateFixedCost')
    
    if psip:
        # keep AES active until 2022 or just before; deactivate after that
        m.PSIP_Retire_AES = Constraint(m.AES_OPERABLE_PERIODS, rule=lambda m, per:
            (m.OperateAES[per] == 1) if per + m.period_length_years[per] <= 2022
            else (m.OperateAES[per] == 0)
        )

        # before 2040: no biodiesel, and only 100-300 GWh of non-LNG fossil fuels
        # period including 2040-2045: <= 300 GWh of oil; unlimited biodiesel or LNG
        
        # no biodiesel before 2040 (then phased in fast enough to meet the RPS)
        m.EARLY_BIODIESEL_MARKETS = Set(dimen=2, initialize=lambda m: [
            (rfm, per) 
                for per in m.PERIODS if per + m.period_length_years[per] <= 2040
                    for rfm in m.REGIONAL_FUEL_MARKET if m.rfm_fuel == 'Biodiesel'
        ])
        m.NoEarlyBiodiesel = Constraint(m.EARLY_BIODIESEL_MARKETS, rule=lambda m, rfm, per:
            m.FuelConsumptionInMarket[rfm, per] == 0
        )

        # # 100-300 GWh of non-LNG fuels in 2021-2040 (based on 2016-04 PSIP fig. 5-5)
        # # Note: this is needed because we assume HECO plans to burn LNG in the future
        # # even in scenarios where it costs more than oil.
        # m.PSIP_HIGH_LNG_PERIODS = Set(initialize=lambda m:
        #     [per for per in m.PERIODS if per + m.period_length_years[per] > 2021 and per < 2045]
        # )
        # m.OilProductionGWhPerYear = Expression(m.PERIODS, rule=lambda m, per:
        #     sum(
        #         m.DispatchProjByFuel[proj, tp, f] * m.tp_weight_in_year[tp] * 0.001 # convert from MWh to GWh
        #             for f in ['Diesel', 'LSFO', 'LSFO-Diesel-Blend']
        #                 for proj in m.PROJECTS_BY_FUEL[f]
        #                     for tp in m.PERIOD_TPS[per] if (proj, tp) in m.PROJ_DISPATCH_POINTS
        #     )
        # )
        # m.Upper_Limit_Oil_Power = Constraint(m.PERIODS, rule=lambda m, per:
        #     (m.OilProductionGWhPerYear[per] <= 300)
        #         if per + 0.5 * m.period_length_years[per] >= 2021
        #     else
        #         Constraint.Skip
        # )
        # # lower limit is in place to roughly reflect HECO's plan
        # m.Lower_Limit_Oil_Power = Constraint(m.PERIODS, rule=lambda m, per:
        #     (m.OilProductionGWhPerYear[per] >= 100)
        #         if per + m.period_length_years[per] < 2040  # relax constraint if period ends after 2040
        #     else
        #         Constraint.Skip
        # )
        
        # force LNG conversion in 2021 (modeled on similar constraint in lng_conversion.py)
        # This could have extra code to skip the constraint if there are no periods after 2021,
        # but it is unlikely ever to be run that way.
        # Note: this is not needed if some plants are forced to run on LNG
        m.PSIP_Force_LNG_Conversion = Constraint(m.LOAD_ZONES, rule=lambda m, z:
                m.ConvertToLNG[
                    z,
                    min(per for per in m.PERIODS if per + m.period_length_years[per] > 2021)
                ] == 1
            )
        
        # # Kahe 5, Kahe 6, Kalaeloa and CC_383 only burn LNG after 2021
        # # This is not used because it creates a weird situation where HECO runs less-efficient non-LNG
        # # plants instead of more efficient LNG-capable plants on oil. 
        # # there may be a faster way to build this, but it's not clear what
        # m.PSIP_Force_LNG_Use = Constraint(m.PROJ_FUEL_DISPATCH_POINTS, rule=lambda m, proj, tp, fuel:
        #     (m.ProjFuelUseRate[proj, tp, fuel] == 0)
        #         if proj in m.LNG_CONVERTED_PLANTS
        #             and fuel != 'LNG'
        #             and m.tp_period[tp] + m.period_length_years[m.tp_period[tp]] > 2021
        #     else
        #         Constraint.Skip
        # )

        # don't allow construction of any advanced technologies (e.g., batteries, pumped hydro, fuel cells)
        advanced_tech_vars = [
            t for t in pha.build_vars 
                if t not in {"BuildProj", "BuildUnits", "ConvertToLNG", "RFMSupplyTierActivate"}
        ]
        def no_advanced_tech_rule_factory(v):
            return lambda m, *k: (getattr(m, v)[k] == 0)
        for v in advanced_tech_vars:
            var = getattr(m, v)
            setattr(m, "PSIP_No_"+v, Constraint(var._index, rule=no_advanced_tech_rule_factory(v)))

        # don't allow any changes to the fuel market, including bulk LNG
        m.PSIP_Deactivate_Limited_RFM_Supply_Tiers = Constraint(m.RFM_SUPPLY_TIERS,
            rule=lambda m, r, p, st:
                Constraint.Skip if (m.rfm_supply_tier_limit[r, p, st] == float('inf'))
                else (m.RFMSupplyTierActivate[r, p, st] == 0)
        )
