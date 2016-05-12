#!/usr/bin/env python

import sys, os, argparse
from textwrap import dedent

import switch_mod.solve
import switch_mod.hawaii.scenario_data as scenario_data

parser = argparse.ArgumentParser()
parser.add_argument('--write-rhos', action='store_true', default=False, 
    help='Store rho settings proportional to objective coefficients for variables. Takes several minutes.')
parser.add_argument('--skip-cf', action='store_true', default=False,
    help='Skip writing variable capacity factors file (for faster execution)')
# note: inputs-dir and time-sample are not supported because this script defines three different ones
# parser.add_argument('--time-sample', default='rps_mini',
#     help='Name of time sample to use to create data; rps_mini is the default; tiny is good for testing')
# parser.add_argument('--inputs-dir', default='inputs')

cmd_line_args = parser.parse_args()

print "arguments:", cmd_line_args

# particular settings chosen for this case
# (these will be passed as arguments when the queries are run)
args = dict(
    # directory to store data in
    # inputs_dir = cmd_line_args.inputs_dir,     
    # skip writing capacity factors file if specified (for speed)
    skip_cf = cmd_line_args.skip_cf,    
    # could be 'tiny', 'rps', 'rps_mini' or possibly '2007', '2016test', 'rps_test_45', or 'main'
    # time_sample = cmd_line_args.time_sample,
    # subset of load zones to model
    load_zones = ('Oahu',),       
    # "hist"=pseudo-historical, "med"="Moved by Passion", "flat"=2015 levels, "PSIP_2016_04"=PSIP 4/16
    load_scen_id = "PSIP_2016_04", 
    # '1'=low, '2'=high, '3'=reference, 'EIA_ref'=EIA-derived reference level       
    fuel_scen_id = 'EIA_ref',      
    # Blazing a Bold Frontier, Stuck in the Middle, No Burning Desire, Full Adoption, 
    # Business as Usual, (omitted or None=none)
    ev_scenario = 'Blazing a Bold Frontier',   
    # should the must_run flag be converted to set minimum commitment for existing plants?
    enable_must_run = 0,     
    # list of technologies to exclude
    exclude_technologies = ('CentralFixedPV', 'DistPV_flat'),     
    base_financial_year = 2016,
    interest_rate = 0.06,
    discount_rate = 0.03,
    # used to convert nominal costs in the tables to real costs
    inflation_rate = 0.025,  
)

# TODO: change fuel_markets_expansion to support more complex supply chains,
# e.g., a regional facility (LNG switch) in line with a market upgrade (bulk LNG),
# and possibly additional upgrades beyond (e.g., adding a second FSRU).
# For now, we include the cost of the LNG switch via ad-hoc constraints in psip.py 
args.update(
    bulk_lng_fixed_cost = 1.75,     # fixed cost per MMBtu/year of capacity developed
    bulk_lng_limit = 43446735.1,    # limit on bulk LNG capacity (MMBtu/year)
)

# data for sodium sulfur batteries from Dropbox/kauai/OPL/Storage/power_plan.xlsx
# and http://www.energystoragenews.com/NGK%20Insulators%20Sodium%20Sulfur%20Batteries%20for%20Large%20Scale%20Grid%20Energy%20Storage.html
# This version was used for model runs before 2016-01-27
# args.update(
#     battery_capital_cost_per_mwh_capacity=363636.3636,
#     battery_n_cycles=4500,
#     battery_max_discharge=0.9,
#     battery_min_discharge_time=6,
#     battery_efficiency=0.75,
# )

# battery data for 7.2 MW sodium sulfide battery from Black & Veatch 2012, 
# "Cost and Performance Data for Power Generation Technologies"
# http://bv.com/docs/reports-studies/nrel-cost-report.pdf (for 2015 system)
# NOTE: this source includes forecasts for cost improvements in the future
# args.update(
#     battery_capital_cost_per_mwh_capacity=3890000/8.1
#     battery_n_cycles=5000,
#     battery_max_discharge=0.80,   # not specified; see notes under EPRI data below
#     battery_min_discharge_time=8.1,
#     battery_efficiency=0.75,
#     # they also include
#     # $59/MWh variable O&M
#     # $25200/MW-year fixed O&M
#     # and say "The O&M cost includes the cost of battery replacement every 5,000 hours."
#     # not sure whether this means 5,000 cycles or hours of use, but it seems to fit with cycles.
# )

# battery data for 50 MW/300 MWh sodium-sulfur system from EPRI, 2010,
# "Electric Energy Storage Technology Options: 
# A White Paper Primer on Applications, Costs, and Benefits",
# http://large.stanford.edu/courses/2012/ph240/doshay1/docs/EPRI.pdf
# This was used as the preferred battery prices in 2016-01-27 - 2016-04-30
# inflate_2010 = (1.0+args["inflation_rate"])**(args["base_financial_year"]-2010)
# args.update(
#     battery_capital_cost_per_mwh_capacity=inflate_2010*3200.0*1000/6,
#     battery_n_cycles=4500,
#     battery_max_discharge=0.85,
#     # DOD not specified;
#     # https://en.wikipedia.org/wiki/Sodium-sulfur_battery says 'Lifetime of 2,500 cycles at 100% depth of discharge (DOD), or 4,500 cycles at 80% DOD';
#     # brochure cited by wikipedia at http://web.archive.org/web/20060205153231/http://www.ulvac-uc.co.jp/prm/prm_arc/049pdf/ulvac049-02.pdf gives 4,500 cycles at 85% DOD.
#     # this article says "Lifetime of 2,500 cycles (at 100% DOD - depth of discharge), or 4,500 cycles (at 80% DOD)":
#     # http://www.rellpower.com/wp/wp-content/uploads/2015/07/Ioxus-The-Case-for-Sodium-Sulfur-Batters-and-Ultracapacitors.pdf
#     # Bito (NGK Insulators), 2005, "Overview of the Sodium-Sulfur Battery for the
#     # IEEE Stationary Battery Committee", http://dx.doi.org/10.1109/PES.2005.1489556 says
#     # "Long calendar and cycle life - 15 years and 2500 for 100% depth-of-discharge (DOD)
#     # or 4500 for 90% DOD or 6500 for 65% DOD, etc."
#     battery_min_discharge_time=6,
#     battery_efficiency=0.75,
# )

# Data for Tesla PowerWall from https://www.teslamotors.com/powerwall
# assuming 10 years * 365 cycles/year = 3650 cycle life
# args.update(
#     battery_capital_cost_per_mwh_capacity=(3000.0+1000.0)*1000/7.0,
#     # ($3000 (batteries) + $1000 (inverter))/7 kWh
#     # inverter cost estimate from
#     # http://www.catalyticengineering.com/top-ten-facts-about-teslas-350kwh-powerwall-battery/
#     # similar to power conversion costs in http://bv.com/docs/reports-studies/nrel-cost-report.pdf
#     # much more than power conversion costs in
#     # http://energyenvironment.pnnl.gov/pdf/National_Assessment_Storage_PHASE_II_vol_2_final.pdf
#     battery_n_cycles=10*365,
#     battery_max_discharge=1.0,
#     # DOD not specified by tesla, probably 100% (i.e. 10 MWh module capped at 7 MWh) according to
#     # http://www.catalyticengineering.com/top-ten-facts-about-teslas-350kwh-powerwall-battery/
#     battery_min_discharge_time=7.0/3.3, # 7.0 kWh / 3.3 kW
#     battery_efficiency=0.87,
#     # AC efficiency not given by tesla, estimated by
#     # http://www.catalyticengineering.com/top-ten-facts-about-teslas-350kwh-powerwall-battery/
# )


# other sources of battery cost data:
# http://energyenvironment.pnnl.gov/pdf/National_Assessment_Storage_PHASE_II_vol_2_final.pdf
# (also has depth-of-discharge vs. cycle life curves for NaS and Li-ion batteries)
# PSIP appendix J (but doesn't look at multi-hour storage or give storage depth): http://files.hawaii.gov/puc/3_Dkt%202011-0206%202014-08-26%20HECO%20PSIP%20Report.pdf
# http://www.hawaiianelectric.com/heco/Clean-Energy/Issues-and-Challenges/Energy-Storage
# http://www.nrel.gov/docs/fy10osti/46719.pdf
# http://www.irena.org/documentdownloads/publications/irena_battery_storage_report_2015.pdf
# https://www.purdue.edu/discoverypark/energy/assets/pdfs/SUFG/publications/SUFG%20Energy%20Storage%20Report.pdf (refs 5 and 22 from EPRI)
# https://www.wecc.biz/Reliability/2014_TEPPC_Generation_CapCost_Report_E3.pdf
# http://www.nrel.gov/docs/fy11osti/48595.pdf
# also see some NaS specs at http://www.eei.org/about/meetings/meeting_documents/abe.pdf
# and https://books.google.com/books?id=hGHr2L6HtKwC&pg=PA101&lpg=PA101&dq=sodium-sulfur+battery+calendar+life&source=bl&ots=ro3kExJx7o&sig=pciJ4sWF8wMmnYMljJPc2xlb4Bc&hl=en&sa=X&ved=0ahUKEwja6JyAmIDKAhUP0mMKHfyTDyIQ6AEINjAD#v=onepage&q=sodium-sulfur%20battery%20calendar%20life&f=false


# battery data from 2016-04-01 PSIP report (pp. J-82 - J-83)
# this was used for main model runs from 2016-05-01 onward
psip_nominal_battery_cost_per_kwh = [
    530, 493, 454, 421, 
    391, 371, 353, 339, 326, 316, 306, 298, 291, 285, 
    280, 275, 271, 268, 264, 262, 259, 257, 255, 253, 
    252, 250, 249, 248, 247, 246
]
psip_battery_years = range(2016, 2045+1)
psip_battery_cost_per_mwh = [
    1000.0 * nom_cost * 1.018**(args["base_financial_year"] - year)
        for year, nom_cost in zip(psip_battery_years, psip_nominal_battery_cost_per_kwh)
]
# TODO: limit battery life to 15 years even if not cycled that much
# TODO: how to count the cycle cost if new batteries will be available cheaper than this batch?
# just set aside based on current-year battery costs?
# TODO: retire and replace with cheaper model after 15 years
args.update(
    BATTERY_CAPITAL_COST_YEARS = psip_battery_years,
    battery_capital_cost_per_mwh_capacity_by_year = psip_battery_cost_per_mwh,
    battery_n_years=15,
    # battery_n_cycles=365*15,
    battery_max_discharge=1.0,
    battery_min_discharge_time=6,
    battery_efficiency=0.88,
)


# electrolyzer data from centralized current electrolyzer scenario version 3.1 in 
# http://www.hydrogen.energy.gov/h2a_prod_studies.html -> 01D_Current_Central_Hydrogen_Production_from_PEM_Electrolysis_version_3.1.xlsm
# (cited by 46719.pdf)
# note: we neglect land costs because they are small and can be recovered later
# TODO: move electrolyzer refurbishment costs from fixed to variable

# liquifier and tank data from http://www.nrel.gov/docs/fy99osti/25106.pdf

# fuel cell data from http://www.nrel.gov/docs/fy10osti/46719.pdf

inflate_1995 = (1.0+args["inflation_rate"])**(args["base_financial_year"]-1995)
inflate_2007 = (1.0+args["inflation_rate"])**(args["base_financial_year"]-2007)
inflate_2008 = (1.0+args["inflation_rate"])**(args["base_financial_year"]-2008)
h2_lhv_mj_per_kg = 120.21   # from http://hydrogen.pnl.gov/tools/lower-and-higher-heating-values-fuels
h2_mwh_per_kg = h2_lhv_mj_per_kg / 3600     # (3600 MJ/MWh)
electrolyzer_kg_per_mwh = 1000.0/54.3    # (1000 kWh/1 MWh)(1kg/54.3 kWh)   # TMP_Usage cell
electrolyzer_mw = 50000.0 * (1.0/electrolyzer_kg_per_mwh) * (1.0/24.0)   # (kg/day) * (MWh/kg) * (day/h)    # design_cap cell

args.update(
    hydrogen_electrolyzer_capital_cost_per_mw=inflate_2007*144641663.0/electrolyzer_mw,        # depr_cap cell
    hydrogen_electrolyzer_fixed_cost_per_mw_year=inflate_2007*7134560.0/electrolyzer_mw,         # fixed cell
    hydrogen_electrolyzer_variable_cost_per_kg=0.0,       # they only count electricity as variable cost
    hydrogen_electrolyzer_kg_per_mwh=electrolyzer_kg_per_mwh,
    hydrogen_electrolyzer_life_years=40,                      # plant_life cell

    hydrogen_liquifier_capital_cost_per_kg_per_hour=inflate_1995*25600,       # 25106.pdf p. 18, for 1500 kg/h plant, approx. 100 MW
    hydrogen_liquifier_fixed_cost_per_kg_hour_year=0.0,   # unknown, assumed low
    hydrogen_liquifier_variable_cost_per_kg=0.0,      # 25106.pdf p. 23 counts tank, equipment and electricity, but those are covered elsewhere
    hydrogen_liquifier_mwh_per_kg=10.0/1000.0,        # middle of 8-12 range from 25106.pdf p. 23
    hydrogen_liquifier_life_years=30,             # unknown, assumed long

    liquid_hydrogen_tank_capital_cost_per_kg=inflate_1995*18,         # 25106.pdf p. 20, for 300000 kg vessel
    liquid_hydrogen_tank_life_years=40,                       # unknown, assumed long

    hydrogen_fuel_cell_capital_cost_per_mw=813000*inflate_2008,   # 46719.pdf
    hydrogen_fuel_cell_fixed_cost_per_mw_year=27000*inflate_2008,   # 46719.pdf
    hydrogen_fuel_cell_variable_cost_per_mwh=0.0, # not listed in 46719.pdf; we should estimate a wear-and-tear factor
    hydrogen_fuel_cell_mwh_per_kg=0.53*h2_mwh_per_kg,   # efficiency from 46719.pdf
    hydrogen_fuel_cell_life_years=15,   # 46719.pdf
)

args.update(
    pumped_hydro_headers=[
        'ph_project_id', 'ph_load_zone', 'ph_capital_cost_per_mw', 'ph_project_life', 'ph_fixed_om_percent',
        'ph_efficiency', 'ph_inflow_mw', 'ph_max_capacity_mw'],
    pumped_hydro_projects=[
        ['Lake_Wilson', 'Oahu', 2800*1000+35e6/150, 50, 0.015, 0.77, 10, 150],
    ]
)

args.update(
    rps_targets = {2015: 0.15, 2020: 0.30, 2030: 0.40, 2040: 0.70, 2045: 1.00}
)

# data definitions for alternative scenarios
alt_args = [
    dict(inputs_dir="inputs", time_sample="rps_mini"),
    dict(inputs_dir="inputs_tiny", time_sample="tiny"),
    dict(
        inputs_dir="inputs_tiny_tiny", time_sample="tiny",
        exclude_technologies=args["exclude_technologies"]+('CentralTrackingPV', 'Wind')
    ),
]


# annual change in capital cost of new renewable projects:
# solar cost projections: decline by 6.4%/year (based on residential PV systems from 1998 to 2014 in Fig. 7 of "Tracking the Sun VIII: The Installed Price of Residential and Non-Residential Photovoltaic Systems in the United States," https://emp.lbl.gov/reports (declines have been faster over more recent time period, and faster for non-residential systems).
# wind cost projections:
# increase by 1.1%/year (based on 1998-2014 in  2014 Wind Technologies Market Report, https://emp.lbl.gov/reports)



for a in alt_args:
    # clone the arguments dictionary and update it with settings from the alt_args entry, if any
    active_args = dict(args.items() + a.items())
    scenario_data.write_tables(**active_args)

    
if cmd_line_args.write_rhos:
    # calculate and store the per-variable rho values
    print "creating rho settings files (takes several minutes)..."
    inputs_dirs = set()
    if "inputs_dir" in args:
        inputs_dirs.add(args["inputs_dir"])
    for a in alt_args:
        inputs_dirs.add(a["inputs_dir"])
    # get standard model options (but ignore command-line arguments)
    base_args = switch_mod.solve.get_option_file_args()
    for inputs_dir in inputs_dirs:
        # build a model using the current inputs_dir
        instance = switch_mod.solve.main(
            args=base_args+['--inputs-dir', inputs_dir], 
            return_instance=True)
        instance.save_pha_rho_file()
else:
    print "SKIPPING rho settings files."
    print "Use --write-rhos flag to write the rho settings files."

