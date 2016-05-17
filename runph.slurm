#!/bin/sh

## reporting
#SBATCH --error=%A.err
#SBATCH --output=%A.out
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=8083489586@vtext.com

#SBATCH --mem-per-cpu=6400

##SBATCH --job-name=ph100ann
##SBATCH --partition=community.q
##SBATCH --time=3-00:00:00
##SBATCH --nodes=6
##SBATCH --tasks-per-node=20

##SBATCH --job-name=ph15og
##SBATCH --partition=kill.q
##SBATCH --time=1-12:00:00
##SBATCH --nodes=1
##SBATCH --tasks-per-node=20
## note: both of the settings below are out of a total of 20 tasks per node
#export coordinator_reserved_slots=5  # number of slots to reserve for the coordinator processes on the first node
#export workers_per_node=20  # number of slots to use for workers on each node (could be reduced to conserve RAM)
#export PHBOUNDINTERVAL=5
#export runph_cmd='runph --phpyro-required-workers=15 --instance-directory=inputs/pha_15  --rho-cfgfile=rho_cfgfile_100 --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=pyomo.pysp.plugins.phboundextension --user-defined-extension=optimality_gap_convergence --user-defined-extension=callbacks --verbose --enable-ww-extensions --ww-extension-cfgfile=wwph_slam.cfg --ww-extension-suffixfile=inputs/wwph_slam.suffixes'
# export runph_cmd='runph --phpyro-required-workers=15 --instance-directory=inputs/pha_15 --ww-extension-suffixfile=inputs/wwph.suffixes --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=pyomo.pysp.plugins.phboundextension --user-defined-extension=optimality_gap_convergence --user-defined-extension=callbacks --verbose --enable-ww-extensions --ww-extension-cfgfile=wwph.cfg'

##SBATCH --job-name=ph117slam
##SBATCH --partition=kill.q
##SBATCH --time=0-02:00:00
##SBATCH --nodes=7
##SBATCH --tasks-per-node=20
#export coordinator_reserved_slots=20  # number of slots to reserve for the coordinator processes on the first node
#export workers_per_node=20  # number of slots to use for workers on each node (could be reduced to conserve RAM)
#export runph_cmd='runph --phpyro-required-workers=117 --instance-directory=inputs/pha_117 --rho-cfgfile=callbacks --user-defined-extension=callbacks --verbose --enable-ww-extensions --ww-extension-cfgfile=wwph_slam.cfg --ww-extension-suffixfile=inputs/wwph_slam.suffixes --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1 --bounds-cfgfile=pha_bounds_cfg.py'

# # large memory: fails with errors
# #SBATCH --mem-per-cpu=25000
# #SBATCH --job-name=ph125evl
# #SBATCH --partition=lm.q
# #SBATCH --time=1-00:00:00
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=40
# export coordinator_reserved_slots=1  # number of slots to reserve for the coordinator processes on the first node
# export workers_per_node=40  # number of slots to use for workers on each node (could be reduced to conserve RAM)
# export runph_cmd='runph --phpyro-required-workers=39 --instance-directory=inputs/pha_125_logistic --disable-xhat-computation --rho-cfgfile=callbacks --user-defined-extension=callbacks --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=0 --bounds-cfgfile=pha_bounds_cfg.py'
# # # pin the build variables to plan from a particular run or runs (used by pha_bounds_cfg.py)
# # # best solution to 125 separate prices
# # export PHA_FIX_BUILD_VARS_FILES=outputs/build_540397_iter0_Scenario_*.tsv
# # export USE_PSIP_PLAN=0
# # find best plan
# # echo "================ finding one build plan for each price scenario (one lm node) ================"
# export PHA_WRITE_SCENARIO_SUMMARY="1"
# export PHA_WRITE_ITER0_SCENARIO_BUILDS="1"
# export USE_PSIP_PLAN=0
# # 540997=best solutions, all scenarios

# evaluate quality of previous solutions
# done 5/13: run 125 times with best single solution for each scenario (540419< 540420< 540423<)(lots of errors but still running?)
# done 5/13: run once for HECO PSIP Preferred Plan (540421<done 540422pd) (540400<shutdown without saving)
# done 5/13: run once for optimal plan with mean prices (540408<err-infeasible and 540409<infeasible)
# bad: run 125 times with best single solution for each scenario (540881, 540882, 540883)
# bad: run once for HECO PSIP Preferred Plan (540879)
# partial (tripped by old queue): run 125 times with best single solution for each scenario (542668, 542669, 542670)
# eval 125 times: (542692, 542693, 542694, 542695)
# done: run once for HECO PSIP Preferred Plan (542667)
#SBATCH --job-name=ph125evl
#SBATCH --partition=community.q
#SBATCH --time=2-00:00:00
#SBATCH --nodes=7
#SBATCH --tasks-per-node=20
# #SBATCH --dependency=afterok:540877
export coordinator_reserved_slots=15  # number of slots to reserve for the coordinator processes on the first node
export workers_per_node=20  # number of slots to use for workers on each node (could be reduced to conserve RAM)
export runph_cmd='runph --phpyro-required-workers=125 --instance-directory=inputs/pha_125_logistic --disable-xhat-computation --rho-cfgfile=callbacks --user-defined-extension=callbacks --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=0 --bounds-cfgfile=pha_bounds_cfg.py'
export PHA_WRITE_SCENARIO_SUMMARY=1
# # pin the build variables to plan from a particular run or runs (used by pha_bounds_cfg.py)
# # best solution to 125 separate prices
export PHA_FIX_BUILD_VARS_FILES=outputs/build_540877_iter0_Scenario_*.tsv
export USE_PSIP_PLAN=0
# # HECO PSIP
# export USE_PSIP_PLAN=1
# export PHA_FIX_BUILD_VARS_FILE=outputs/build_540876_iter0_Scenario_0000.tsv
# best solution for average prices
# export PHA_FIX_BUILD_VARS_FILE=outputs/build_540398_iter0_Scenario_0000.tsv
# export USE_PSIP_PLAN=0

# # one solution for each scenario
# #SBATCH --job-name=ph10gen
# #SBATCH --partition=community.q
# #SBATCH --time=0-12:00:00
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=20
# export coordinator_reserved_slots=10  # number of slots to reserve for the coordinator processes on the first node
# export workers_per_node=20  # number of slots to use for workers on each node (could be reduced to conserve RAM)
# # runph fails when we use the ww extensions and suffixes, because ConvertToLNG is not being added to the RootNode
# # for some reason. This is hard to debug, but we don't need the ww extensions for the portfolio-generation
# # approach anyway, so we disable them.
# # runph also fails due to not finding bounds for BuildUnits when we use the linearize-nonbinary-penalty-terms
# # breakpoint-strategy and bounds-cfgfile options, even though they should work too, so we also turn those off.
# export runph_cmd='runph --phpyro-required-workers=10 --instance-directory=inputs/pha_125_logistic --user-defined-extension=callbacks --verbose --default-rho=100000000 --termdiff-threshold=0.001 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=0'
# echo "================ finding one build plan for each price scenario (one node) ================"
# export PHA_WRITE_SCENARIO_SUMMARY="1"
# export PHA_WRITE_ITER0_SCENARIO_BUILDS="1"
# export USE_PSIP_PLAN=0
# # 540877=community.q
# # 540897=1-node kill.q < tries to use too much memory, workers get killed off
# # 540963=1-node community.q < tries to use too much memory, workers get killed off

# # just one solution, using the mean pricing
# #SBATCH --job-name=ph125mean
# #SBATCH --partition=kill.q
# #SBATCH --time=0-02:00:00
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=2
# export coordinator_reserved_slots=1  # number of slots to reserve for the coordinator processes on the first node
# export workers_per_node=2  # number of slots to use for workers on each node (could be reduced to conserve RAM)
# # runph fails when we use the ww extensions and suffixes, because ConvertToLNG is not being added to the RootNode
# # for some reason. This is hard to debug, but we don't need the ww extensions for the portfolio-generation
# # approach anyway, so we disable them.
# # runph also fails due to not finding bounds for BuildUnits when we use the linearize-nonbinary-penalty-terms
# # breakpoint-strategy and bounds-cfgfile options, even though they should work too, so we also turn those off.
# # export runph_cmd='runph --phpyro-required-workers=1 --instance-directory=inputs/pha_125_logistic_mean --user-defined-extension=callbacks --verbose --default-rho=100000000 --termdiff-threshold=0.001 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=0'
# export runph_cmd='runph --phpyro-required-workers=1 --instance-directory=inputs/pha_125_logistic_mean --disable-xhat-computation --rho-cfgfile=callbacks --user-defined-extension=callbacks --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=0 --bounds-cfgfile=pha_bounds_cfg.py'
# echo "========= Solving for mean prices ============"
# export USE_PSIP_PLAN=0
# export PHA_WRITE_SCENARIO_SUMMARY=1
# export PHA_WRITE_ITER0_SCENARIO_BUILDS=1

# # one solution, for the PSIP preferred plan
# #SBATCH --job-name=psip
# #SBATCH --partition=community.q
# #SBATCH --time=0-01:00:00
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=2
# export coordinator_reserved_slots=1  # number of slots to reserve for the coordinator processes on the first node
# export workers_per_node=2  # number of slots to use for workers on each node (could be reduced to conserve RAM)
# export runph_cmd='runph --phpyro-required-workers=1 --instance-directory=inputs/pha_125_logistic_mean --user-defined-extension=callbacks --verbose --default-rho=100000000 --termdiff-threshold=0.001 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=0'
# echo ========= running PSIP scenario with mean prices ============
# export USE_PSIP_PLAN=1
# export PHA_WRITE_SCENARIO_SUMMARY=1
# export PHA_WRITE_ITER0_SCENARIO_BUILDS=1
# # 540875=kill.q, 540876=community.q

##SBATCH --job-name=phtiny
##SBATCH --partition=sb.q
##SBATCH --time=0-01:00:00
##SBATCH --nodes=1
##SBATCH --tasks-per-node=5
#export coordinator_reserved_slots=1  # number of slots to reserve for the coordinator processes on the first node
#export workers_per_node=5  # number of slots to use for workers on each node (could be reduced to conserve RAM)
#export runph_cmd='runph --phpyro-required-workers=4 --instance-directory=inputs_tiny/pha --rho-cfgfile=callbacks --user-defined-extension=callbacks --verbose --enable-ww-extensions --ww-extension-cfgfile=wwph_slam.cfg --ww-extension-suffixfile=inputs_tiny/wwph_slam.suffixes --default-rho=100000000 --termdiff-threshold=0.01 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py'

# report various scenario-by-scenario data
#export PHA_WRITE_SCENARIO_SUMMARY="1"
#export PHA_WRITE_ITER0_SCENARIO_BUILDS="1"
#export PHA_WRITE_XHAT_BUILD="1"

# this version of the phsolverserver cmd enables the various PHA_WRITE_... options
export phsolverserver_cmd='phsolverserver --traceback --pyro-port=$nsp --user-defined-extension=callbacks'

## NOTE: any #SBATCH directives that come after executable code will be ignored
## so all export statements should be commented out along with the corresponding SBATCH statements


## max memory per cpu: 6400 for standard nodes, 26214 for large memory nodes 

# NOTE: specifying --ntasks and --mem-per-cpu instead of --nodes and --tasks-per-node
# might allow tasks to fit in on partial nodes, but there aren't many of these, and our
# coordinator thread uses a lot of memory, so we need to allocate as much memory as possible
# on that node. (Maybe this should go in the exclusive.q)

## 3 day max run time for community.q, kill.q, exclusive.q, and htc.q. 1 Hour max run time for sb.q 
## task-per-node x cpus-per-task should not typically exceed core count on an individual node 

## for output files: %A - filled with jobid. %a - filled with job arrayid

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html 

# note: you should submit this by executing "sbatch runph.slurm"

# test version
# export runph_cmd='runph --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --instance-directory=inputs_tiny/pha_19 --default-rho=1000000000 --traceback --max-iterations=1000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=reporting_phextension --verbose'

# production version

# export runph_cmd='runph --phpyro-required-workers=100 --instance-directory=inputs/pha_100_annual --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=callbacks --verbose'

# export runph_cmd='runph --phpyro-required-workers=15 --instance-directory=inputs/pha_15 --rho-cfgfile=rho_cfgfile_100x --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=callbacks --verbose'
#export runph_cmd='runph --phpyro-required-workers=117 --instance-directory=inputs/pha_117 --rho-cfgfile=rho_cfgfile_050 --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver-manager=phpyro --shutdown-pyro --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=callbacks --verbose'
# single-core test:
# runph --instance-directory=inputs_tiny/pha --rho-cfgfile=rho_cfgfile_100 --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=callbacks --verbose > progress_cp_1.00.txt
# runph --instance-directory=inputs_tiny/pha --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=callbacks --verbose > progress_fx_1e8.txt
# with ww sep rho-setters:
# runph --instance-directory=inputs_tiny/pha --ww-extension-suffixfile=inputs_tiny/wwph.suffixes --default-rho=100000000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --traceback --max-iterations=1000 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=callbacks --verbose --enable-ww-extensions --ww-extension-cfgfile=wwph.cfg > progress_sep_1.00.txt

#export runph_cmd='runph --solver-manager=phpyro --shutdown-pyro --phpyro-required-workers=117 --pyro-port=$nsp --solver=cplexamp --scenario-solver-options="threads=1" --model-directory=. --instance-directory=inputs/pha_117 --default-rho=1000000000 --traceback --max-iterations=1000 --termdiff-threshold=0.001 --linearize-nonbinary-penalty-terms=7 --breakpoint-strategy=1 --bounds-cfgfile=pha_bounds_cfg.py --user-defined-extension=reporting_phextension --verbose'


# note: it appears that environment variables exported here will be inherited
# by all the tasks. We use this to pass the name server port and the main
# runph command to the worker script. This way all tasks know about the port
# and the worker script can be pretty generic. (All changes to define new runs
# occur in this .slurm file.)


launch_job ()
{ 
    export nsp=$(/home/mfripp/apps/next_port.py)
    echo "Starting runph.slurm; will use port $nsp for pyro name server."

    # show the runph command that will be run for this job
    echo =============================== runph command ====================================
    echo $runph_cmd
    env | grep PHA_
    echo ================================================================================== 
    # run the job
    # note: we use stdbuf to switch to line buffering instead of default (4k?) buffering
    # it's possible that the srun --unbuffered flag would help with this.
    srun stdbuf -oL -eL ./runph_slurm

}

if [ "$PHA_FIX_BUILD_VARS_FILES" ]; then
    for f in $PHA_FIX_BUILD_VARS_FILES; do
        export PHA_FIX_BUILD_VARS_FILE=$f
        # create a blank "in progress" file; this guarantees grep will run OK and also 
        # throws any canceled job back into the queue.
        echo "" > $SLURM_JOBID.running
        if grep $PHA_FIX_BUILD_VARS_FILE *.run* > /dev/null ; then 
            echo ================================================================================== 
            echo "Skipping $PHA_FIX_BUILD_VARS_FILE; already processed by another worker."
            echo ================================================================================== 
        else
            echo $PHA_FIX_BUILD_VARS_FILE > $SLURM_JOBID.running
            launch_job
            # it would be nice to do remove the .running file even if the job is canceled,
            # but only add the job to the .run file if it succeeds.
            echo adding $PHA_FIX_BUILD_VARS_FILE to $SLURM_JOBID.run
            echo $PHA_FIX_BUILD_VARS_FILE >> $SLURM_JOBID.run
            rm $SLURM_JOBID.running
        fi
    done
else
    launch_job
fi
