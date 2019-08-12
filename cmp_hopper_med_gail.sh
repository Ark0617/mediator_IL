#!/bin/bash
cd ~/mediator_IL/baselines/mediator/ 
for seed in $(seq 0 3); do 
	OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/NewHopperCmp/med-$seed python run_mujoco.py  --m_step=10 --pi_stepsize=1e-2 --med_stepsize=1e-3 --seed=$seed &
	
done
cd ~/mediator_IL/baselines/gail/
for seed in $(seq 0 3); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/NewHopperCmp/gail-$seed python run_mujoco.py --g_step=1 --seed=$seed &
done

