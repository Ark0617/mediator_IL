#!/bin/bash
cd ~/mediator_IL/baselines/mediator/ 
for seed in $(seq 0 5); do 
	OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperCmp/Pure_BC-$seed python run_mujoco.py --m_step=0 --g_step=0 --seed=$seed &
done


