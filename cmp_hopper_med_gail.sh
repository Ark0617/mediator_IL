#!/bin/bash
cd ~/mediator_IL/baselines/mediator/ 
for seed in $(seq 0 5); do 
	OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/NewHopperCmp/med1000-$seed python run_mujoco.py --expert_step=0 --inner_iter=1200 --m_step=1 --seed=$seed &
done
cd ~/mediator_IL/baselines/gail/
for seed in $(seq 0 5); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperCmp/gail-$seed python run_mujoco.py --seed=$seed --g_step=1 &
done

