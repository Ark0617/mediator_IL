#!/bin/bash
cd ~/mediator_IL/baselines/mediator/ 
for seed in $(seq 0 5); do 
	OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/NewHopperCmp/med-$seed python run_mujoco.py --expert_step=0 --inner_iter=500 --m_step=1 --seed=$seed &
done
cd ~/mediator_IL/baselines/gail/
for seed in $(seq 0 5); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/NewHopperCmp/gail-$seed python run_mujoco.py --g_step=1 --seed=$seed &
done

