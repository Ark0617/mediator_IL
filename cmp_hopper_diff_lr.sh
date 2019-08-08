#!/bin/bash
cd ~/mediator_IL/baselines/mediator/ 
for seed in $(seq 0 2); do 
	OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperLrCmp/med5e_6-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=5e-6 --m_step=1 --seed=$seed &
done
for seed in $(seq 0 2); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperLrCmp/med5e_5-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=5e-5 --m_step=1 --seed=$seed &
done
for seed in $(seq 0 2); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperLrCmp/med8e_5-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=8e-5 --m_step=1 --seed=$seed &
done
for seed in $(seq 0 2); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperLrCmp/med5e_4-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=5e-4 --m_step=1 --seed=$seed &
done

cd ~/mediator_IL/baselines/gail/
for seed in $(seq 0 2); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperLrCmp/gail-$seed python run_mujoco.py --g_step=1 --seed=$seed &
done
