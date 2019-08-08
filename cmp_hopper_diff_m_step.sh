#!/bin/bash
cd ~/mediator_IL/baselines/mediator/ 
for seed in $(seq 0 2); do 
	OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperMstepCmp/medm1-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=5e-5 --m_step=1 --seed=$seed &
done

for seed in $(seq 0 2); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperMstepCmp/medm5-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=3e-4 --m_step=5 --seed=$seed &
done

for seed in $(seq 0 2); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperMstepCmp/medm20-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=1e-3 --m_step=20 --seed=$seed &
done

for seed in $(seq 0 2); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperMstepCmp/medm50-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=3e-3 --m_step=50 --seed=$seed &
done

for seed in $(seq 0 2); do
        OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/HopperMstepCmp/medm50_lowlr-$seed python run_mujoco.py --expert_step=0 --inner_iter=1000 --pi_stepsize=1e-3 --m_step=50 --seed=$seed &
done



