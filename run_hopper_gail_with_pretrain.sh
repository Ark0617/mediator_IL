#!/bin/bash
cd ~/baselines/baselines/gail/ 
for seed in $(seq 0 5); do 
	OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/Hopper/GAIL_pretrained-$seed python run_mujoco.py --seed=$seed --pretrained=True  & 
done

