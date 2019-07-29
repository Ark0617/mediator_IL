#!/bin/bash
alg_list="ppo2 trpo_mpi"
for algo in $alg_list; do 
	for seed in $(seq 0 5); do 
		OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/Humanoid/$algo-$seed python -m baselines.run --alg=$algo --env=Humanoid-v2 --num_timesteps=2e7 --seed=$seed & 
	done; 
done








