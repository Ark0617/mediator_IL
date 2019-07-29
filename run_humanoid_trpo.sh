#!/bin/bash

OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/Humanoid/trpo_mpi/0 python -m baselines.run --alg=trpo_mpi --env=Humanoid-v2 --num_timesteps=2e7 --seed=0 &

OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/Humanoid/trpo_mpi/1 python -m baselines.run --alg=trpo_mpi --env=Humanoid-v2 --num_timesteps=2e7 --seed=1 &

OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/Humanoid/trpo_mpi/2 python -m baselines.run --alg=trpo_mpi --env=Humanoid-v2 --num_timesteps=2e7 --seed=2 &

OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/Humanoid/trpo_mpi/3 python -m baselines.run --alg=trpo_mpi --env=Humanoid-v2 --num_timesteps=2e7 --seed=3 &

OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/Humanoid/trpo_mpi/4 python -m baselines.run --alg=trpo_mpi --env=Humanoid-v2 --num_timesteps=2e7 --seed=4 &

OPENAI_LOG_FORMAT=csv OPENAI_LOGDIR=$HOME/logs/Humanoid/trpo_mpi/5 python -m baselines.run --alg=trpo_mpi --env=Humanoid-v2 --num_timesteps=2e7 --seed=5 &

