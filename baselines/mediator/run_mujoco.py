import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

from baselines.mediator import generator, mediator, learner
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.mediator.dataset.mujoco_dset import Mujoco_Dset


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow implement of Mediator Imitation Learning")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='../../data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=1)
    parser.add_argument('--m_step', help='number of steps to train mediator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--mediator_hidden_size', type=int, default=100)
    # Algorithms Configuration
    # parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    # parser.add_argument('--max_kl', type=float, default=0.01)
    # parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    # parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    parser.add_argument('--pretrained', type=bool, default=False, help='Use BC to pretrain')
    # boolean_flag(parser, '--pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(args):
    task_name = 'med_step_'+str(args.m_step)+'_mediator_IL'
    if args.pretrained:
        task_name += '_with_pretrained'
    if args.traj_limitation != np.inf:
        task_name += 'transition_limitation_%d' % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    task_name += "_seed__" + str(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return generator.Generator(name=name, ob_space=ob_space, ac_space=ac_space, reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)

    def med_fn(name, ob_space, ac_space, reuse=False):
        return mediator.Mediator(name=name, ob_space=ob_space, ac_space=ac_space, reuse=reuse, hid_size=args.mediator_hidden_size, num_hid_layers=2)

    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    if args.task == 'train':
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        train(env,
              args.seed,
              policy_fn,
              med_fn,
              dataset,
              args.g_step,
              args.m_step,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name)
    # elif args.task == 'evaluate':
    #     runner(env,
    #            policy_fn,
    #            args.load_model_path,
    #            timesteps_per_batch=1024,
    #            number_trajs=10,
    #            stochastic_policy=args.stochastic_policy,
    #            save=args.save_sample
    #            )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, med_fn, dataset, g_step, m_step, num_timesteps, save_per_iter, checkpoint_dir, log_dir,
          pretrained, BC_max_iter, task_name=None):
    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset, max_iters=BC_max_iter)
    rank = MPI.COMMON_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMMON_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env.seed(workerseed)
    learner.learn(env, policy_fn, med_fn, dataset, pretrained, pretrained_weight, g_step, m_step, save_per_iter,
          checkpoint_dir, log_dir, timesteps_per_batch=1024, task_name=task_name)


if __name__ == '__main__':
    args = argsparser()
    main(args)