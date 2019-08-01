import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque
import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, Dataset, fmt_row
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.gail.statistics import stats


def traj_segment_generator(pi, env, horizon, stochastic):

    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])

    while True:
        ac = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "true_rew": true_rews, "new": news,
                   "ac": acs,  "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}
            _,  = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_true_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac

        ob, true_rew, new, _ = env.step(ac)
        true_rews[i] = true_rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def learn(env, policy_func, med_func, expert_dataset, pretrained, pretrained_weight, g_step, m_step, e_step, inner_iters, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name, max_timesteps=0, max_episodes=0, max_iters=0,
          batch_size=128, med_stepsize=3e-4, pi_stepsize=3e-4, callback=None, writer=None):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=(pretrained_weight != None))
    med = med_func("mediator", ob_space, ac_space, reuse=False)
    pi_var_list = pi.get_trainable_variables()
    med_var_list = med.get_trainable_variables()
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
    ac = U.get_placeholder(name='ac', dtype=tf.float32, shape=[None] + list(ac_space.shape))
    med_loss = -tf.reduce_mean(med.pd.logp(ac))
    #pi_loss = -0.5 * (tf.reduce_mean(pi.pd.logp(ac) - med.pd.logp(ac)))
    pi_loss = tf.reduce_mean(pi.pd.kl(med.pd)) #- tf.reduce_mean(pi.pd.entropy())
    expert_loss = -tf.reduce_mean(pi.pd.logp(ac))
    compute_med_loss = U.function([ob, ac], med_loss)
    compute_pi_loss = U.function([ob, ac], pi_loss)
    compute_exp_loss = U.function([ob, ac], expert_loss)
    compute_med_lossandgrad = U.function([ob, ac], [med_loss, U.flatgrad(med_loss, med_var_list)])
    compute_pi_lossandgrad = U.function([ob],  [pi_loss, U.flatgrad(pi_loss, pi_var_list)])
    compute_exp_lossandgrad = U.function([ob, ac], [expert_loss, U.flatgrad(expert_loss, pi_var_list)])
    get_pi_flat = U.GetFlat(pi_var_list)
    get_med_flat = U.GetFlat(med_var_list)
    set_pi_from_flat = U.SetFromFlat(pi_var_list)
    set_med_from_flat = U.SetFromFlat(med_var_list)
    med_adam = MpiAdam(med_var_list)
    pi_adam = MpiAdam(pi_var_list)

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    # th_pi_init = get_pi_flat()
    # th_med_init = get_med_flat()
    # MPI.COMM_WORLD.Bcast([th_pi_init, th_med_init], root=0)
    # set_pi_from_flat(th_pi_init)
    # set_med_from_flat(th_med_init)
    med_adam.sync()
    pi_adam.sync()
    # if rank == 0:
    #     print("Init pi param sum %d, init med param sum %d." % (th_pi_init.sum(), th_med_init.sum()), flush=True)

    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1
    loss_stats = stats(["expert_loss", "med_loss", "pi_loss"])
    ep_stats = stats(["True_rewards", "Episode_length"])

    if pretrained_weight is not None:
        U.load_state(pretrained_weight, var_list=pi_var_list)

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname)

        logger.log("********** Iteration %i ************" % iters_so_far)

        # ======= Optimize Mediator=========

        seg = seg_gen.__next__()
        g_ob, g_ac = seg['ob'], seg['ac']
        d = Dataset(dict(ob=g_ob, ac=g_ac))
        optim_batchsize = min(batch_size, len(g_ob))
        med_losses = []
        pi_losses = []
        exp_losses = []
        for _ in range(inner_iters):
            logger.log("Expert Pretrain...")
            for _ in range(e_step):
                ob, ac = expert_dataset.get_next_batch(batch_size)
                exp_loss, g = compute_exp_lossandgrad(ob, ac)
                pi_adam.update(allmean(g), pi_stepsize)
                exp_losses.append(exp_loss)
            #logger.record_tabular("expert_loss_each_iter", np.mean(np.array(exp_losses)))
            logger.log("Optimizing Mediator...")
            for _ in range(m_step):
                g_batch = d.next_batch(optim_batchsize // 2)
                e_ob_batch, e_ac_batch = expert_dataset.get_next_batch(optim_batchsize // 2)
                mix_ob_batch = np.concatenate((g_batch['ob'], e_ob_batch), 0)
                mix_ac_batch = np.concatenate((g_batch['ac'], e_ac_batch), 0)
                idx = np.arange(len(mix_ob_batch))
                np.random.shuffle(idx)
                mix_ob_batch = mix_ob_batch[idx, :]
                mix_ac_batch = mix_ac_batch[idx, :]
                if hasattr(med, "obs_rms"):
                    med.obs_rms.update(mix_ob_batch)
                newlosses, g = compute_med_lossandgrad(mix_ob_batch, mix_ac_batch)
                med_adam.update(allmean(g), med_stepsize)
                med_losses.append(newlosses)

            #logger.record_tabular("med_loss_each_iter", np.mean(np.array(med_losses)))

            logger.log("Optimizing Generator...")
            for _ in range(g_step):
                g_batch = d.next_batch(optim_batchsize)
                g_ob, g_ac = g_batch['ob'], g_batch['ac']
                if hasattr(pi, "obs_rms"):
                    pi.obs_rms.update(g_ob)
                pi_loss, g = compute_pi_lossandgrad(g_ob)
                pi_adam.update(allmean(g), pi_stepsize)
                pi_losses.append(pi_loss)
            #logger.record_tabular("gen_loss_each_iter", np.mean(np.array(pi_losses)))
        logger.record_tabular("expert_loss_each_iter", np.mean(np.array(exp_losses)))
        logger.record_tabular("med_loss_each_iter", np.mean(np.array(med_losses)))
        logger.record_tabular("gen_loss_each_iter", np.mean(np.array(pi_losses)))

        lrlocal = (seg["ep_lens"], seg["ep_true_rets"])
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
        lens, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if writer is not None:
            loss_stats.add_all_summary(writer, [np.mean(np.array(exp_losses)), np.mean(np.array(med_losses)), np.mean(np.array(pi_losses))], episodes_so_far)
            ep_stats.add_all_summary(writer, [np.mean(true_rewbuffer), np.mean(lenbuffer)], episodes_so_far)
        if rank == 0:
            logger.dump_tabular()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]