import tensorflow as tf
import gym
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
from baselines.acktr.utils import dense


class Mediator(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # if reuse:
            #     tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        g_ob = U.get_placeholder(name='g_ob', dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        e_ob = U.get_placeholder(name='e_ob', dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        with tf.variable_scope("obfilter", reuse=tf.AUTO_REUSE):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        g_obz = tf.clip_by_value((g_ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        e_obz = tf.clip_by_value((e_ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        g_last_out = g_obz
        e_last_out = e_obz
        #with tf.variable_scope("med_model", reuse=tf.AUTO_REUSE):

        def modeling(input_ob):
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(dense(input_ob, hid_size, "fc%i" % (i+1), weight_init=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = dense(last_out, pdtype.param_shape()[0] // 2, "final", U.normc_initializer(0.01))
                logstd = tf.get_variable(name='med_logstd', shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = dense(input_ob, pdtype.param_shape()[0], "final", U.normc_initializer(0.01))
            return pdparam

        self.g_pd = pdtype.pdfromflat(modeling(g_last_out))
        self.e_pd = pdtype.pdfromflat(modeling(e_last_out))
        # self.state_in = []
        # self.state_out = []
        # stochastic = U.get_placeholder(name='stochastic', dtype=tf.bool, shape=())
        # ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        # self.ac = ac
        # self._act = U.function([stochastic, ob], [ac])

    # def act(self, stochastic, ob):
    #     ac1 = self._act(stochastic, ob[None])
    #     return ac1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []