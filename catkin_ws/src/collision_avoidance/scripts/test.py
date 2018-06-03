import gym
import gym_collision_avoidance
import tensorflow as tf
import tensorflow.contrib.slim as slim
from network_models import DRQN
import numpy as np

MODEL_PATH = "models/DRQN"

env = gym.make('CollisionGazebo-v0')
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
trace_length = 8
learning_rate = 0

tf.reset_default_graph()
lstm = tf.contrib.rnn.BasicLSTMCell(num_units=trace_length, state_is_tuple=True)
main_qn = DRQN(name="main", state_size=state_size, action_size=action_size, hidden_size=trace_length, learning_rate=learning_rate, rnn_cell=lstm)

with tf.Session() as sess: 
    saver = tf.train.Saver()
    print('Loading latest saved model...')
    ckpt = tf.train.latest_checkpoint(MODEL_PATH)
    saver.restore(sess, ckpt)

    env.render() 
    env.reset()
    state, reward, done, _ = env.step(env.action_space.sample())
    reccurent_state_in = (np.zeros([1, trace_length]), np.zeros([1, trace_length]))
    
    while True:
        Qs = main_qn.get_Q_values(sess, [state/255.0], 1, reccurent_state_in, 1, 1.0)
        next_reccurent_state_in = main_qn.rnn_hidden_state(sess, [state/255.0], 1, reccurent_state_in, 1)
        action = np.argmax(Qs)

        state, _, done, _ = env.step(action)
        reccurent_state_in = next_reccurent_state_in

        if done:
            env.reset()
            state, _, done, _ = env.step(env.action_space.sample())
            reccurent_state_in = (np.zeros([1, trace_length]), np.zeros([1, trace_length]))