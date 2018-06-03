#!/usr/bin/env python
import gym
import gym_collision_avoidance
import tensorflow as tf
import numpy as np
from utils import ExperienceReplayBuffer
from network_models import DRQN
import json
import argparse
import datetime

def train(params):
    # Create the Gym environment as defined in the collision_avoidance_env folder
    env = gym.make('CollisionGazebo-v0')
    # Get the size of the action space and the state space, to be used in the neural network
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    # create a replay buffer that returns batch_size episodes, each of trace_length
    replay_buffer = ExperienceReplayBuffer(params.batch_size, params.trace_length)    

    tf.reset_default_graph()        
    # Create the main and the target Q-networks (Double architecture)
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=params.trace_length, state_is_tuple=True)
    main_qn = DRQN(name="main", state_size=state_size, action_size=action_size, hidden_size=params.trace_length, learning_rate=params.learning_rate, rnn_cell=lstm)
    target_lstm = tf.contrib.rnn.BasicLSTMCell(num_units=params.trace_length, state_is_tuple=True)
    target_qn = DRQN(name="target", state_size=state_size, action_size=action_size, hidden_size=params.trace_length, learning_rate=params.learning_rate, rnn_cell=target_lstm)
    
    # From the current TF graph, we extract the variables relevant to the Target network (which is the second set of variables since it was loaded onto TF after the main QN)
    # We do this because we do not perform actual training for the target network, but instead use the values of the main QN and assign them to the target QN
    target_vars_update = []
    trainable_vars = tf.trainable_variables()
    target_trainables_idx = len(trainable_vars) // 2                                                                    # Get starting inbex of the second half
    for main_trainable_idx, main_trainable_var in enumerate(trainable_vars[0:target_trainables_idx]):                   # Get all the main QN vars
        target_trainable = trainable_vars[main_trainable_idx + target_trainables_idx]                                   # Get the corresponding target QN var
        updated_value =  target_trainable.value() * params.tau + main_trainable_var.value() * (1 - params.tau)          # Use the update rule to update target QN var
        target_vars_update.append(target_trainable.assign(updated_value))                                               # Append the assign operation to the list to run on current tf session later

    with tf.Session() as sess:
        # Init all vars
        sess.run(tf.global_variables_initializer())
        
        # Create the saver to save model, and the summary file writer to write loss and reward to a file for visualizatino on tensorboard
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(params.summary_path, sess.graph)
        all_summaries = tf.summary.merge_all()
            
        all_rewards = []
        total_step_count = 0

        # Exploration parameters - start with full stochastic policy and converge to deterministic policy with no exploration
        explore_start = 1.0
        explore_stop = 0.01 
        explore_p = 1.0

        for ep in range(params.total_episodes):
            # Get initial state
            state = env.reset()

            # Reset the hidden state for the RNN cell
            reccurent_state_in = (np.zeros([1, params.trace_length]), np.zeros([1, params.trace_length]))

            episode_reward = 0
            episode_steps = 0
            while episode_steps < params.max_episode_length:
                if total_step_count < params.pretrain_steps:
                    next_reccurent_state_in = main_qn.rnn_hidden_state(sess, [state/255.0], 1, reccurent_state_in, 1)
                    # During pretraining, we always take a random action to fill the experience replay buffer with varied data samples
                    action = env.action_space.sample()
                else:
                    # We pass the explore param as the final args which is used in the dropout layer
                    Qs = main_qn.get_Q_values(sess, [state/255.0], 1, reccurent_state_in, 1, ((explore_start - explore_p) + explore_stop))
                    # During actual training, we calculate the Q values from the current state . . .
                    next_reccurent_state_in = main_qn.rnn_hidden_state(sess, [state/255.0], 1, reccurent_state_in, 1)
                    # . . and take the action that gives the maximum Q-value (i.e. greedy policy)
                    action = np.argmax(Qs)
                
                # Perform action on the Gym env
                next_state, reward, done, _ = env.step(action)
                # Add one step for current episode into the replay_buffer's episodic buffer 
                replay_buffer.add_step([state, action, reward, next_state, done])

                if total_step_count > params.pretrain_steps:
                    # Decay the exploration param
                    explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-params.epsilon_decay_rate*(total_step_count)) if explore_p > explore_stop else explore_p

                    # Take random samples from the Replay Buffer
                    states, actions, rewards, next_states, dones = replay_buffer.sample() 
                    # Reset the hidden state of the RNN cell before training
                    reccurent_hidden_state_reset = (np.zeros([params.batch_size, params.trace_length]), np.zeros([params.batch_size, params.trace_length])) 

                    # Update the target network towards the main network
                    for op in target_vars_update:
                        sess.run(op)

                    # Get Q vals for next state . . 
                    Qs_main_next_state = main_qn.get_Q_values(sess, next_states, params.trace_length, reccurent_hidden_state_reset, params.batch_size, 1.0)
                    # . . max of which is the action that will be taken in the next state
                    action_next_state = np.argmax(Qs_main_next_state, axis=1)

                    # Using the target Q network, we get the target Q values for the next state
                    Qs_target_next_state = target_qn.get_Q_values(sess, next_states, params.trace_length, reccurent_hidden_state_reset, params.batch_size, 1.0)
                    target_Qs = Qs_target_next_state[range(params.batch_size * params.trace_length), action_next_state]

                    # Q-Learning rule
                    targets = (target_Qs * (-(dones - 1)) * params.gamma) + rewards
                    
                    # Update the network to minimize the loss
                    loss = main_qn.update_network(sess, states, targets, actions, params.trace_length, reccurent_hidden_state_reset, params.batch_size, 1.0)
                
                    # Write relevant information to the summary file
                    if total_step_count % params.summary_out_every == 0:
                        scalar_summ = tf.Summary()
                        scalar_summ.value.add(simple_value=explore_p, tag='Explore P')
                        scalar_summ.value.add(simple_value=loss, tag='Mean loss')
                        scalar_summ.value.add(simple_value=np.mean(all_rewards[-10:]), tag='Mean reward')
                        summary_writer.add_summary(scalar_summ, total_step_count)
                        summary_writer.flush()

                # Update vars for next step
                state = next_state
                reccurent_state_in = next_reccurent_state_in
                total_step_count += 1        
                episode_steps += 1
                episode_reward += reward

                # If done from the Gym env, break from the inner loop
                if done:
                    break

            # If episode is longer than the trace length, we flush the episodic buffer into the replay buffer
            if len(replay_buffer.episode_buffer) >= params.trace_length:
                replay_buffer.flush()

            all_rewards.append(episode_reward)
            print('Episode: {}'.format(ep+1), 'Total reward: {}'.format(episode_reward), 'Explore P: {:.4f}'.format(explore_p)) 

            # Save model
            if params.save_model and total_step_count > params.pretrain_steps and ep % params.save_model_interval == 0:
                print('Saving model...')
                saver.save(sess, params.model_path +'/model' + str(ep+1) + '.ckpt', total_step_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", help='Size of batches drawn from the experience buffer for training', type=int, default=16)
    parser.add_argument("--trace_length", help='Length of each batch from the experience buffer', type=int, default=8)    
    parser.add_argument('--total_episodes', help='Number of episodes for training', type=int, default=1000000)
    parser.add_argument('--pretrain_steps', help='Initial random steps taken before training', type=int, default=1500)
    parser.add_argument('--h_size', help='Units of the hidden layer', type=int, default=512)
    parser.add_argument('--max_episode_length', help='Maximum length of an episode', type=int, default=300)
    parser.add_argument('-re', '--render_env', help='Boolean flag on whether the graphics of the env should be shown', action='store_true')
    parser.add_argument('--gamma', help='Gamma used as the discount factor for credit assignment', type=float, default=0.99)
    parser.add_argument('--tau', help='Rate of convergence of the target network to the main netork', type=float, default=0.98)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate for the Adam Optimizer', type=float, default=1e-5)
    parser.add_argument('--epsilon_decay_rate', help='Rate at which the exploration parameter decays', type=float, default=1e-5)
    parser.add_argument('--save_model_interval', help='Rate of model saving', type=int, default=5)
    parser.add_argument('--summary_out_every', help='Steps after which summary is written to disk', type=int, default=200)
    parser.add_argument('-sm', '--save_model', help='Periodically save model parameters', action='store_true')
    parser.add_argument('--model_path', help='Location on disk where the model is saved', default='./models/' + str(datetime.datetime.now()))
    parser.add_argument('--summary_path', help='Location on disk where the summary is saved', default='./summary/' + str(datetime.datetime.now()))

    params = parser.parse_args()
    train(params)