import numpy as np
import os, torch, sys
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from copy import deepcopy

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import *
from agents.base import Agent
from neural_networks.actor import ModularActor
from neural_networks.critic import ModularDistributedCritic
from agent_modules.memory import PriorityNStepReplayBuffer, NStepReplayBuffer
from agent_modules.noise import AdaptiveParamNoiseSpec, GaussianExploration
    
class D4PG_Agent(Agent):
    """An advance D4PG agent with an option to run on a simpler DDPG mode.
    The agent uses a distributional value estimation when running on D4PG vs
    the traditional single value estimation when running on DDPG mode."""
    
    def __init__(self, params):
        """Initialize an Agent object."""

        self.params = params
        self.update_target_every = params['update_target_every']
        self.gamma = params['gamma']
        self.action_size = params['actor_params']['action_size']
        self.agent_count = params['agent_count']
        self.num_atoms = params['critic_params']['num_atoms']
        self.v_min = params['critic_params']['v_min']
        self.v_max = params['critic_params']['v_max']
        self.update_target_type = params['update_target_type']
        self.device = params['device']
        self.name = params['agent_name']
        self.lr_reduction_factor = params['lr_reduction_factor']
        self.tau = params['tau']
        self.d4pg = params['d4pg']
        self.rolling_window = params['rolling_window']
        self.per = params['per']
        self.writer = params['writer']
        self.param_noise = AdaptiveParamNoiseSpec() if params['param_noise'] == True else None
        self.clip_action = params['clip_action']
        self.action_max = params['action_max']
        self.action_min = params['action_min']
        self.mode = TRAIN
        self.clip_gradients = params['clip_gradients']
        self.gradient_clip = params['gradient_clip']

        # Distributes the number of atoms across the range of v min and max
        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)

        # Initialize time step count
        self.timesteps = 0
        
        # Normal and Target Actor networks
        self.actor = ModularActor(params['actor_params']).float().to(self.device)        
        self.actor_target = deepcopy(self.actor)

        if self.param_noise is not None:
            self.actor_perturbed = ModularActor(params['actor_params']).float().to(self.device)   

        if self.d4pg:
            # Normal and Target D4PG Critic networks
            self.critic = ModularDistributedCritic(params['critic_params']).float().to(self.device)
            self.critic_target = deepcopy(self.critic)
        else:
            # Normal and Target Critic networks
            self.critic = Critic(params['critic_params']).float().to(self.device)
            self.critic_target = Critic(params['critic_params']).float().to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = params['actor_params']['lr'], weight_decay = params['actor_params']['weight_decay'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = params['critic_params']['lr'], weight_decay = params['critic_params']['weight_decay'])
        
        self.schedule_lr = params['schedule_lr']
        self.lr_steps = 0

        # Create learning rate schedulers if required to reduce the learning rate
        # depeninding on plateuing of scores
        if self.schedule_lr:
            self.actor_scheduler = ReduceLROnPlateau(
                self.actor_optimizer, 
                mode = 'max', 
                factor = params['lr_reduction_factor'],
                patience = params['lr_patience_factor'],
                verbose = False,

            )
            self.critic_scheduler = ReduceLROnPlateau(
                self.critic_optimizer, 
                mode = 'max',
                factor = params['lr_reduction_factor'],
                patience = params['lr_patience_factor'],
                verbose = False,
            )
        print("\n################ ACTOR ################\n")
        print(self.actor)
        
        print("\n################ CRITIC ################\n")
        print(self.critic)
        
        # Replay memory
        if params['per']:
            self.memory = PriorityNStepReplayBuffer(params['memory_replay_params'])
        else:
            self.memory = NStepReplayBuffer(params['memory_replay_params'])

        # Initiate exploration parameters by adding noise to the action
        self.noise = GaussianExploration(params['noise_params'])

    def act(self, states, portfolio_vectors, action_noise):
        """Returns action for given state as per current policy."""
        portfolio_vectors = portfolio_vectors.to(self.device).float()
        states = states.to(self.device).float()

        # Enables or disables tools depending on mode
        if self.mode == TRAIN or self.mode == VALID:
            # Implements Parameter noise requirement of perturbed actor or normal active actor
            if self.param_noise is not None:
                with torch.no_grad():
                    if str(self.device) == 'cuda':
                        # print("CUDA")
                        action = self.actor_perturbed(states, portfolio_vectors).detach().to('cpu').numpy()
                    elif str(self.device) == 'cpu':
                        # print("CPU")
                        action = self.actor_perturbed(states, portfolio_vectors).numpy()
                    else:
                        print("\n\n################# WARNING INVALID DEVICE {}".format(str(self.device)))
                        # exit()

            else:
                with torch.no_grad():
                    if str(self.device) == 'cuda':
                        # print("CUDA")
                        action = self.actor(states, portfolio_vectors).detach().to('cpu').numpy()
                    elif str(self.device) == 'cpu':
                        # print("CPU")
                        action = self.actor(states, portfolio_vectors).numpy()
                    else:
                        print("\n\n################# WARNING INVALID DEVICE {}".format(str(self.device)))
                        # exit()
                    # self.writer.add_graph(self.actor, input_to_model=(states, portfolio_vectors))

            if action_noise:
                # print("\n#### ACTION", action)
                noise = self.noise.create_noise(action.shape)
                # print("\n#### NOISE", noise)
                action += noise
                # print("\n#### ACTION WITH NOISE", action)

        elif self.mode == TEST:
            with torch.no_grad():
                action = self.actor(states, portfolio_vectors).detach().to('cpu').numpy()

        action = torch.tensor(action)
        
        if self.clip_action:
            action = np.clip(action, self.action_min, self.action_max)      

        # print(action.shape)
    
        return action, self.noise.epsilon

    def step(self, states, actions, rewards, next_states, dones, portfolio_vectors, pretrain):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        actor_loss, critic_loss, param_noise = False, False, None
        self.timesteps += 1
        # print("AGENT ACTIONS", actions.shape)
        self.memory.add((states, actions, rewards, next_states, dones, portfolio_vectors))
        
        if self.mode == TRAIN or self.mode == VALID:
            learn = True
        else:
            learn = False    

        actor_loss, critic_loss = self.calculate_losses()

        if learn and actor_loss != False and critic_loss != False:
            self.learn_(actor_loss, critic_loss)
        
        if self.param_noise is not None and np.any(dones) and pretrain == False:
            with torch.no_grad():
                unperturbed_actions = self.actor(states.to(self.device).float(), portfolio_vectors.to(self.device).float()).detach().to('cpu').numpy()
                param_noise = self.adapt_param_noise(actions.to('cpu').numpy(), unperturbed_actions)

        return actor_loss, critic_loss, param_noise

    def reset(self):
        """Resets the agent"""
        self.memory.reset_rollouts()
        
        if self.param_noise is not None:
            self.perturb_actor_parameters()

    def calculate_losses(self):
        "Learns from experience using a distributional value estimation when in D4PG mode"
        actor_loss = False
        critic_loss = False

        # If enough samples are available in memory and its time to learn, then learn!
        if self.memory.ready():
            # Samples from the replay buffer which has calculated the n step returns in advance
            # Next state represents the state at the n'th step
            if self.per:
                states, next_states, actions, rewards, dones, portfolio_vectors, idxs, is_weights = self.memory.sample()
            else:
                states, next_states, actions, rewards, dones, portfolio_vectors = self.memory.sample()
                       
            # print("MEMORY P VECTORS", portfolio_vectors.shape)

            if self.d4pg:
                atoms = self.atoms.unsqueeze(0)
                # Calculate log probability distribution using Zw with regards to stored action
                # print("\n### ACTIVE CRITIC ONE\n")
                log_probs = self.critic(states, portfolio_vectors, actions, log = True)
                # self.writer.add_graph(self.critic, input_to_model=(states, portfolio_vectors, actions))
                # Calculate the projected log probabilities from the target actor and critic networks
                # Since back propogation is not required. Tensors are detach to increase speed
                target_dist, target_log_probs = self._get_targets(rewards, next_states, actions)
                # The critic loss is calculated using a weighted distribution instead of the mean to
                # arrive at a more accurate result. Cross Entropy loss is used as it is considered to 
                # be the most ideal for categorical value distributions as utlized in the D4PG
                critic_losses = -(target_dist * log_probs).sum(-1)
                critic_loss = critic_losses.mean()

                if self.per:
                    errors = target_log_probs - log_probs
                    self.memory.update_priority(idxs, critic_losses.to('cpu').detach().numpy())
            else:
                # Get predicted next-state action and Q values from target models
                action_next = self.actor_target(next_states)
                # target_critic_states = self._create_critic_input(next_states, action_next)
                Q_targets_next = self.critic_target(next_states, action_next).detach()
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
                # Compute critic loss
                Q_expected = self.critic(states, actions)
                critic_loss = F.mse_loss(Q_expected, Q_targets)
                if self.per:
                    errors = Q_targets - Q_expected
                    self.memory.update_priority(idxs, errors)

            if self.d4pg:
                # Predicts the action for the actor networks loss calculation
                predicted_actions = self.actor(states, portfolio_vectors)
                # Predict the value distribution using the critic with regards to action predicted by actor
                # print("\n### ACTIVE CRITIC TWO\n")
                probs = self.critic(states, portfolio_vectors, predicted_actions)
                # Multiply probabilities by atom values and sum across columns to get Q values
                expected_reward = (probs * atoms).sum(-1)
                # Calculate the actor network loss (Policy Gradient)
                # Get the negative of the mean across the expected rewards to do gradient ascent
                actor_loss = -expected_reward.mean()
            else:
                action_pred = self.actor(states)
                actor_loss = -self.critic(states, action_pred).mean()


        # Returns the actor and critic losses to store on tensorboard
        return actor_loss, critic_loss     

    def _get_targets(self, rewards, next_states, next_portfolio_vectors):
        """
        Calculate Yᵢ from target networks using the target actor and 
        and distributed critic networks
        """
        # print("TARGET NEXT PV {}".format(next_portfolio_vectors.shape))
        target_actions = self.actor_target(next_states, next_portfolio_vectors)
        # print("NEXT ACTION {}".format(target_actions.shape))
        target_log_probs = self.critic_target(next_states, next_portfolio_vectors, target_actions)

        # Project the categorical distribution
        projected_probs = self._get_value_distribution(rewards, target_log_probs)
        return projected_probs.detach(), target_log_probs.detach()

    def _get_value_distribution(self, rewards, probs):
        """
        Returns the projected value distribution for the input state/action pair
        """

        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Rewards were stored with the first reward followed by each of the discounted rewards, sum up the 
        # reward with its discounted reward
        projected_atoms = rewards.unsqueeze(-1) + self.gamma**self.memory.rollout_length * self.atoms.unsqueeze(0)
        projected_atoms.clamp_(self.v_min, self.v_max)
        b = (projected_atoms - self.v_min) / delta_z

        # Professional level GPUs have floating point math that is more accurate 
        # to the n'th degree than traditional GPUs. This might be due to binary
        # imprecision resulting in 99.000000001 ceil() rounding to 100 instead of 99.
        # According to sources, forcibly reducing the precision seems to be the only
        # solution to the problem. Luckily it doesn't result in any complications to
        # the accuracy of calculating the lower and upper bounds correctly
        precision = 1
        b = torch.round(b * 10**precision) / 10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()
        # print("\nREWAWRDS", rewards.unsqueeze(-1).shape, "ATOMS", self.atoms.unsqueeze(0).shape, "PROBS", probs.shape, "OUTPUT B ", b.shape, "UPPER BOUND", upper_bound.shape,"LOWER BOUND",  lower_bound.shape)

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(self.device)

        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()

    def adapt_param_noise(self, perturbed_actions, unperturbed_actions):
        """Adds noise to the agents parameters"""
        return self.param_noise.adapt(perturbed_actions, unperturbed_actions)
    
    def perturb_actor_parameters(self):
        """Apply parameter noise to actor model, for exploration"""
        self._hard_update(self.actor, self.actor_perturbed)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name].float()
            random = torch.randn(param.shape).to(self.device)
            param += random * self.param_noise.get_param_noise()
            param = param
        
        self.actor_perturbed = self.actor_perturbed.to(self.device)
        self.actor = self.actor.to(self.device)