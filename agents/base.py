import os, torch, sys
from abc import ABCMeta, abstractmethod

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import *

class Agent(object):
    """Interacts with and learns from the environment."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, params):
        """Initialize an Agent object given a dictionary of parameters.
        
        Params
        ======
        * **params** (dict-like) --- a dictionary of parameters
        """
        pass

    @abstractmethod
    def act(self, state, action):
        """Returns action for given state as per current policy.
        
        Params
        ======
        * **state** (array_like) --- current state
        * **action** (array_like) --- the action values
        """
        pass

    @abstractmethod
    def step(self, states, action, rewards, next_states, dones):
        """Perform a step in the environment given a state, action, reward,
        next state, and done experience.
        Params
        ======
        * **states** (torch.Variable) --- the current state
        * **action** (torch.Variable) --- the current action
        * **rewards** (torch.Variable) --- the current reward
        * **next_states** (torch.Variable) --- the next state
        * **dones** (torch.Variable) --- the done indicator
        """
        pass

    @abstractmethod
    def learn_(self):
        """Update value parameters using given batch of experience tuples."""
        pass

    def _update_target_networks(self):
        """
        Updates the target networks using the active networks in either a 
        soft manner with the variable TAU or in a hard manner at every
        x timesteps
        """

        if self.update_target_type == "soft":
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        elif self.update_target_type == "hard":
            self._hard_update(self.actor, self.actor_target)
            self._hard_update(self.critic, self.critic_target)

    def _soft_update(self, active, target):
        """
        Slowly updates the network using every-step partial network copies
        modulated by parameter TAU.
        """

        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(self.tau*param.data + (1-self.tau)*t_param.data)

    def _hard_update(self, active, target):
        """
        Fully copy parameters from active network to target network
        """
        # print("\n\n############################# STATE DICT #################################\n", active.state_dict())
        target.load_state_dict(active.state_dict())

    def step_lr(self, score):
        """Steps the learning rate scheduler"""
        if self.schedule_lr:
            self.actor_scheduler.step(score)
            self.critic_scheduler.step(score)
            self.lr_steps += 1

        self.noise.decay(score)
        
    
    def get_lr(self):
        """Returns the learning rates"""
        actor_lr = 0
        critic_lr = 0
        for params in self.actor_optimizer.params:
            actor_lr =  params['lr']
        for params in self.critic_optimizer.params:
            critic_lr =  params['lr']
        return actor_lr, critic_lr

    def set_mode(self, mode):
        """Changes the mode between train and test"""
        if mode == TRAIN:
            self.mode = TRAIN
            
        elif mode == TEST:
            self.mode = TEST
        elif mode == VALID:
            self.mode = VALID
        else:
            raise Exception('Invalid Mode being set for Agent')


    def save_agent(self, average_reward, trajectory, test_trajectory, timesteps, test_timesteps, timestep_index, episodes, save_history = False):
        """Save the checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor_target.state_dict(), 
            'critic_state_dict': self.critic_target.state_dict(), 
            'average_reward': average_reward,
            'trajectory': trajectory, 
            'test_trajectory': test_trajectory,                        
            'timesteps': timesteps, 
            'test_timesteps': test_timesteps,
            "timestep_index": timestep_index, 
            'episodes' : episodes
        }

        if self.mode == VALID:
            checkpoint['trajectory'] = 1
            checkpoint['test_trajectory'] = 1
            checkpoint['timesteps'] = 0
            checkpoint['test_timesteps'] = 0
            checkpoint['timestep_index'] = self.rolling_window + 1
        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints") 
        
        if self.mode == VALID:
            # filePath = 'checkpoints/' + self.name + '.pth'
            filePath = '../../../AI Checkpoints VALID/' + self.name + '.pth'
        else:
            filePath = '../../../AI Checkpoints TRAIN/' + self.name + '.pth'

        torch.save(checkpoint, filePath)

        if save_history:
            filePath = 'checkpoints\\' + self.name + '_' + str(trajectory) + '.pth'
            torch.save(checkpoint, filePath)


    def load_agent(self):
        """Load the checkpoint"""
        filePath = 'checkpoints\\' + self.name + '.pth'

        if os.path.exists(filePath):
            checkpoint = torch.load(filePath, map_location = lambda storage, loc: storage)

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_state_dict'])

            if self.param_noise is not None:
                self.actor_perturbed.load_state_dict(checkpoint['actor_state_dict'])

            average_reward = checkpoint['average_reward']
            trajectory = checkpoint['trajectory']
            timesteps = checkpoint['timesteps']
            # test_timesteps = checkpoint['test_timesteps']
            timestep_index = checkpoint['timestep_index']
            # test_trajectory = checkpoint['test_trajectory']
            episodes = checkpoint['episodes']

            test_timesteps = 0
            test_trajectory = 1

            self.save_agent(average_reward, trajectory, test_trajectory, timesteps, test_timesteps, timestep_index, episodes)
            
            print("Loading checkpoint - Average Reward {} at Trajectory {} Timesteps {} Episodes {} Timestep Index {}".format(average_reward, trajectory, timesteps, episodes, timestep_index))
            print("Agent Name -", self.name)
            self.timesteps = timesteps
            self.test_timesteps = test_timesteps
            return trajectory + 1, test_trajectory + 1, timesteps, test_timesteps, timestep_index, episodes
            
        else:
            print("\nCannot find {} checkpoint... Proceeding to create fresh neural network\n".format(self.name))   
            return 1, 1, 0, 0, self.rolling_window + 1, 0

    
    def learn_(self, actor_loss, critic_loss):
        "Runs backward pass through the network updating the weights"
        
        # Execute gradient descent for the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph = True)
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()

        # Execute gradient ascent for the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)

        self.actor_optimizer.step()                    
        actor_loss = actor_loss.item()

        # Updates the target networks every n steps
        if self.timesteps % self.update_target_every == 0:
            self._update_target_networks()  