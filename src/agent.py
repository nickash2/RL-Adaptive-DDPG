from torch.utils.tensorboard import SummaryWriter
from src.utils.models import Actor, Critic
from gymnasium.spaces import Space
from typing import List
import torch
from src.utils.replay_buffer import ReplayBuffer
from src.utils.metrictracker import MetricsTracker
import numpy as np
from src.utils.noise import AbstractNoise
from typing import Tuple
from tqdm import trange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

class DDPG:
    def __init__(self, 
                 discount_factor: float, 
                 action_space: Space, 
                 hidden_size: List[int], 
                 input_size: int, 
                 tau: float,
                 critic_lr: float,
                 actor_lr: float,
                 buffer_size: int,
                 callback: List[MetricsTracker] = None,
                 seed: int | None = None,
                 log_dir: str = "./runs/DDPG",
                 update_interval: Tuple[int, str] = (1, "step"),
                 learning_starts: int = 1000,
                 batch_size: int = 256,
                 verbose: bool = False,
                 ):
        # Set random seed if seed is NoneType
        self.seed = np.random.randint(0, 2**32 - 1) if seed is None else seed

        # Set seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.verbose = verbose
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.tau = tau
        self.callback = callback
        self.writer = SummaryWriter(log_dir=log_dir)  # Initialize TensorBoard writer

        self.actor = Actor(hidden_size, input_size, action_space)
        self.target_actor = Actor(hidden_size, input_size, action_space)

        self.critic = Critic(hidden_size, input_size, action_space)
        self.target_critic = Critic(hidden_size, input_size, action_space)

        # Move the actor and target actor networks to the specified device (e.g., GPU or CPU)
        self.actor = self.actor.to(device)
        self.target_actor = self.target_actor.to(device)

        # Move the critic and target critic networks to the specified device
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)

        # Initialize the target networks with the same weights as the original networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)

        # Initialize the optimizers for the actor and critic networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.update_interval = update_interval
        self.learning_starts = learning_starts
        self.episode_rewards = []

    def train(self, env, num_episodes: int, noise: AbstractNoise = None, max_steps: int = 1000):
        if self.callback:
            for cb in self.callback:
                cb._on_training_start()  # Notify callbacks that training is starting

        episode_range = trange(num_episodes, desc="Training Progress", unit="episode") if self.verbose else range(num_episodes)

        for episode in episode_range:
            state, _ = env.reset()  # Reset the environment at the start of each episode
            episode_reward = 0
            noise.reset() if noise else None  # Reset noise for each episode
            critic_loss, actor_loss = None, None

            for step in range(max_steps):
                state_tensor = torch.FloatTensor(state).to(device)
                action = self.select_action(state_tensor, noise=noise).cpu().numpy()

                next_state, reward, done, truncated, _ = env.step(action)

                self.replay_buffer.add_entry(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state
                # Step-based update interval (update every N steps)
                if self.update_interval[1] == "step" and step % self.update_interval[0] == 0:
                    if len(self.replay_buffer) >= self.learning_starts:
                        critic_loss, actor_loss = self.update(self.batch_size)

                # Notify callbacks about step progress
                if self.callback:
                    for cb in self.callback:
                        cb.locals = {"rewards": reward, "dones": done, "truncated": truncated}  # Set required callback locals
                        if not cb._on_step():  # If a callback returns False, stop training early
                            print("Training interrupted by callback.")
                            return self.episode_rewards

                if done or truncated:
                    state, _ = env.reset()
                    break

            # Episode-based update interval (update every N episodes)
            if self.update_interval[1] == "episode" and episode % self.update_interval[0] == 0:
                if len(self.replay_buffer) >= self.learning_starts:
                    critic_loss, actor_loss = self.update(self.batch_size)

            self.writer.add_scalar("Reward/Episode", episode_reward, episode)
            if critic_loss is not None and actor_loss is not None:
                self.writer.add_scalar("Loss/Critic", critic_loss, episode)
                self.writer.add_scalar("Loss/Actor", actor_loss, episode)

            self.episode_rewards.append(episode_reward)

        if self.callback:
            for cb in self.callback:
                cb._on_training_end()  # Notify callbacks that training is ending

        self.writer.close()  # Close the TensorBoard writer

        return self.episode_rewards



    def select_action(self, state: torch.FloatTensor, noise=None):
        x = state.to(device)

        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)  # Forward pass

        self.actor.train()  # Sets the actor back in training mode
        mu = mu.data

        if noise is not None:
            noise = torch.Tensor(noise.noise()).to(device)  # Assuming noise object has a noise() function
            mu += noise

        # Clip the output according to the action space
        return mu.clamp(self.action_space.low[0], self.action_space.high[0])

    def soft_update(self, target, source, tau):
        """
        Soft update implementation (change this for proposed modification)
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def update(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return None, None  # Skip update if there's not enough data

        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Critic update
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.discount_factor * target_q

        current_q = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()

