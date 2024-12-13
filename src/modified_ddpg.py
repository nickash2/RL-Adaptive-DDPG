from src.agent import DDPG
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdaptiveDDPG(DDPG):
    def __init__(self, 
                alpha,
                beta,
                tau_min,
                tau_max,
                performance_metric = 0,
                *args,
                **kwargs
                ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.performance_metric = performance_metric
        self.p_min = 1
        self.p_max = 0

    def compute_performance_metric(self) -> None:
        # alpha * R_avg + beta * var(Q)
        average_reward = np.mean(self.episode_rewards)
        print("varq")
        var_Q = torch.var(self.critic.value, dim=None, unbiased=True, keepdim=False).item()
        print("after varq")
        new_performace_metric = self.alpha * average_reward + self.beta * var_Q
        
        if new_performace_metric > self.p_max:
                self.p_max = new_performace_metric
        elif new_performace_metric < self.p_min:
                self.p_min = new_performace_metric
    
        self.performance_metric = new_performace_metric
    
    def update_tau(self) -> float:
        # tau = tau_min + (tau_max - tau_min) * sin^2 (pi * (P - P_min) \ (P_max - P_min)
        P = np.clip(self.performance_metric, self.p_min, self.p_max)
        self.tau = self.tau_min + (self.tau_max - self.tau_min) * np.sin(np.pi * (P - self.p_min) / (self.p_max - self.p_min))**2
        return self.tau


    def update(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return None, None

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

        # Compute Performance Metric and update tau adaptively
        self.compute_performance_metric()
        self.update_tau()

        # Soft update target networks
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()