import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import pickle


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        eps_returns = []
        for i in range(eval_episodes):
                state, done, eps_return = eval_env.reset(), False, 0
                while not done:
                        action = policy.select_action(np.array(state))
                        state, reward, done, _ = eval_env.step(action)
                        avg_reward += reward
                        eps_return += reward
                eps_returns.append( eps_return )
                print(f"Evaluating episode [{i+1}/{eval_episodes}]: {eps_return:.2f}")

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward, eps_returns

if __name__ == "__main__":
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
        parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--discount", default=0.99)                 # Discount factor
        parser.add_argument("--tau", default=0.005)                     # Target network update rate
        parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
        parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
        parser.add_argument("--save_dir", type=str, required=True,
                help="dir to save evaluation results")
        args = parser.parse_args()

        file_name = f"{args.policy}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print("LongTail Evaluation")
        print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        env = gym.make(args.env)

        # Set seeds
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] 
        max_action = float(env.action_space.high[0])

        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": args.discount,
                "tau": args.tau,
        }

        # Initialize policy
        if args.policy == "TD3":
                # Target policy smoothing is scaled wrt the action scale
                kwargs["policy_noise"] = args.policy_noise * max_action
                kwargs["noise_clip"] = args.noise_clip * max_action
                kwargs["policy_freq"] = args.policy_freq
                kwargs["seed"] = args.seed
                kwargs["env_name"] = args.env
                policy = TD3.TD3(**kwargs)
        elif args.policy == "OurDDPG":
                policy = OurDDPG.DDPG(**kwargs)
        elif args.policy == "DDPG":
                policy = DDPG.DDPG(**kwargs)

        if args.load_model != "":
                policy_file = file_name if args.load_model == "default" else args.load_model
                policy.load(policy_file)
        
        avg_return, eps_returns = eval_policy(policy, args.env, args.seed, 1000)
        save_dict = { "eps-returns" : eps_returns }
        save_path = os.path.join( args.save_dir, file_name + "_longtail.pkl" )
        pickle.dump( save_dict, open(save_path, "wb") )
