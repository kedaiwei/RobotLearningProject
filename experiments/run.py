from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

from graphs import plot_stats

import argparse
import numpy as np

exp_name_map = {
    '2lane' : '2lane_intersection',
    '2x2' : '2x2grid',
    '3x3' : '3x3grid',
    '4x4' : '4x4grid',
    'single' : 'basic_intersection',
    'complex' : 'complex_intersection'
}

def runExperiment(exp_name = 'single', rwd_fns = ['avg-speed'], n_sec = 500, gui = True, n_runs = 1, plot = False,
                  alpha = 0.1, gamma = 0.99, decay = 0.995):
    
    tot_r = {}

    for rwd_fn in rwd_fns:
        net_file = 'nets/'+exp_name_map[exp_name]+'/'+exp_name+'.net.xml'
        route_file = 'nets/'+exp_name_map[exp_name]+'/'+exp_name+'.rou.xml'
        output_file = 'outputs/'+exp_name+'_'+rwd_fn

        env = SumoEnvironment(net_file=net_file,
                                route_file=route_file,
                                out_csv_name=output_file,
                                use_gui=gui,
                                reward_fn=rwd_fn,
                                num_seconds=n_sec, 
                                max_depart_delay=0)
        
        tot_r[rwd_fn] = 0

        for run in range(1, n_runs+1):
            initial_states = env.reset()
            ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                        state_space=env.observation_space,
                                        action_space=env.action_space,
                                        alpha=alpha,
                                        gamma=gamma,
                                        exploration_strategy=EpsilonGreedy(decay=decay)) for ts in env.ts_ids}
            infos = []
            done = {'__all__': False}
            
            while not done['__all__']:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)
                for agent_id in s.keys():
                    tot_r[rwd_fn] += r[agent_id]
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

            env.close()

    if plot:
        plot_stats(exp_name, rwd_fns)

    return {rwd_fn : tot_r[rwd_fn] / n_runs for rwd_fn in rwd_fns}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation experiments.')
    parser.add_argument('--exp', type=str, default='2lane', help='Experiment name (e.g., 2lane, single, complex)')
    parser.add_argument('--rwds', nargs='+', default=['avg-wait-time'], help='Rewards to compare (e.g., avg-wait-time, balance-queues, wait-lane)')
    parser.add_argument('--n_sec', type=int, default=500, help='Number of simulation seconds')
    parser.add_argument('--hide_gui', action='store_true', help='Whether to hide GUI')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--hide_plot', action='store_true', help='Whether to hide plot')

    args = parser.parse_args()
    runExperiment(exp_name=args.exp, rwd_fns=args.rwds, n_sec=args.n_sec, gui=not args.hide_gui, n_runs=args.n_runs, plot=not args.hide_plot)