# Adaptive Traffic Light Control
**Adeniyi Fagbewesa (amf249), Katie Wei (kw462), Eshita Sangani (ens57)**
### Introduction
In light of the rapid increase in vehicular traffic globally, traffic congestion has increasingly become a concern. The current traffic lights utilize fixed time control policies resulting in long waiting periods for all vehicles, increasing congestion. Notably, during peak traffic hours, the duration of green lights prove to be insufficient to accommodate the volume of vehicles. Moreover, existing traffic light policies lack the ability to adapt to unforeseen events such as accidents or road construction, because of their stagnant policies. 

This project addresses the issues of traffic congestion by implementing smart traffic lights which are able to adapt as traffic flows change by studying traffic patterns for a given intersection. Specifically, they will learn from a simulation using Sumo-RL which is an environment that enables the learner to optimize signal timing and sequence adjustments to facilitate smoother traffic flow. This project represents a proactive step towards the realization of smarter urban transportation networks. Through the integration of advanced robot learning algorithms, the envisioned smart traffic lights aspire to mitigate congestion, enhance road safety, and improve the overall commuter experience in urban environments.

### Problem
The implementation of Q-Learning requires the establishment of a well-defined and contextually relevant reward function. Although Sumo-RL provides a rudimentary reward function, iterative training of different networks reveal limited improvement in performance indicators such as average speed and waiting time. This prompted the need for devising a new reward function tailored towards specific metrics defined in the later sections. 

Essential criteria encompassed by the new metric included vehicular speed, queue length, waiting time, and lane queue variance. Central to the refinement process was the recognition of speed as a fundamental determinant of traffic efficiency, with higher velocities indicating smoother traffic flow and reduced congestion. Queue length emerged as another pivotal factor, reflecting the extent of accumulation at critical junctures, thereby directly influencing traffic throughput. The incorporation of lane variance assures equal distribution of traffic across multiple lanes which mitigates bottlenecks and optimizes resource utilization. Additionally, waiting time assumes significance as a key performance metric, delineating the temporal cost incurred by motorists at intersections.

In this project, we attempt to find the optimal combination of metrics for the reward function. We aim to discover the weights corresponding to each metric and any interaction terms between them. Additionally, the reward function should improve congestion in every network. Through this integration, the ensuing reinforcement learning framework is poised to encompass more nuanced and adaptive traffic signal control policies thus enhancing overall traffic efficiency and commuter experience.

### Approach
#### High Level
By modeling the intersection as a Markov decision problem, the end goal is to use deep q-learning neural networks with an offline pretrain and online training to successfully implement our new traffic light system. The intersection will be knowledgeable of the presence of vehicles and their speed as they approach the intersection. It would be able to learn a set of state and action policies that allow traffic lights to make optimized decisions based on their current state. These state action pairs, (q values), would be learned and saved at every iteration, and eventually would be updated until they reach convergence. To quantify the goal, the agent’s (traffic light) goal is to reduce the vehicle resting time at an intersection and to maximize the flow of traffic. The traffic light is trained using a rewards based system, where at each time step, the model has to decide whether to change the lights’ phases or not. 

#### State Space
The state space of the TrafficSignal class (a part of Sumo-RL) captures the current state of the traffic signal and the traffic conditions at the intersection it controls. There are three main components - the current green phase of the traffic signal, the lane traffic density, and the queue length. The current green phase is a one-hot encoded vector in which each element corresponds to a possible green phase, with a '1' marking the active phase and '0's elsewhere. For each incoming lane controlled by the traffic signal, the lane traffic density is calculated as the number of vehicles in the lane divided by the maximum capacity of the lane (the number of vehicles that could fit if the lane were fully occupied). This metric provides a normalized view of how full each lane is, helping the agent assess congestion levels. Similar to density, the queue length component measures the extent of vehicle queuing in each lane. Specifically, it counts the number of vehicles moving at a very low speed (indicative of a halt or slow crawl) and normalizes this by the lane's capacity. This information is crucial for identifying and responding to congestion hotspots where vehicles are frequently stopped. 

With these three elements forming our state space, the agent has a comprehensive view of the traffic signal’s current state and the traffic conditions at the intersection. 

#### Agent Design
##### Agent Initialization
The agent is initialized with a starting state, a state space, an action space, a learning rate $\alpha$, a discount factor $\gamma$, a Q table, and an exploration strategy. 

- The starting state is the initial state from which the agent begins its interaction with the environment - it is a snapshot of the traffic conditions and signal phase at the start of the simulation. 
- The state space represents all the possible states the agent could encounter, as defined in the earlier section. 
- The action space defines all the actions the agent can take. In the context of a traffic signal, these actions correspond to the selection of which green phase to activate next. 
- The learning rate is the step size that determines how much new information overrides the old information in the Q-table during updates. It balances the trade-off between exploration of new knowledge and exploitation of already learned values. 
- The discount factor determines the importance of future rewards. A value close to 1 gives greater importance to future rewards, while a lower value prioritizes immediate rewards. 
- The Q table is essential in q learning because it is where the agent stores the estimated rewards (Q-values) for each state-action pair. Initially, it's populated with zeros for the starting state. 
- The exploration strategy manages how the agent chooses actions, and in our case, is an instance of Epsilon Greedy.

##### Agent Behavior + Methods
The agent has two functions - act and learn. The act method  determines the next action the agent will take based on its current state and the Q-table. Using the exploration strategy, it decides whether to take a well-known action for maximizing reward or to explore new actions. The learn method takes in the next state and reward. After taking an action and observing the outcome (next state and reward), this method updates the Q-table. It uses the formula: 

$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$

Here, $s$ and $s$ are the current and next states, $a$ is the action taken, $R(s, a)$ is the reward received, and $\max Q(s, a)$ is the maximum Q-value for the next state across all possible actions. This update refines the agent's knowledge about the value of taking action $a$ in state $s$.

#### Reward Functions
We consider 4 different reward functions to optimize various aspects of traffic flow through the Q learning approach. Each reward function described below is tailored to encourage specific desirable behaviors or discourage inefficiencies in traffic management at intersections. Each of them are described in the following sections. 
##### Balance Queues Reward
This function aims to achieve balanced vehicle queues across all lanes. It calculates the reward based on the standard deviation of queue lengths. 
- Objective: Minimize the variability of queues across different lanes to prevent some lanes from becoming much more congested than others.
- Calculation: It computes the standard deviation of the vehicle queues across the lanes and then negates this value as the reward. A lower standard deviation (more balanced queues) results in a higher reward because of the negation.

##### Improve Flow Reward
This function focuses on improving the overall flow of traffic through the intersection:
 Objective: Increase the average speed of vehicles moving through the intersection, effectively improving the traffic flow.
- Calculation: It calculates the difference between the current average speed and the last recorded average speed. A positive difference (improvement in flow) yields a positive reward.

#### Lane Imbalance Punishment Reward
This function penalizes imbalances in vehicle counts across lanes:
- Objective: Maintain a balanced distribution of vehicles across lanes to prevent some lanes from being underused while others are overcrowded. 
- Calculation: It computes the variance of squared vehicle counts across lanes and adds the total vehicle count, then negates this sum as the reward. High variance and high vehicle count both contribute to a larger penalty.

#### Wait Lane Combined Reward
This is a combined function that applies both the exponential waiting time punishment and the lane imbalance punishment:
- Objective: Address both long waiting times and imbalances in vehicle distribution simultaneously.
- Calculation: It adds the rewards (actually penalties) from exponential-wait-time-punishment and lane-imbalance-punishment. Both penalties are designed to work together to optimize traffic flow and lane usage.

### Results
#### Parameter Analysis
Before applying the reward functions, we worked on the cross validation analysis to pick the optimal values of alpha, gamma, and decay.

The surface plot in Figure below visualizes the interaction between the learning rate $\alpha$ and the discount factor $\gamma$ in relation to the performance of a reinforcement learning algorithm, as measured by some metric of "reward."

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{param} 

The highest reward regions (yellow) occur around mid-range alpha values (0.3 to 0.7) and high gamma values (close to 0.9990). This suggests that moderate learning rates coupled with a high valuation of future rewards tend to yield the best performance for this specific task or model.  The surface is highly volatile, with multiple peaks and valleys. This also implies that the performance is sensitive to the settings of alpha and gamma, and small changes can significantly affect outcomes. Peaks in the surface likely indicate combinations where the learning dynamics and reward structure are well-matched. By focusing on regions where the reward is maximized, we fine tuned our agent to achieve better balance between exploration (learning new strategies) and exploitation (leveraging known strategies) while maintaining a strong focus on future outcomes. 

#### Reward Function Analysis
To understand the effects of the 10 reward functions, we simulated them on different traffic intersections to be able to gain a full understanding of advantages and disadvantages of each reward function after tuning our agent. Not every intersection created and tested upon will be described in the upcoming paragraphs, but are linked in the github in the references section. 

The metrics that we chose to analyze are listed below: 
- Total Stopped + Agents Total Spotted: the number of vehicles stopped at an intersection over time, an indicator of congestion. 
- Total Waiting Time + Agents Total Waiting Time: These measure the cumulative waiting time for all vehicles, reflecting the efficiency of traffic flow management.
- Total Waiting Time: This provides the average waiting time per vehicle
- Mean Speed: Average speed of vehicles in the system, with higher speeds generally representing smoother traffic flow.

In Figure 2, we provide a comparison between two reward functions, Lane-Equal and Improved-Flow for a 3x3 intersection setup. 

\begin{figure}[htb]
    \centering
    \includegraphics[width=0.8\textwidth]{3x3graph} % Replace 
    \caption{Measuring Stats of Different Reward Functions of 3x3 Intersection}
    \label{fig:example}
\end{figure}

Both Improved-Flow and Lane-Equal show increases in total stops and waiting times as the simulation progresses. However, Improved-Flow generally results in fewer total stops and a slower rate of increase in waiting times compared to Lane-Equal. Improved-Flow consistently exhibits lower mean waiting times throughout the simulation, although it does show an upward trend. In contrast, Lane-Equal demonstrates a steeper and continuous increase in mean waiting time. Improved-Flow maintains higher mean speeds for a longer duration than Lane-Equal, though both experience a decline over time. This suggests that while Improved-Flow is more effective at keeping traffic moving, there are still inefficiencies that could be addressed to optimize performance further. Improved mean speeds might come at the cost of more frequent stopping under certain conditions, indicating a trade-off between maintaining flow and minimizing stops.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{2lane} % Replace 
    \caption{Measuring Stats of Different Reward Functions of 2 lane Intersection}
    \label{fig:example}
\end{figure}

In Figure 3, we provide a comparison between all four reward functions for a 2 lane intersection setup. 

Wait-Lane generally results in fewer stops compared to other models, suggesting it effectively reduces idling and promotes smoother traffic flow. Lane-Equal and Balance-Queues show mid-level performance. They are better than the no-model scenario but not as effective as Improved-Flow, which might suggest a trade-off between balancing and overall traffic fluidity. Improved-Flow seems to have a similar or slightly worse impact compared to no-model, indicating that focusing solely on lane-specific waiting times may not adequately address broader intersection efficiency. We can also see that the data from no model performs consistently worse, showing the importance of having a tuned reward function. 

Upon these results, it shows that the wait lane reward function is the most optimal when considering the different metrics described above. The complexity of the intersection (3x3 vs. 2-lane) influences the effectiveness of the reward functions. Continuous monitoring and adjustment based on real-time data will further optimize performance, ensuring the system remains effective under different traffic conditions.

### References
1. Agand, P., Iskrov, A., \& Chen, M. (2023). Deep Reinforcement Learning-based Intelligent Traffic Signal Controls with Optimized CO2 emissions. In \textit{2023 IEEE/RSJ International Conference on Intelligent Robots and Systems}. IEEE.
2. Agand, P., \& Iskrov, A. (2021). EcoLight: Reward Shaping in Deep Reinforcement Learning for Ergonomic Traffic Signal Control. In \textit{NeurIPS 2021 Workshop on Tackling Climate Change with Machine Learning}.
