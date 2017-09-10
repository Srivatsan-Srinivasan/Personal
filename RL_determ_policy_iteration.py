import numpy as np

#inputs
rewards    = [-1,-1,-1,-1,10] #rewards function.
states     = [0,1,2,3,4] # different states.
actions    = [0,1] #0 -> left, 1-> right at each state.
next_state = [[0,0,1,2,3],[1,2,3,4,4]] #first row are results of "left" and second row results of "right"

#Helper function
def inttostr(x):
    if x == 0:
        value = "left"
    else:
        value = "right"
    return (value)

def policy_eval( policy, rewards, gamma, threshold = 1e-7, print_active = False ):
    values = np.zeros(len(states))
    count  = 0
    delta  = 1000 #arbitrarily high value.

    while delta > threshold :
        count+=1
        for s in states:
            #print("in state" + str(s))
            action    = policy[s]
            itervalue = rewards[next_state[action][s]] + gamma * values[next_state[action][s]]
            delta     = np.abs(itervalue - values[s]) #positive and negative deviances treated alike.
            values[s] = itervalue

    if print_active:
        policy_string = list(map(inttostr, policy))
        print("\n --------------Policy Evaluation--------------")
        print("Total number of iterations for policy evaluation with threshold "+ str(threshold)+ " is " + str(count))
        print("Policy value vector with given discount factor of "+ str(gamma)+" on policy " + str(policy_string)+ " is below")
        print(np.array(values))

    return(np.array(values))

def q_value(policy, rewards, gamma, threshold = 1e-7):
    print("\n ------------Q-function Computation----------")
    q_values   = np.zeros((len(actions),len(states)))
    policy_val = policy_eval( policy, rewards, gamma, threshold)
    policy_string = list(map(inttostr, policy))

    for action in actions:
        for state in states:
            q_values[action][state] = rewards[next_state[action][state]] + gamma * policy_val[next_state[action][state]]

    print("Q Values with first column representing left(action 0) and second representing right(action 1) on policy " + str(policy_string) + " is below")
    print(q_values)
    return q_values

def policy_improvement(init_policy, rewards, gamma, threshold = 1e-7 ):
    policy        = init_policy
    policy_update = True
    count         = 0
    print("\n -----------Policy Improvement Iterations----------")
    print("Sequence of policy iterations is as follows")

    while policy_update is True: #Trivial loop which will be eventually exited if policy converges.
        policy_update = False
        policy_val    = policy_eval(policy, rewards, gamma, threshold)
        count         = count + 1
        for state in states:
            init_action   = policy[state] #take left in all cases
            action_values = [0]* len(actions) #Array initialize with zeroes.
            for action in actions :
                #Calculate the q-value for a single action at this stage.
                action_values[action] = rewards[next_state[action][state]] + gamma * policy_val[next_state[action][state]]
            #Maximize the q-values
            action_max = np.argmax(action_values)
            #check if update needed.
            if action_max != init_action:
                policy_update = True #Checks if policy has been updated in this stage to re-iterate the loop.
            policy[state] = action_max
        policy_string = list(map(inttostr, policy))
        print(policy_string)

    print("Number of policy improvement operations is including the convergence step is " + str(count))
    print("Final improved policy is as below")

    policy_string = list(map(inttostr,policy))
    print( policy_string)
    return(policy)

#Policy here specifies "all left" for each state.
policy_l = [0,0,0,0,0]
policy_r = [1,1,1,1,1]
policies = [policy_l,policy_r]
for policy in policies:

    print('\n*********************************************************************')
    print("Input policy is " + str(list(map(inttostr, policy)) ))
    policy_values = policy_eval(policy,rewards,0.75,threshold = 1e-7,print_active=True)
    q_values = q_value(policy,rewards,0.75)
    opt_policy = policy_improvement(policy,rewards, 0.75)
