# small_world.py
#
# Try training NN agents with simple genetic algorithm.
#
# Inspired in part by Such et al. 2017, Deep Neuroevolution.
# http://arxiv.org/abs/1712.06567

# World:
#
# Color small board that includes:
#   One pixel as the agent (green)
#   Pixel of barriers. Locations where the agent cannot be. (red)
#   Pixels that indicate where agent has been. (blue)
#   All other pixels are black
# Agent can see entire board.
# Agent can move in any of 4 diretions.
# If agent hits a barrier, it suffers a timeout penalty.
# Agent scores point for each pixel it travels over.
# Agent can move in environment for fixed period of time, equal to the number of open pixels on the board.

# Genetic algorithm:
#
# Genes are weights of a NN.
# Weights are initialized as ~N(0,1). Biases initialized as zeros.
# Run GA in discrete generations:
#   Population size P.
#   Evaluate each agents.
#   Mutate N-1 new agents from top T, by adding random ~N(0,sig) noise to parameters.
#   Keep top agent, to get N agents for next generation.


# TODO:
# 1. Have children in relative proportion to score, probabilistically
# 2. Make action decision stochastically.
# 3. Add input that indicates a barrier was hit on previous step.


import numpy as np
import time
import sys
import random
from collections import OrderedDict 
import pickle
import copy
import pdb
import matplotlib.pyplot as plt
plt.ion()


if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    # seed = random.randint(0, 1e9)
    # seed = 267939820
    seed = 1
print('seed = %d' % seed)
np.random.seed(seed)
random.seed(seed)



######################################
## User specifications

color_wall = np.asarray([255, 0, 0]) / 255
color_agent = np.asarray([0, 255, 0]) / 255
color_path = np.asarray([0, 0, 255]) / 255

timeout_wall = 5

n_agents = 500
n_parents = 20
n_generations = 5000

sig_mutate = 0.05

######################################



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x + 1e-10) / (e_x.sum() + 1e-10)

def mutate(x, p=0.01):
    ix = np.where(np.random.rand(*x.shape) < p)
    # x[ix] = np.random.randn(len(ix[0]))
    x[ix] = x[ix] + np.random.randn(len(ix[0]))/10
    return x


class Agent():
    # A neural network.
    # Need to eventually make this recurrent, and with PyTorch.

    # params are in format: [[W1,bi], [W2,b2], ...]

    def __init__(self, model_size=None, params=None):
        # Either model_size of params must be defined, but not both.
        assert model_size is not None or params is not None
        assert (not (model_size is not None and params is not None))

        if params:
            self.params = params
            self.n_layers = len(params)
            # self.n_neurons = np.zeros(self.n_layers+1)
            # for i in range(self.n_layers):
            #     self.n_neurons[i] = params[i][0].shape[0]
            # self.n_neurons[-1] = params[-1][0].shape[1]
        else:
            self.n_layers = len(model_size) - 1
            self.params = []
            for i in range(self.n_layers):
                W = np.random.randn(model_size[i+1], model_size[i])
                b = np.zeros((model_size[i+1]))
                self.params.append([W, b])

    def action(self, image, stochastic=False):
        output = image
        for i in range(self.n_layers):
            output = np.matmul(self.params[i][0], output) + self.params[i][1]
            if i < self.n_layers - 1:
                output = np.minimum(0, output)  # ReLU, if not final layer
        
        if stochastic:
            output = softmax(output)
            action = np.random.choice([0, 1, 2, 3], p=output)
            return action
        else:
            return np.argmax(output)

    def mutate(self):
        params = copy.deepcopy(self.params)
        for i_layer in range(self.n_layers):
            params[i_layer][0] += np.random.randn(*params[i_layer][0].shape) * sig_mutate
            params[i_layer][1] += np.random.randn(*params[i_layer][1].shape) * sig_mutate
        agent = Agent(params=params)
        return agent


class World():
    def __init__(self, board_size=[7, 7], f_obstacles=0.2):
        self.board_barriers = np.zeros(board_size+[3])
        self.score = 0
        self.agent_steps_taken = 0

        # Add barriers
        board_type = 'random'

        if board_type=='random':
            n = np.prod(board_size)
            p = np.random.permutation(n)
            p = p[:int(n*f_obstacles)]
            ix = np.unravel_index(p, board_size)
            self.board_barriers[ix[0], ix[1], :] = color_wall

        # Add wall barriers
        self.board_barriers[:,0,:] = color_wall
        self.board_barriers[:,-1,:] = color_wall
        self.board_barriers[0,:,:] = color_wall
        self.board_barriers[-1,:,:] = color_wall

        self.unoccupied = np.sum(self.board_barriers, axis=2) == 0
        self.max_possible_steps = np.sum(self.unoccupied)

        self.agent_start_loc = None

        self.reset()

    def add_agent(self, agent, random_loc=False):
        if random_loc:
            self.agent_start_loc = None
        self.agent = agent
        self._place_agent()

    def reset(self):
        self.board = np.copy(self.board_barriers)
        self.score = 0
        self.agent_steps_taken = 0
        self.agent = None
        self.bumped = -1

    def _place_agent(self):
        if self.agent_start_loc is None:
            ix = np.where(self.unoccupied)
            if len(ix[0])==0:
                print('\nEnding simulation: No free locations remain on board.')
                sys.exit()
            i = np.random.randint(len(ix[0]))
            loc = np.array([ix[0][i], ix[1][i]])
        else:
            loc = self.agent_start_loc
        self.agent_loc = loc
        self.agent_start_loc = loc
        self.board[loc[0], loc[1], :] = color_agent

    def _is_occupied(self, loc):
        content = self.board_barriers[loc[0], loc[1], :]
        return np.sum(content) > 0

    def step(self, stochastic=False):
        self.agent_steps_taken += 1

        sensation = np.append(self.board.flatten(), self.bumped)
        action = self.agent.action(sensation, stochastic=stochastic)
        assert action in [0, 1, 2, 3]

        # Get *desired* new agent location
        if action==0:
            new_loc = [self.agent_loc[0], self.agent_loc[1]+1]    # move right
        elif action==1:
            new_loc = [self.agent_loc[0], self.agent_loc[1]-1]    # move left
        elif action==2:
            new_loc = [self.agent_loc[0]+1, self.agent_loc[1]]    # move down
        elif action==3:
            new_loc = [self.agent_loc[0]-1, self.agent_loc[1]]    # move up

        # Move agent, or give time penalty if hit barrier
        b_occupied = self._is_occupied(new_loc)
        if b_occupied:
            # Time penalty
            self.agent_steps_taken += timeout_wall
            self.bumped = 1
        else:
            self.bumped = -1
            self.board[self.agent_loc[0], self.agent_loc[1]] = color_path
            self.board[new_loc[0], new_loc[1]] = color_agent
            self.agent_loc = new_loc

        return action, self.agent_steps_taken

    def get_score(self):
        n_black = np.sum(np.sum(self.board, axis=2)==0)
        score = self.max_possible_steps - n_black
        return score



######################
######################
######################

board_size = 8  # num of pixels on one size of square board
# model_size = [3*board_size**2, board_size**2, 4] # number of neurons in densely connected layers. Last layer must be 4.
# model_size = [3*board_size**2+1, board_size**2, 4] # number of neurons in densely connected layers. Last layer must be 4.
model_size = [3*board_size**2+1, 2*board_size, board_size, 4] # number of neurons in densely connected layers. Last layer must be 4.
agents = [Agent(model_size=model_size) for i in range(n_agents)]
agents = np.asarray(agents)

world = World(board_size=[board_size, board_size])

score_gen_median = np.full(n_generations, np.nan)
score_gen_mean = np.full(n_generations, np.nan)
score_gen_max = np.full(n_generations, np.nan)


plt.figure(1)
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)

t = time.time()
for i_gen in range(n_generations):

    world = World(board_size=[board_size, board_size])    # Build a new world with each generation
    
    scores = np.zeros(n_agents)

    n_iter = 3
    for i_iter in range(n_iter):
        for i_agent in range(n_agents):
            world.reset()
            world.add_agent(agents[i_agent], random_loc=True)
            
            b_done = False
            while not b_done:
                act, n_steps = world.step(stochastic=False)
                if n_steps >= world.max_possible_steps:
                    b_done = True

                # if i_gen > 9:
                #     plt.clf()
                #     plt.imshow(world.board)
                #     score = world.get_score()
                #     plt.title('action=%d, steps=%d, score=%d' % (act, n_steps, score))
                #     plt.draw()
                #     # plt.waitforbuttonpress(0.05)
                #     pdb.set_trace()

            scores[i_agent] += world.get_score()


    score_gen_median[i_gen] = np.median(scores)
    score_gen_mean[i_gen] = np.mean(scores)
    score_gen_max[i_gen] = np.max(scores)

    # Use the best agents as parents
    ix_sort = np.argsort(-scores)
    ix_best = ix_sort[:n_parents]
    parent_agents = agents[ix_best]

    # Also keep some random agents ad parents, for diverity
    i = np.random.permutation(n_agents-n_parents)
    ix_diverse = ix_sort[i[:n_parents] + n_parents]
    parent_agents = np.concatenate((parent_agents, agents[ix_diverse]))

    # agents = parent_agents[0:1]  # Keep the "elite" best agent for next generation
    agents = Agent(params=parent_agents[0].params) # Keep clone of the "elite" best agent for next generation

    # Create mutated children for next generation
    # TODO: HAVE MORE CHILDREN IF HAVE HIGHER SCORE, PROBABILISTICALLY.
    for i_child in range(n_agents-1):
        i_parent = np.random.randint(0, high=len(parent_agents))
        try:
            child = parent_agents[i_parent].mutate()
        except:
            pdb.set_trace()
        agents = np.append(agents, child)
    del parent_agents

    print('\tGeneration %d, duration = %0.0f sec, %0.2f sec/gen.' % (i_gen, time.time()-t, (time.time()-t)/(i_gen+1)), end='\r')

    plt.sca(ax1)
    plt.cla()
    plt.plot(np.sort(scores/(n_iter*world.max_possible_steps)), '.')
    ax = plt.axis()
    plt.axis([ax[0], ax[1], 0, 1])
    plt.grid(True)
    plt.title('Gen %d: Max=%d, Mean=%0.2f, Median=%2.f' % 
        (i_gen, score_gen_max[i_gen], score_gen_mean[i_gen], score_gen_median[i_gen]))
    plt.draw()
    plt.pause(0.01)
    # pdb.set_trace()

    n_points_goal = 100
    if i_gen%10==0:
        plt.sca(ax2)
        plt.cla()
        gen = np.arange(n_generations)

        n_points = min(n_points_goal, i_gen+1)
        n_avg = int((i_gen+1) / n_points)
        gen = gen[:n_avg*n_points]
        binned = np.split(gen, n_points)
        gen = np.asarray([np.max(b) for b in binned])

        x = score_gen_mean[:n_avg*n_points]
        binned = np.split(x, n_points)
        x = np.asarray([np.median(b) for b in binned])

        y = score_gen_median[:n_avg*n_points]
        binned = np.split(y, n_points)
        y = np.asarray([np.median(b) for b in binned])

        z = score_gen_max[:n_avg*n_points]
        binned = np.split(z, n_points)
        z = np.asarray([np.median(b) for b in binned])

        # plt.plot(gen, score_gen_mean, label='mean')
        # plt.plot(gen, score_gen_median, label='median')
        # plt.plot(gen, score_gen_max, label='max')
        plt.plot(gen, x, label='mean')
        plt.plot(gen, y, label='median')
        plt.plot(gen, z, label='max')
        
        ax = plt.axis()
        plt.axis([ax[0], ax[1], 0, n_iter*world.max_possible_steps])
        plt.grid(True)
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Scores')
        plt.draw()
        plt.pause(0.01)


# Play best agent
while True:
    plt.figure(3)
    ix_best = np.argmax(scores)
    world.reset()
    world.add_agent(agents[ix_best], random_loc=True)
    b_done = False
    while not b_done:
        act, n_steps = world.step(stochastic=False)
        if n_steps >= world.max_possible_steps:
            b_done = True

        plt.clf()
        plt.imshow(world.board)
        score = world.get_score()
        plt.title('action=%d, steps=%d, score=%d' % (act, n_steps, score))
        plt.draw()
        plt.waitforbuttonpress()


'''
class Agent:
    def __init__(self, id, parents=None, view_distance=5):
        assert id, 'Agent ID must be greater than 0.'
        self.id = id
        self.mates = []
        self.direction = self._get_random_direction()
        self.view = None
        self.bumped_object = 'none'
        self.color = np.random.randint(0,256,(3))
        self.color[np.random.randint(3)] = 0 # use saturation colors only
        self.color = (255*self.color/np.sqrt(np.sum(self.color**2))).astype(np.uint8)
        self.timeout = 0
        self.lifetime_steps = 0
        self.lifetime_wall_bumps = 0
        self.lifetime_forwards = 0
        self.steps_without_food = 100   # start off hungry
        self.fertility_countdown = 0   # can create offspring if less than zero
        self.lifetime_waits = 0        
        self.lifetime_turns = 0
        if parents:
            self.parents = [parents[0].id, parents[1].id]
        else:
            self.parents = None

        # vd = view_distance
        # self.weights_in2hid = np.random.randn(3*vd, 3*vd**2 + 2)
        # self.weights_hid2hid = np.random.randn(3*vd, 3*vd)
        # self.bias_hid = np.random.randn(3*vd)
        # self.act_hid = np.zeros(3*vd)
        # self.weights_hid2out = np.random.randn(3, 3*vd)
        # self.bias_out = np.random.randn(3)
        if not parents:
            vd = view_distance
            self.weights_in2hid = np.random.randn(3*vd, 3*vd**2 + 2)
            self.weights_hid2hid = np.random.randn(3*vd, 3*vd)
            self.bias_hid = np.random.randn(3*vd)
            self.act_hid = np.zeros(3*vd)
            self.weights_hid2out = np.random.randn(3, 3*vd)
            # self.weights_in2hid = np.random.randn(vd**2, 3*vd**2 + 2)
            # self.weights_hid2hid = np.random.randn(vd**2, vd**2)
            # self.bias_hid = np.random.randn(vd**2)
            # self.act_hid = np.zeros(vd**2)
            # self.weights_hid2out = np.random.randn(3, vd**2)
            self.bias_out = np.random.randn(3)
        else:
            sz_in2hid = parents[0].weights_in2hid.shape
            sz_hid2hid = parents[0].weights_hid2hid.shape
            sz_hid2out = parents[0].weights_hid2out.shape
            n_hid = sz_hid2hid[0]
            n_out = sz_hid2out[0]
            self.weights_in2hid = np.zeros(sz_in2hid)
            self.weights_hid2hid = np.zeros(sz_hid2hid)
            self.bias_hid = np.zeros(n_hid)
            self.act_hid = np.zeros(n_hid)
            self.weights_hid2out = np.zeros(sz_hid2out)
            self.bias_out = np.zeros(n_out)

            # Perform crossover on hidden neurons
            parent0 = np.random.rand(n_hid) > 0.5
            self.weights_in2hid[parent0,:] = parents[0].weights_in2hid[parent0,:]
            self.weights_hid2hid[parent0,:] = parents[0].weights_hid2hid[parent0,:]
            self.bias_hid[parent0] = parents[0].bias_hid[parent0]
            self.weights_in2hid[~parent0,:] = parents[1].weights_in2hid[~parent0,:]
            self.weights_hid2hid[~parent0,:] = parents[1].weights_hid2hid[~parent0,:]
            self.bias_hid[~parent0] = parents[1].bias_hid[~parent0]

            # Perform crossover on output neurons
            parent0 = np.random.rand(n_out) > 0.5
            self.weights_hid2out[parent0,:] = parents[0].weights_hid2out[parent0,:]
            self.bias_out[parent0] = parents[0].bias_out[parent0]
            self.weights_hid2out[~parent0,:] = parents[1].weights_hid2out[~parent0,:]
            self.bias_out[~parent0] = parents[1].bias_out[~parent0]

            # Perform mutation on weights and biases
            p = 0.01
            self.weights_in2hid = mutate(self.weights_in2hid, p=p)
            self.weights_hid2hid = mutate(self.weights_hid2hid, p=p)
            self.bias_hid = mutate(self.bias_hid, p=p)
            self.weights_hid2out = mutate(self.weights_hid2out, p=p)
            self.bias_out = mutate(self.bias_out, p=p)


    def step(self, deterministic=False):
        turn = False

        # Assemble sensory input
        visual = self.view.flatten() / 255
        somato = np.zeros(2)
        if self.bumped_object=='wall':
            somato[0] = 1
        elif self.bumped_object=='agent':
            somato[1] = 1
        sensation = np.concatenate((visual, somato))

        # Process input with agent's model
        self.act_hid = np.matmul(self.weights_in2hid, sensation) + np.matmul(self.weights_hid2hid, self.act_hid) + self.bias_hid
        # self.act_hid = np.maximum(self.act_hid, 0)  # ReLU
        self.act_hid = np.clip(self.act_hid, 0, 1)
        # self.act_hid = softmax(self.act_hid)
        out = np.matmul(self.weights_hid2out, self.act_hid)

        if deterministic:
            ix = np.argmax(out)
            action = ['forward', 'turn_left', 'turn_right'][ix]
        else:
            out = softmax(out)
            action = np.random.choice(('forward', 'turn_left', 'turn_right'), 1, p=out)
        # if self.bumped_object!='none':
        #     action = np.random.choice(('turn_left', 'turn_right'), 1)
        # else:
        #     action = 'forward'

        # Set new direction if a turn was made
        if action=='turn_left':
            turn = True
            ix_dir = np.where(np.all(directions_possible==self.direction, axis=1))[0][0]
            ix_dir = (ix_dir+1)%4
            self.direction = directions_possible[ix_dir]
            self.timeout += 1
        if action=='turn_right':
            turn = True
            ix_dir = np.where(np.all(directions_possible==self.direction, axis=1))[0][0]
            ix_dir = (ix_dir-1)%4
            self.direction = directions_possible[ix_dir]
            self.timeout += 1

        # Update timeout
        if self.bumped_object=='wall':
            # self.direction = self._get_random_direction()
            self.timeout = timeout_wall
        if self.bumped_object=='agent':
            # self.direction = self._get_random_direction()
            self.timeout -= 1
        else:
            self.timeout -= 1

        return self.direction, turn, self.timeout

    def _get_random_direction(self):
        n = np.random.randint(len(directions_possible))
        return directions_possible[n]


# Create new agents or load from file
load_agents = True
agents = OrderedDict()
if load_agents:
    with open('agents.pkl', 'rb') as input:
        while True:
            try:
                a = pickle.load(input)
                agents[a.id] = a
            except:
                break
else:
    for i in range(1, n_agents+1):
        agents[i] = Agent(i, view_distance = view_distance)


# Create board and add agents in random locations
board_size = [64,64]   # height, width
# board_size = [128, 128]   # height, width
board = Board(board_size=board_size, f_obstacles=0.1, view_distance=view_distance)
for key, a in agents.items():
    board.add_agent(a)

if b_render:
    plt.figure(1)
    plt.imshow(board.image)
    plt.pause(t_frame)

i_step = 0
t_start = time.time()
cnt_mate_collisions = 0

hist_step = np.asarray([np.nan])
hist_speed_mean = np.asarray([np.nan])
hist_speed_median = np.asarray([np.nan])
hist_speed_std = np.asarray([np.nan])
hist_speed_max = np.asarray([np.nan])
hist_bumprate_mean = np.asarray([np.nan])
hist_age_median = np.asarray([np.nan])
hist_speed_quartile = np.full((3,1), np.nan)

best_speed = 0

# while i_step < n_steps:
print('Step: {}, n_alive: {}, n_total: {}, Dur: {:.0f}, Steps per sec: {:.1f}'.format(i_step, n_agents, n_agents, 0, 0), end='\r')
while True:
    # board.add_food(frac_pix=0.0005)
    # board.add_food(n_pix=n_agents_start)
    dur = time.time() - t_start
    n_alive_agents = len(board.agent_locations)
    print('Step: {}, speed: {:.2f}, n_alive: {}, n_total: {}, Dur: {:.0f}, Steps per sec: {:.1f}'.format(i_step, hist_speed_mean[-1], n_alive_agents, n_agents, dur, i_step/dur), end='\r')
    keys = list(agents.keys())
    agents_to_kill = np.empty(0, dtype=np.int)
    for key in keys:
        a = agents[key]
        a.lifetime_steps += 1
        a.steps_without_food += 1
        a.fertility_countdown -= 1
        bumped_object = 'none'
        if i_step < np.inf:
            direction, turn, timeout = a.step(deterministic=False)
        else:
            direction, turn, timeout = a.step(deterministic=True)

        if turn:
            a.lifetime_turns += 1

        if timeout > 0:
            a.lifetime_waits += 1

        speed_avg_factor = 0.95


        if timeout <=0 and not turn and not np.any(key==agents_to_kill):
            loc_new = board.get_agent_location(a.id) + direction

            if not board.unoccupied[loc_new[0], loc_new[1]]:
                # Bumped into wall or other agent
                board.agent_speeds[a.id] = speed_avg_factor*board.agent_speeds[a.id] + 0.0
                mate_id = board.agent_at_location(loc_new)

                if mate_id is None:
                    # Bumped into wall or food
                    if np.array_equal(board.image[loc_new[0], loc_new[1], :], wall_gray3):
                        bumped_object = 'wall'
                        a.lifetime_wall_bumps += 1
                    elif np.array_equal(board.image[loc_new[0], loc_new[1], :], food_gray3):
                        bumped_object = 'none'  # bumped into food, but just eat it.
                        a.steps_without_food = 0
                        board.move_agent(a, loc_new)
                        a.lifetime_forwards += 1
                        board.agent_speeds[a.id] = speed_avg_factor*board.agent_speeds[a.id] + (1-speed_avg_factor)
                        board.board_food[loc_new[0], loc_new[1]] = 0
                    else:
                        print('\nEnding simulation: Unknown object at new agent location.')
                        sys.exit()

                else:
                    # Bumped into other agent. Might be able to mate.
                    bumped_object = 'agent'

                    b_create_offspring = True

                    ## Don't create offspring if pair is parent/child.
                    #  Not because of taboo, but to prevent immediate mating after a birth.
                    if a.parents is not None and mate_id in a.parents:
                        b_create_offspring = False
                    if agents[mate_id].parents is not None and a.id in agents[mate_id].parents:
                        b_create_offspring = False

                    # ## Don't have offspring if too hungry
                    # if a.steps_without_food > 100:
                    #     b_create_offspring = False

                    ## Don't mate if this pair has mated before.
                    # This is to prevent agents from just idling in a local area and mating
                    # regularly with the same partners. Goal is to promote navigation by agents.
                    if mate_id in a.mates:
                        b_create_offspring = False

                    ## Don't mate if either agent is infertile
                    if a.fertility_countdown > 0 or agents[mate_id].fertility_countdown > 0:
                        b_create_offspring = False

                    # Create offspring, if all conditions satisfied
                    if b_create_offspring:
                        a.fertility_countdown = 30
                        agents[mate_id].fertility_countdown = 30

                        # n_offspring = random.choice([1, 2])
                        # n_offspring = np.random.choice([1, 2], 1, p=[0.8, 0.2])[0]
                        n_offspring = 1
                        id_offspring = np.arange(n_agents+1, n_agents+n_offspring+1)
                        for id_off in id_offspring:
                            n_agents += 1
                            agents[n_agents] = Agent(n_agents, parents=(a, agents[mate_id]), view_distance=view_distance)
                            board.add_agent(agents[n_agents], loc_target=board.get_agent_location(a.id))

                        # Add siblings to each other's "former mate" list (really it's a "dont't mate"
                        # list), to reduce likelihood of local population explosions.
                        for id1 in id_offspring:
                            for id2 in id_offspring:
                                agents[id1].mates.append(id2)

                        # Add IDs to mate lists, for both parents
                        a.mates.append(mate_id)
                        agents[mate_id].mates.append(a.id)

                        # Timeout for both parents
                        a.timeout = timeout_mate
                        agents[mate_id].timeout = timeout_mate

                        # # Add submissive mate to list of agents to kill after this step
                        # agents_to_kill = np.append(agents_to_kill, mate_id)

                        # Give agent credit for a forward move even though agent isn't actually
                        # moved due to difficulty with current code configuration.
                        a.lifetime_forwards += 1

            else:
                board.move_agent(a, loc_new)
                a.lifetime_forwards += 1
                board.agent_speeds[a.id] = speed_avg_factor*board.agent_speeds[a.id] + (1-speed_avg_factor)
        else:
            # Still in timeout, or turning
            board.agent_speeds[a.id] = speed_avg_factor*board.agent_speeds[a.id] + 0.0

        a.bumped_object = bumped_object


    # Kill the submissive mates
    agents_to_kill = np.unique(agents_to_kill)
    for mate_id in agents_to_kill:
        board.remove_agent(agents[mate_id])
        del agents[mate_id]

    # Determine how many additional agents to kill based on overpopulation
    n_non_agent = np.sum(np.sum(board.board, axis=2)==0)
    # n_max = int(round(percent_population_max/100*n_non_agent))
    n_max = n_agents_start
    n_alive_agents = len(board.agent_locations)
    n_kill = max(0, n_alive_agents-n_max)

    # # Randomly select agents to kill
    # keys = np.random.choice(list(agents.keys()), n_kill, replace=False)

    # # Kill the slowest agents
    # keys = list(agents.keys())
    # # speeds = np.asarray([board.agent_speeds[k] for k in keys])
    # lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
    # lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
    # speeds = lt_forwards / (lt_steps+0.001)
    # ix = np.argsort(speeds)
    # keys = [keys[i] for i in ix[:n_kill]]

    # # Kill the oldest agents
    # keys = list(agents.keys())
    # age = np.asarray([agents[k].lifetime_steps for k in keys])
    # ix = np.argsort(-age)
    # keys = [keys[i] for i in ix[:n_kill]]

    # # Kill the hungriest agents
    # keys = list(agents.keys())
    # hunger = np.asarray([agents[k].steps_without_food for k in keys])
    # ix = np.argsort(-hunger)
    # keys = [keys[i] for i in ix[:n_kill]]
    # # ix = np.where(hunger > 100)[0]
    # # keys = [keys[i] for i in ix]

    # # Kill the oldest, hungriest agents
    # keys = list(agents.keys())
    # hunger = np.asarray([agents[k].steps_without_food for k in keys])
    # age = np.asarray([agents[k].lifetime_steps for k in keys])
    # ix = np.argsort(-hunger*age)
    # keys = [keys[i] for i in ix[:n_kill]]

    # Kill agents, once every "generation"
    n_alive_agents = len(board.agent_locations)
    if n_alive_agents > 2*n_agents_start:
        n_kill = n_alive_agents - n_agents_start

        # Kill the slowest agents
        keys = list(agents.keys())
        # speeds = np.asarray([board.agent_speeds[k] for k in keys])
        lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
        lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
        speeds = lt_forwards / (lt_steps+0.001)
        ix = np.argsort(speeds)
        keys = [keys[i] for i in ix[:n_kill]]
    else:
        keys = []


    # Do the actual killing
    for key in keys:
        board.remove_agent(agents[key])
        del agents[key]


    if b_render and i_step%100==0:
    # if b_render and i_step>4000:
        plt.figure(1)
        plt.clf()
        plt.imshow(board.image)
        n_alive_agents = len(board.agent_locations)
        plt.title('n_agents = %d' % (n_alive_agents))
        plt.pause(t_frame)


        # Get more agent data for plotting
        keys = list(agents.keys())
        # speeds = np.asarray([board.agent_speeds[k] for k in keys])
        lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
        lt_bumps = np.asarray([agents[k].lifetime_wall_bumps for k in keys])
        lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
        lt_turns = np.asarray([agents[k].lifetime_turns for k in keys])
        lt_waits = np.asarray([agents[k].lifetime_waits for k in keys])
        bump_rate = lt_bumps / (lt_steps+1)
        speeds = lt_forwards / (lt_steps+0.001)
        hunger = np.asarray([agents[k].steps_without_food for k in keys])

        speed_mean = np.mean(speeds)
        speed_median = np.median(speeds)
        hist_step = np.append(hist_step, i_step)
        hist_speed_mean = np.append(hist_speed_mean, np.mean(speeds))
        hist_speed_median = np.append(hist_speed_median, np.median(speeds))
        hist_speed_std = np.append(hist_speed_std, np.std(speeds))
        hist_speed_max = np.append(hist_speed_max, np.max(speeds))
        hist_bumprate_mean = np.append(hist_bumprate_mean, np.mean(bump_rate))
        hist_age_median = np.append(hist_age_median, np.median(lt_steps))

        # ix = np.where(lt_steps > 30)[0]
        # if ix.size > 0:
        #     hist_speed_quartile = np.concatenate((hist_speed_quartile, np.expand_dims(np.quantile(speeds[ix], [0.25, 0.5, 1.0]), axis=1)), axis=1)
        # else:
        #     hist_speed_quartile = np.concatenate((hist_speed_quartile, np.full((3,1), np.nan)), axis=1)

        ## Save the agents if the population is better than all previous ones
        if speed_median > best_speed:
            best_speed = speed_median
            with open('agents.pkl', 'wb') as output:
                keys = list(agents.keys())
                for key in keys:
                    pickle.dump(agents[key], output, pickle.HIGHEST_PROTOCOL)


        plt.figure(2)
        plt.subplot(3,1,1)
        plt.cla()
        # plt.hist(speeds, bins=np.arange(0,1,0.05))
        # plt.title('mean speed = %f' % (speed_mean))
        plt.hist(lt_steps, bins='auto')
        plt.title('age (lifetime_steps)')

        plt.subplot(3,1,2)
        plt.cla()
        # plt.plot(hist_step, hist_bumprate_mean)
        plt.plot(hist_step, hist_speed_median)
        # plt.errorbar(hist_step, hist_speed_mean, hist_speed_std)
        # plt.errorbar(hist_step, hist_speed_quartile[1,:], hist_speed_quartile[[0,2],:])
        # plt.plot(np.full(speeds.shape, i_step), speeds, 'bo', mfc='none')
        plt.title('Median speed')
        ax = plt.axis()
        plt.axis(list(ax[0:2]) + [0,1])
        plt.grid(True)

        plt.subplot(3,1,3)
        plt.cla()
        # plt.plot(hist_step, hist_bumprate_mean)
        plt.plot(hist_step, hist_age_median)
        # plt.errorbar(hist_step, hist_speed_mean, hist_speed_std)
        plt.title('Median age')
        plt.grid(True)
        plt.pause(t_frame)

        # if i_step >= 2000:
        #     pdb.set_trace()

    i_step += 1

t_total = time.time() - t_start
print('Total time = %f seconds, time/step=%f' % (t_total, t_total/n_steps))

if b_render:
    plt.imshow(board.image)

'''

