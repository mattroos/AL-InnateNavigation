# Single-agent simulation
#
# Observe the behavior of a single agent...

import numpy as np
import time
import sys
import random
from collections import OrderedDict 
import pickle
import pdb
import matplotlib.pyplot as plt
plt.ion()

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    # seed = random.randint(0, 1e9)
    seed = 634602288
print('seed = %d' % seed)
np.random.seed(seed)
random.seed(seed)


######################################
## User specifications

timeout_wall = 5
timeout_mate = 0

view_distance = 5
# wall_gray = 85
# food_gray = 170
wall_gray = 128
food_gray = 255

n_agents_start = 80
b_render = True
t_frame = 0.001
######################################



# Order directions possible in counter-clockwise manner
directions_possible = np.asarray([[-1,0], [0,-1], [1,0], [0,1]])

wall_gray3 = np.full(3, wall_gray, dtype=np.uint8)
food_gray3 = np.full(3, food_gray, dtype=np.uint8)
n_agents = n_agents_start


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x + 1e-10) / (e_x.sum() + 1e-10)

def mutate(x, p=0.01):
    ix = np.where(np.random.rand(*x.shape) < p)
    # x[ix] = np.random.randn(len(ix[0]))
    x[ix] = x[ix] + np.random.randn(len(ix[0]))/10
    return x

class Board():
    def __init__(self, board_size=[128,128], f_obstacles=0.05, view_distance=5):
        assert view_distance%2==1, 'Agent view distance must be odd-valued.'
        board = np.zeros(board_size, dtype=np.uint8)
        board_type = 'lines'   # random, grouped
    
        if board_type=='random':
            n = np.prod(board_size)
            p = np.random.permutation(n)
            p = p[:int(n*f_obstacles)]
            ix = np.unravel_index(p, board_size)
            board[ix] = wall_gray

        elif board_type=='lines':
            n_lines = 30

            loc_x = np.random.randint(0, board_size[1], size=(n_lines,1))
            loc_y = np.random.randint(0, board_size[0], size=(n_lines,1))
            board[loc_y, loc_x] = wall_gray

            n_pix = n_lines
            n_wall = int(np.prod(board_size) * f_obstacles)
            while n_pix < n_wall:
                loc_x = loc_x + np.random.choice([-1, 0, 1], size=loc_x.shape)
                loc_x = np.mod(loc_x, board_size[1])
                loc_y = loc_y + np.random.choice([-1, 0, 1], size=loc_y.shape)
                loc_y = np.mod(loc_y, board_size[0])
                board[loc_y, loc_x] = wall_gray
                n_pix = np.sum(board > 0)
        
        elif board_type=='grouped':
            # Start with a few randomly located parts
            n = np.prod(board_size)
            p = np.random.permutation(n)
            p = p[:10]
            ix = np.unravel_index(p, board_size)
            board[ix] = wall_gray

            # Then add points dependent on the location of the seed points and beyond
            n_obs = int(np.prod(board_size)*f_obstacles)    # number of obstacle pixels to add
            cnt = len(p)
            while cnt < n_obs:
                # Pick an existing random point, and add a point at a distance
                # drawn from a Gaussian distribution.
                idx = np.where(board>0)
                ix = np.random.randint(0,cnt)
                y = idx[0][ix]
                x = idx[1][ix]
                d = np.random.randn(2)
                y = int(round(y + d[0]*board_size[0]/30))
                x = int(round(x + d[1]*board_size[1]/30))
                if x<0 or x>board_size[1]-1:
                    continue
                if y<0 or y>board_size[0]-1:
                    continue
                if board[y,x] > 0:
                    continue
                board[y,x] = wall_gray
                cnt += 1

        # Build border wall, with extra margin so agent view cannot extend
        # beyond edge of board...
        self.view_distance = view_distance
        board[0:view_distance,:] = wall_gray # top wall
        board[board_size[0]-view_distance-1:,:] = wall_gray # bottom wall
        board[:,0:view_distance] = wall_gray # left wall
        board[:,board_size[0]-view_distance-1:] = wall_gray # right wall

        self.board = np.repeat(np.expand_dims(board, axis=2), 3, axis=2)
        self.board_agents = np.zeros(board_size, dtype=np.uint32)
        self.board_food = np.zeros(board_size, dtype=np.uint8)
        self.image = self.board     # for rendering
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents
        self.agent_locations = {}
        self.agent_colors = {}
        self.agent_speeds = {}

    def add_food(self, frac_pix=0.01, n_pix=None):
        # Allowing food to be placed where it already exists. Just avoiding walls and agents.
        available = self.board[:,:,0] + self.board_agents.astype(np.uint32) == 0  # board plus agents
        ix_open = np.where(available)
        n_open = len(ix_open[0])

        if n_pix is None:
            n_pix = int(frac_pix * n_open)
        n_food_existing = int(np.sum(self.board_food))
        n_pix = n_pix - n_food_existing

        if n_open < n_pix:
            print('\nEnding simulation: Not enough locations left for food growth.')
            sys.exit()

        ix = np.random.permutation(n_open)
        self.board_food[ix_open[0][ix[0:n_pix]], ix_open[1][ix[0:n_pix]]] = 1
        self.image[ix_open[0][ix[0:n_pix]], ix_open[1][ix[0:n_pix]], :] = food_gray
        # self.image = self.image + np.repeat(np.expand_dims(self.board_food*food_gray, axis=2), 3, axis=2)
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents plus food

    def add_agent(self, agent, loc_target=None):
        if loc_target is None:
            loc = self._get_free_location()
        else:
            loc = self._get_free_location(loc_target=loc_target)
        self.agent_locations[agent.id] = loc
        self.agent_colors[agent.id] = agent.color
        self.agent_speeds[agent.id] = 0.5
        self.image[loc[0],loc[1],:] = agent.color
        self.board_agents[loc[0],loc[1]] = agent.id
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents
        self.set_agent_view(agent)

    def _get_free_location(self, loc_target=None):
        ix = np.where(self.unoccupied)
        if len(ix[0])==0:
            print('\nEnding simulation: No free locations remain on board.')
            sys.exit()
        if loc_target is None:
            i = np.random.randint(len(ix[0]))
        else:
            # Find closest free location...
            dist = np.sqrt((ix[0]-loc_target[0])**2 + (ix[1]-loc_target[1])**2)
            i = np.argmin(dist)
        loc = np.array([ix[0][i], ix[1][i]])
        return loc

    def get_agent_location(self, id):
        loc = self.agent_locations[id]
        return loc

    def move_agent(self, agent, loc_new=None):
        id = agent.id
        loc_old = self.agent_locations[id]
        if loc_new is None:
            loc_new = self._get_free_location()
        self.image[loc_old[0], loc_old[1], :] = [0,0,0]
        self.image[loc_new[0], loc_new[1], :] = self.agent_colors[id]
        self.board_agents[loc_old[0], loc_old[1]] = 0
        self.board_agents[loc_new[0], loc_new[1]] = id
        self.agent_locations[id] = loc_new
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents
        self.set_agent_view(agent)

    def remove_agent(self, agent):
        id = agent.id
        loc = self.agent_locations[id]
        self.image[loc[0],loc[1],:] = [0,0,0]
        self.board_agents[loc[0],loc[1]] = 0
        del self.agent_locations[id]
        del self.agent_colors[id]
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents

    def agent_at_location(self, loc):
        id = self.board_agents[loc[0], loc[1]]
        if id:
            return id
        else:
            return None

    def set_agent_view(self, agent):
        loc = self.get_agent_location(agent.id)
        d = agent.direction
        s = (self.view_distance-1)//2
        agent.view = self.image[loc[0] - s + d[0]*(s+1) : loc[0] + s+1 + d[0]*(s+1),
                                      loc[1]-s+d[1]*(s+1) : loc[1]+s+1+d[1]*(s+1),:]
        # Orient the view to the perspective of the agent.
        # Currently this only works for main four cardinal directions.
        if np.array_equal(d, [-1,0]):
            # agent is facing north
            pass
        elif np.array_equal(d, [1,0]):
            # agent is facing south
            agent.view = np.flip(agent.view, axis=[0,1])
        elif np.array_equal(d, [0,-1]):
            # agent is facing west
            agent.view = np.rot90(agent.view, k=3)
        elif np.array_equal(d, [0,1]):
            # agent is facing east
            agent.view = np.rot90(agent.view, k=1)
        return


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
        print(action)

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


## Create new agents or load from file
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


## Kill all agent except one
keys = list(agents.keys())
# speeds = np.asarray([board.agent_speeds[k] for k in keys])
lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
# lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
# lt_bumps = np.asarray([agents[k].lifetime_wall_bumps for k in keys])
# lt_turns = np.asarray([agents[k].lifetime_turns for k in keys])
# lt_waits = np.asarray([agents[k].lifetime_waits for k in keys])
# bump_rate = lt_bumps / (lt_steps+1)
# speeds = lt_forwards / (lt_steps+0.001)
ix_best = np.argmax(lt_steps)
keys = np.delete(keys, ix_best)
for key in keys:
    del agents[key]


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

# print('Step: {}, n_alive: {}, n_total: {}, Dur: {:.0f}, Steps per sec: {:.1f}'.format(i_step, n_agents, n_agents, 0, 0), end='\r')
while True:
    # board.add_food(frac_pix=0.0005)
    # board.add_food(n_pix=n_agents_start)
    dur = time.time() - t_start
    n_alive_agents = len(board.agent_locations)
    # print('Step: {}, speed: {:.2f}, n_alive: {}, n_total: {}, Dur: {:.0f}, Steps per sec: {:.1f}'.format(i_step, hist_speed_mean[-1], n_alive_agents, n_agents, dur, i_step/dur), end='\r')
    keys = list(agents.keys())
    agents_to_kill = np.empty(0, dtype=np.int)
    for key in keys:
        a = agents[key]
        a.lifetime_steps += 1
        a.steps_without_food += 1
        a.fertility_countdown -= 1
        bumped_object = 'none'

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


    # # Kill the submissive mates
    # agents_to_kill = np.unique(agents_to_kill)
    # for mate_id in agents_to_kill:
    #     board.remove_agent(agents[mate_id])
    #     del agents[mate_id]

    # # Determine how many additional agents to kill based on overpopulation
    # n_non_agent = np.sum(np.sum(board.board, axis=2)==0)
    # # n_max = int(round(percent_population_max/100*n_non_agent))
    # n_max = n_agents_start
    # n_alive_agents = len(board.agent_locations)
    # n_kill = max(0, n_alive_agents-n_max)

    # # # Randomly select agents to kill
    # # keys = np.random.choice(list(agents.keys()), n_kill, replace=False)

    # # # Kill the slowest agents
    # # keys = list(agents.keys())
    # # # speeds = np.asarray([board.agent_speeds[k] for k in keys])
    # # lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
    # # lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
    # # speeds = lt_forwards / (lt_steps+0.001)
    # # ix = np.argsort(speeds)
    # # keys = [keys[i] for i in ix[:n_kill]]

    # # # Kill the oldest agents
    # # keys = list(agents.keys())
    # # age = np.asarray([agents[k].lifetime_steps for k in keys])
    # # ix = np.argsort(-age)
    # # keys = [keys[i] for i in ix[:n_kill]]

    # # # Kill the hungriest agents
    # # keys = list(agents.keys())
    # # hunger = np.asarray([agents[k].steps_without_food for k in keys])
    # # ix = np.argsort(-hunger)
    # # keys = [keys[i] for i in ix[:n_kill]]
    # # # ix = np.where(hunger > 100)[0]
    # # # keys = [keys[i] for i in ix]

    # # # Kill the oldest, hungriest agents
    # # keys = list(agents.keys())
    # # hunger = np.asarray([agents[k].steps_without_food for k in keys])
    # # age = np.asarray([agents[k].lifetime_steps for k in keys])
    # # ix = np.argsort(-hunger*age)
    # # keys = [keys[i] for i in ix[:n_kill]]

    # # Kill agents, once every "generation"
    # n_alive_agents = len(board.agent_locations)
    # if n_alive_agents > 2*n_agents_start:
    #     n_kill = n_alive_agents - n_agents_start

    #     # Kill the slowest agents
    #     keys = list(agents.keys())
    #     # speeds = np.asarray([board.agent_speeds[k] for k in keys])
    #     lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
    #     lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
    #     speeds = lt_forwards / (lt_steps+0.001)
    #     ix = np.argsort(speeds)
    #     keys = [keys[i] for i in ix[:n_kill]]
    # else:
    #     keys = []


    # # Do the actual killing
    # for key in keys:
    #     board.remove_agent(agents[key])
    #     del agents[key]


    if b_render and i_step%1==0:
    # if b_render and i_step>4000:
        plt.figure(1)
        plt.clf()
        plt.imshow(board.image)
        n_alive_agents = len(board.agent_locations)
        plt.title('n_agents = %d' % (n_alive_agents))
        # plt.pause(t_frame)
        plt.waitforbuttonpress()

        # # Get more agent data for plotting
        # keys = list(agents.keys())
        # # speeds = np.asarray([board.agent_speeds[k] for k in keys])
        # lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
        # lt_bumps = np.asarray([agents[k].lifetime_wall_bumps for k in keys])
        # lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
        # lt_turns = np.asarray([agents[k].lifetime_turns for k in keys])
        # lt_waits = np.asarray([agents[k].lifetime_waits for k in keys])
        # bump_rate = lt_bumps / (lt_steps+1)
        # speeds = lt_forwards / (lt_steps+0.001)
        # hunger = np.asarray([agents[k].steps_without_food for k in keys])

        # speed_mean = np.mean(speeds)
        # speed_median = np.median(speeds)
        # hist_step = np.append(hist_step, i_step)
        # hist_speed_mean = np.append(hist_speed_mean, np.mean(speeds))
        # hist_speed_median = np.append(hist_speed_median, np.median(speeds))
        # hist_speed_std = np.append(hist_speed_std, np.std(speeds))
        # hist_speed_max = np.append(hist_speed_max, np.max(speeds))
        # hist_bumprate_mean = np.append(hist_bumprate_mean, np.mean(bump_rate))
        # hist_age_median = np.append(hist_age_median, np.median(lt_steps))

        # # ix = np.where(lt_steps > 30)[0]
        # # if ix.size > 0:
        # #     hist_speed_quartile = np.concatenate((hist_speed_quartile, np.expand_dims(np.quantile(speeds[ix], [0.25, 0.5, 1.0]), axis=1)), axis=1)
        # # else:
        # #     hist_speed_quartile = np.concatenate((hist_speed_quartile, np.full((3,1), np.nan)), axis=1)

        # ## Save the agents if the population is better than all previous ones
        # if speed_median > best_speed:
        #     best_speed = speed_median
        #     with open('agents.pkl', 'wb') as output:
        #         keys = list(agents.keys())
        #         for key in keys:
        #             pickle.dump(agents[key], output, pickle.HIGHEST_PROTOCOL)


        # plt.figure(2)
        # plt.subplot(3,1,1)
        # plt.cla()
        # # plt.hist(speeds, bins=np.arange(0,1,0.05))
        # # plt.title('mean speed = %f' % (speed_mean))
        # plt.hist(lt_steps, bins='auto')
        # plt.title('age (lifetime_steps)')

        # plt.subplot(3,1,2)
        # plt.cla()
        # # plt.plot(hist_step, hist_bumprate_mean)
        # plt.plot(hist_step, hist_speed_median)
        # # plt.errorbar(hist_step, hist_speed_mean, hist_speed_std)
        # # plt.errorbar(hist_step, hist_speed_quartile[1,:], hist_speed_quartile[[0,2],:])
        # # plt.plot(np.full(speeds.shape, i_step), speeds, 'bo', mfc='none')
        # plt.title('Median speed')
        # ax = plt.axis()
        # plt.axis(list(ax[0:2]) + [0,1])
        # plt.grid(True)

        # plt.subplot(3,1,3)
        # plt.cla()
        # # plt.plot(hist_step, hist_bumprate_mean)
        # plt.plot(hist_step, hist_age_median)
        # # plt.errorbar(hist_step, hist_speed_mean, hist_speed_std)
        # plt.title('Median age')
        # plt.grid(True)
        # plt.pause(t_frame)

        # # if i_step >= 2000:
        # #     pdb.set_trace()

    i_step += 1

t_total = time.time() - t_start
print('Total time = %f seconds, time/step=%f' % (t_total, t_total/n_steps))

if b_render:
    plt.imshow(board.image)
