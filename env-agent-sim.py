# Multi-agent simulation
#
# Try to build simple code that has:
# 1. Environment
# 2. Multiple agents, running as separate threads
# 3. Interaction between environment and agents
# 4. Interaction between agents, independent of environment?

### NEVERMIND. Try turn-based simulation first, rather than threaded.

## TODO
# 1. Buid board with more distinct areas
# 2. Build agents with genetric programming. Maybe try these:
#       https://deap.readthedocs.io/en/0.7-0/examples/symbreg.html
#       https://github.com/trevorstephens/gplearn

import numpy as np
import time
import sys
import random
from collections import OrderedDict 
import pdb
import matplotlib.pyplot as plt
plt.ion()

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = random.randint(0, 1e9)
print('seed = %d' % seed)
np.random.seed(seed)
random.seed(seed)


# Order directions possible in counter-clockwise manner
directions_possible = np.asarray([[-1,0], [0,-1], [1,0], [0,1]])

timeout_wall = 5
timeout_mate = 0

view_distance = 5

n_agents = 40
b_render = True
t_frame = 0.001
n_steps = 100

percent_population_max = 5  # maximum are of open board space that can be occupied by agents



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x + 1e-10) / (e_x.sum() + 1e-10)


class Board():
    def __init__(self, board_size=[128,128], f_obstacles=0.05, view_distance=5):
        assert view_distance%2==1, 'Agent view distance must be odd-valued.'
        board = np.zeros(board_size, dtype=np.uint8)
        board_type = 'random'   # random, grouped
        obstacle_gray = 128
    
        if board_type=='random':
            n = np.prod(board_size)
            p = np.random.permutation(n)
            p = p[:int(n*f_obstacles)]
            ix = np.unravel_index(p, board_size)
            board[ix] = obstacle_gray
        
        elif board_type=='grouped':
            # Start with a few randomly located parts
            n = np.prod(board_size)
            p = np.random.permutation(n)
            p = p[:10]
            ix = np.unravel_index(p, board_size)
            board[ix] = obstacle_gray

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
                board[y,x] = obstacle_gray
                cnt += 1

        # Build border wall, with extra margin so agent view cannot extend
        # beyond edge of board...
        self.view_distance = view_distance
        board[0:view_distance,:] = obstacle_gray # top wall
        board[board_size[0]-view_distance-1:,:] = obstacle_gray # bottom wall
        board[:,0:view_distance] = obstacle_gray # left wall
        board[:,board_size[0]-view_distance-1:] = obstacle_gray # right wall

        self.board = np.repeat(np.expand_dims(board, axis=2), 3, axis=2)
        self.board_agents = np.zeros(board_size, dtype=np.uint32)
        self.image = self.board     # for rendering
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents
        self.agent_locations = {}
        self.agent_colors = {}
        self.agent_speeds = {}

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
        if parents:
            self.parents = [parents[0].id, parents[1].id]
        else:
            self.parents = None

        if not parents:
            vd = view_distance
            self.weights_in2hid = np.random.randn(3*vd, 3*vd**2 + 2)
            self.weights_hid2hid = np.random.randn(3*vd, 3*vd)
            self.bias_hid = np.random.randn(3*vd)
            self.act_hid = np.zeros(3*vd)
            self.weights_hid2out = np.random.randn(3, 3*vd)
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
            ix = np.where(np.random.rand(*self.weights_in2hid.shape) < p)
            self.weights_in2hid[ix] = np.random.randn(len(ix[0]))
            ix = np.where(np.random.rand(*self.weights_hid2hid.shape) < p)
            self.weights_hid2hid[ix] = np.random.randn(len(ix[0]))
            ix = np.where(np.random.rand(*self.bias_hid.shape) < p)
            self.bias_hid[ix] = np.random.randn(len(ix[0]))
            ix = np.where(np.random.rand(*self.weights_hid2out.shape) < p)
            self.weights_hid2out[ix] = np.random.randn(len(ix[0]))
            ix = np.where(np.random.rand(*self.bias_out.shape) < p)
            self.bias_out[ix] = np.random.randn(len(ix[0]))


    def step(self):
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
        out = softmax(out)
        action = np.random.choice(('forward', 'turn_left', 'turn_right'), 1, p=out)

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



agents = OrderedDict()
for i in range(1, n_agents+1):
    agents[i] = Agent(i, view_distance = view_distance)

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

hist_step = np.zeros(0)
hist_speed_med = np.zeros(0)
hist_speed_mean = np.zeros(0)
hist_speed_std = np.zeros(0)
hist_speed_max = np.zeros(0)
hist_bumprate_mean = np.zeros(0)

# while i_step < n_steps:
print('Step: {}, n_alive: {}, n_total: {}, Dur: {:.0f}, Steps per sec: {:.1f}'.format(i_step, n_agents, n_agents, 0, 0), end='\r')
while True:
    dur = time.time() - t_start
    n_alive_agents = len(board.agent_locations)
    print('Step: {}, n_alive: {}, n_total: {}, Dur: {:.0f}, Steps per sec: {:.1f}'.format(i_step, n_alive_agents, n_agents, dur, i_step/dur), end='\r')
    keys = list(agents.keys())
    agents_to_kill = np.empty(0, dtype=np.int)
    for key in keys:
        a = agents[key]
        a.lifetime_steps += 1
        bumped_object = 'none'
        direction, turn, timeout = a.step()

        speed_avg_factor = 0.95

        if timeout <=0 and not turn and not np.any(key==agents_to_kill):
            loc_new = board.get_agent_location(a.id) + direction

            if not board.unoccupied[loc_new[0], loc_new[1]]:
                # Bumped into wall or other agent
                board.agent_speeds[a.id] = speed_avg_factor*board.agent_speeds[a.id] + 0.0
                mate_id = board.agent_at_location(loc_new)

                if mate_id is None:
                    # Bumped into wall
                    bumped_object = 'wall'
                    a.lifetime_wall_bumps += 1
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

                    ## Don't mate if this pair has mated before.
                    # This is to prevent agents from just idling in a local area and mating
                    # regularly with the same partners. Goal is to promote navigation by agents.
                    if mate_id in a.mates:
                        b_create_offspring = False

                    if b_create_offspring:
                        # Create offspring
                        # n_offspring = random.choice([1, 2])
                        n_offspring = np.random.choice([1, 2], 1, p=[0.8, 0.2])[0]
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

                        # Add submissive mate to list of agents to kill after this step
                        agents_to_kill = np.append(agents_to_kill, mate_id)

                        # Give agent credit for a forward move even though agent isn't actually moved
                        # due to difficulty with current code configuration
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
    n_max = int(round(percent_population_max/100*n_non_agent))
    n_alive_agents = len(board.agent_locations)
    n_kill = max(0, n_alive_agents-n_max)

    # # Randomly select agents to kill
    # keys = np.random.choice(list(agents.keys()), n_kill, replace=False)

    # Kill the slowest agents
    keys = list(agents.keys())
    # speeds = np.asarray([board.agent_speeds[k] for k in keys])
    lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
    lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
    speeds = lt_forwards / (lt_steps+0.001)
    ix = np.argsort(speeds)
    keys = [keys[i] for i in ix[:n_kill]]

    # Kill the oldest agents
    keys = list(agents.keys())
    age = np.asarray([agents[k].lifetime_steps for k in keys])
    ix = np.argsort(-age)
    keys = [keys[i] for i in ix[:n_kill]]

    # loc_dead_agents = [[], []]
    for key in keys:
        board.remove_agent(agents[key])
        del agents[key]

        # id = agents[key].id
        # loc = board.agent_locations[id]
    
        # # # Store location so we can briefly show dying agent as white pixel
        # # loc_dead_agents[0].append(loc[0])
        # # loc_dead_agents[1].append(loc[1])

        # # Then remove dead agent
        # board.image[loc[0],loc[1],:] = [0,0,0]
        # board.board_agents[loc[0],loc[1]] = 0
        # del board.agent_locations[id]
        # del board.agent_colors[id]
        # board.unoccupied = np.sum(board.image, axis=2) == 0  # board plus agents
        # del agents[key]

    # # Briefly show dying agent as white pixel
    # if b_render and len(loc_dead_agents[0])>0:
    #     board.image[loc_dead_agents[0],loc_dead_agents[1],:] = 255
    #     plt.figure(1)
    #     plt.clf()
    #     plt.imshow(board.image)
    #     n_alive_agents = len(board.agent_locations)
    #     plt.title('n_agents = %d' % (n_alive_agents))
    #     plt.pause(t_frame)

    # Remove dead agent pixels from board
    # board.image[loc_dead_agents[0],loc_dead_agents[1],:] = 0

    if b_render and i_step%50==0:
        plt.figure(1)
        plt.clf()
        plt.imshow(board.image)
        n_alive_agents = len(board.agent_locations)
        plt.title('n_agents = %d' % (n_alive_agents))
        plt.pause(t_frame)


        plt.figure(2)
        plt.clf()
        plt.subplot(2,1,1)

        keys = list(agents.keys())
        # speeds = np.asarray([board.agent_speeds[k] for k in keys])
        lt_steps = np.asarray([agents[k].lifetime_steps for k in keys])
        lt_bumps = np.asarray([agents[k].lifetime_wall_bumps for k in keys])
        lt_forwards = np.asarray([agents[k].lifetime_forwards for k in keys])
        bump_rate = lt_bumps / (lt_steps+1)
        speeds = lt_forwards / (lt_steps+0.001)

        speed_med = np.median(speeds)
        hist_step = np.append(hist_step, i_step)
        hist_speed_med = np.append(hist_speed_med, speed_med)
        hist_speed_mean = np.append(hist_speed_mean, np.mean(speeds))
        hist_speed_std = np.append(hist_speed_std, np.std(speeds))
        hist_speed_max = np.append(hist_speed_max, np.max(speeds))
        hist_bumprate_mean = np.append(hist_bumprate_mean, np.mean(bump_rate))

        plt.hist(speeds, bins=np.arange(0,1,0.05))
        plt.title('median speed = %f' % (speed_med))

        plt.subplot(2,1,2)
        # plt.plot(hist_step, hist_bumprate_mean)
        plt.plot(hist_step, hist_speed_med)
        plt.plot(hist_step, hist_speed_max)
        plt.errorbar(hist_step, hist_speed_mean, hist_speed_std)
        plt.grid(True)
        plt.pause(t_frame)

    i_step += 1

t_total = time.time() - t_start
print('Total time = %f seconds, time/step=%f' % (t_total, t_total/n_steps))

if b_render:
    plt.imshow(board.image)
