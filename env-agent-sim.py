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

directions_possible = np.asarray([[-1,0], [1,0], [0,-1], [0,1]])

class Board():
    def __init__(self, board_size=[64,64], f_obstacles=0.1, view_distance=5):
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
                # Pick an exisiting random point, and add a point at a distance
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
        board[0:view_distance-1,:] = obstacle_gray # top wall
        board[board_size[0]-view_distance:,:] = obstacle_gray # bottom wall
        board[:,0:view_distance-1] = obstacle_gray # left wall
        board[:,board_size[0]-view_distance:] = obstacle_gray # right wall

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
        self.agent_speeds[agent.id] = 1.0
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

    def move_agent(self, agent, loc_new):
        id = agent.id
        loc_old = self.agent_locations[id]
        self.image[loc_old[0], loc_old[1], :] = [0,0,0]
        self.image[loc_new[0], loc_new[1], :] = self.agent_colors[id]
        self.board_agents[loc_old[0], loc_old[1]] = 0
        self.board_agents[loc_new[0], loc_new[1]] = id
        self.agent_locations[id] = loc_new
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents
        self.set_agent_view(agent)

    def remove_agent(self):
        pass

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
        agent.agent_view = self.image[loc[0]-s+d[0]*(s+1):loc[0]+s+1+d[0]*(s+1),
                                      loc[1]-s+d[1]*(s+1):loc[1]+s+1+d[1]*(s+1),:]
        return


class Agent:
    def __init__(self, id, parents=None):
        assert id, 'Agent ID must be greater than 0.'
        self.id = id
        self.parents = parents
        self.mates = []
        self.direction = self._get_random_direction()
        self.view = None
        self.bumped_object = 'none'
        self.color = np.random.randint(0,256,(3))
        self.color[np.random.randint(3)] = 0 # use saturation colors only
        self.color = (255*self.color/np.sqrt(np.sum(self.color**2))).astype(np.uint8)
        self.timeout = 0

    def step(self):
        # print('Agent %d, bumped_object=%s' % (self.id, self.bumped_object))
        if self.bumped_object=='wall':
            self.direction = self._get_random_direction()
            self.timeout = 5
        if self.bumped_object=='agent':
            self.direction = self._get_random_direction()
            self.timeout -= 1
        else:
            self.timeout -= 1
        return self.direction, self.timeout

    def _get_random_direction(self):
        n = np.random.randint(len(directions_possible))
        return directions_possible[n]


n_agents = 20
b_render = True
t_frame = 0.001
n_steps = 100

agents = OrderedDict()
for i in range(1, n_agents+1):
    agents[i] = Agent(i)

board_size = [64,64]   # height, width
# board_size = [32,32]   # height, width
board = Board(board_size=board_size, f_obstacles=0.1)
for key, a in agents.items():
    board.add_agent(a)

if b_render:
    plt.figure(1)
    plt.imshow(board.image)
    plt.pause(t_frame)

i_step = 0
t_start = time.time()
cnt_mate_collisions = 0
# while i_step < 100:
while True:
    t_step = time.time()
    keys = list(agents.keys())
    for key in keys:
        a = agents[key]
        bumped_object = 'none'
        direction, timeout = a.step()
        # print('Agent %d: dir=%d,%d  timeout=%d\n' % (a.id, direction[0], direction[1], timeout))

        speed_avg_factor = 0.95

        if timeout <=0:
            loc_new = board.get_agent_location(a.id) + direction

            if not board.unoccupied[loc_new[0], loc_new[1]]:
                # Bumped into wall or other agent
                board.agent_speeds[a.id] = speed_avg_factor*board.agent_speeds[a.id] + 0.0
                mate_id = board.agent_at_location(loc_new)

                if mate_id is None:
                    # Bumped into wall
                    bumped_object = 'wall'
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
                        n_agents += 1
                        agents[n_agents] = Agent(n_agents, parents=(a.id, mate_id))

                        board.add_agent(agents[n_agents], loc_target=board.get_agent_location(a.id))
                        # Add IDs to mate lists, for both parents
                        a.mates.append(mate_id)
                        agents[mate_id].mates.append(a.id)

            else:
                board.move_agent(a, loc_new)
                board.agent_speeds[a.id] = speed_avg_factor*board.agent_speeds[a.id] + (1-speed_avg_factor)
        else:
            # Still in timeout
            board.agent_speeds[a.id] = speed_avg_factor*board.agent_speeds[a.id] + 0.0

        a.bumped_object = bumped_object


    # Determine how many agents to kill
    percent_max = 20
    n_non_agent = np.sum(np.sum(board.board, axis=2)==0)
    n_max = int(round(percent_max/100*n_non_agent))
    n_alive_agents = len(board.agent_locations)
    n_kill = max(0, n_alive_agents-n_max)

    ## Randomly select agents to kill
    # keys = np.random.choice(list(agents.keys()), n_kill, replace=False)

    # Kill the slowest agents
    keys = list(agents.keys())
    speeds = np.asarray([board.agent_speeds[k] for k in keys])
    ix = np.argsort(speeds)
    keys = [keys[i] for i in ix[:n_kill]]

    for key in keys:
        id = agents[key].id
        loc = board.agent_locations[id]
        board.image[loc[0],loc[1],:] = [0,0,0]
        board.board_agents[loc[0],loc[1]] = 0
        del board.agent_locations[id]
        del board.agent_colors[id]
        board.unoccupied = np.sum(board.image, axis=2) == 0  # board plus agents
        del agents[key]

    if b_render:
        plt.figure(1)
        plt.clf()
        plt.imshow(board.image)
        n_alive_agents = len(board.agent_locations)
        plt.title('n_agents = %d' % (n_alive_agents))
        plt.pause(t_frame)

    i_step += 1

t_total = time.time() - t_start
print('Total time = %f seconds, time/step=%f' % (t_total, t_total/n_steps))

if b_render:
    plt.imshow(board.image)
