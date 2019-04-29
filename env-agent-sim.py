# Multi-agent simulation
#
# Try to build simple code that has:
# 1. Environment
# 2. Multiple agents, running as separate threads
# 3. Interaction between environment and agents
# 4. Interaction between agents, independent of environment?

### NEVERMIND. Try turn-based simulation first, rather than threaded.

# TODO: Do not create offspring if agents parents have mated
# before or are parent/child.

import numpy as np
import time
import sys
import pdb
import matplotlib.pyplot as plt
plt.ion()


class Board():
    def __init__(self, board_size=[64,64], f_obstacles=0.1):
        board = np.zeros(board_size, dtype=np.uint8)
        n = np.prod(board_size)
        p = np.random.permutation(n)
        p = p[:int(n*f_obstacles)]
        ix = np.unravel_index(p, board_size)
        obstacle_gray = 128
        board[ix] = obstacle_gray
        board[0,:] = obstacle_gray # top wall
        board[board_size[0]-1,:] = obstacle_gray # bottom wall
        board[:,0] = obstacle_gray # left wall
        board[:,board_size[0]-1] = obstacle_gray # right wall
        self.board = np.repeat(np.expand_dims(board, axis=2), 3, axis=2)
        self.board_agents = np.zeros(board_size, dtype=np.uint32)
        self.image = self.board     # for rendering
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents
        self.agent_locations = {}
        self.agent_colors = {}

    def add_agent(self, agent):
        loc = self._get_free_location()
        self.agent_locations[agent.id] = loc
        self.agent_colors[agent.id] = agent.color
        self.image[loc[0],loc[1],:] = agent.color
        self.board_agents[loc[0],loc[1]] = agent.id
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents

    def _get_free_location(self):
        ix = np.where(self.unoccupied)
        if len(ix[0])==0:
            print('\nEnding simulation: No free locations remain on board.')
            sys.exit()
        i = np.random.randint(len(ix[0]))
        loc = np.array([ix[0][i], ix[1][i]])
        return loc

    def get_agent_location(self, id):
        loc = self.agent_locations[id]
        return loc

    def move_agent(self, id, loc_new):
        loc_old = self.agent_locations[id]
        self.image[loc_old[0], loc_old[1], :] = [0,0,0]
        self.image[loc_new[0], loc_new[1], :] = self.agent_colors[id]
        self.board_agents[loc_old[0], loc_old[1]] = 0
        self.board_agents[loc_new[0], loc_new[1]] = id
        self.agent_locations[id] = loc_new
        self.unoccupied = np.sum(self.image, axis=2) == 0  # board plus agents

    def remove_agent(self):
        pass

    def agent_at_location(self, loc):
        id = self.board_agents[loc[0], loc[1]]
        if id:
            return id
        else:
            return None


class Agent:
    def __init__(self, id, parents=None):
        assert id, 'Agent ID must be greater than 0.'
        self.id = id
        self.parents = parents
        self.mates = []
        self.direction = self._get_random_direction()
        self.view = None
        self.bumped_object = False
        self.color = np.random.randint(0,256,(3))
        self.color[np.random.randint(3)] = 0 # use saturation colors only
        self.color = (255*self.color/np.sqrt(np.sum(self.color**2))).astype(np.uint8)

    def step(self):
        if self.bumped_object==True:
            self.direction = self._get_random_direction()
        return self.direction

    def _get_random_direction(self):
        moving = False
        while not moving:
            direction = np.random.choice([-1,0,1],(2))
            if not (direction[0]==0 and direction[1]==0):
                moving = True
        return direction

    def state_update(self, agent_view, bumped_object=False):
        self.view = agent_view
        self.bumped_object = bumped_object


n_agents = 20
b_render = True
t_frame = 0.01
n_steps = 100

agents = [Agent(i) for i in range(1, n_agents+1)]
board_size = [64,64]   # height, width
board = Board(board_size=board_size, f_obstacles=0.1)
for a in agents:
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
    for i_a, a in enumerate(agents):
        direction = a.step()
        loc_new = board.get_agent_location(a.id) + direction

        if not board.unoccupied[loc_new[0], loc_new[1]]:
            bumped_object = True
            mate_id = board.agent_at_location(loc_new)
            if mate_id is not None:
                b_create_offspring = True

                ## Don't create offspring if pair is parent/child
                # (Why not!? This is just a human taboo!)
                if a.parents is not None and mate_id in a.parents:
                    b_create_offspring = False
                if agents[mate_id-1].parents is not None and a.id in agents[mate_id-1].parents:
                    b_create_offspring = False

                ## Don't mate if this pair has mated before.
                # This is to prevent agents from just idling in a local area and mating
                # regularly with the same partners. Goal is to promote navigation by agents.
                if mate_id in a.mates:
                    b_create_offspring = False

                if b_create_offspring:
                    # TODO?: May want to create babies after all agents have had a step,
                    # so this enumerate loop doesn't grow while the looping is occurring.

                    # Create offspring
                    n_agents += 1
                    agents.append(Agent(n_agents, parents=(a.id, mate_id)))
                    board.add_agent(agents[-1])
                    # Add IDs to mate lists, for both parents
                    a.mates.append(mate_id)
                    agents[mate_id].mates.append(a.id)

        else:
            bumped_object = False
            board.move_agent(a.id, loc_new)

        a.state_update(None, bumped_object=bumped_object)

    if b_render:
        plt.figure(1)
        plt.clf()
        plt.imshow(board.image)
        plt.title('n_agents = %d' % (n_agents))
        plt.pause(t_frame)
    i_step += 1

t_total = time.time() - t_start
print('Total time = %f seconds, time/step=%f' % (t_total, t_total/n_steps))

if b_render:
    plt.imshow(board.image)
