#####
# NEEDS TO BE DEBUGGED. IS NOT LEARNING FOR SOME REASON ##




# gridworld setup from
#MJeremy2017
 #https://github.com/MJeremy2017/Reinforcement-Learning-Implementation/blob/master/GridWorld/gridWorld.py

import numpy as np
import math
from random import seed
from random import random
from random import randint
import csv



# global variables
BOARD_ROWS = 7
BOARD_COLS = 8
CHASE_START = (3, 6)
FLEE_START = (2, 4)
CHASE_IDX = 0   #is actually the fleer in this one. the single 'optimising' agent
FLEE_IDX=1      #the randomly moving thing
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP,DOWN,LEFT,RIGHT]
DETERMINISTIC = True
APPROACH="sarsa"

ALPHA=0.25
GAMMA=0.9
LAMBDA=0.5

class State:
    def __init__(self, state=[CHASE_START, FLEE_START]):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[5, 1] = -1
        self.board[5, 2] = -1
        self.board[5, 3] = -1
        self.board[5, 4] = -1
        self.board[5, 5] = -1
        self.board[5, 6] = -1
        self.state = state
        self.epNum=1
        #self.targetState=WIN_STATE
        self.isEnd = False
        self.determine = DETERMINISTIC
       

    def chaseReward(self):
        if self.state[0] == self.state[1]:
            return -1
        else:
            return 0

    def isEndFunc(self):
        if (self.state[CHASE_IDX]==self.state[FLEE_IDX]):
            self.isEnd = True

    def nxtPosition(self, agent_index, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if self.determine:
            if action == UP:
                nxtState = (self.state[agent_index][0] - 1, self.state[agent_index][1])
            elif action == DOWN:
                nxtState = (self.state[agent_index][0] + 1, self.state[agent_index][1])
            elif action == LEFT:
                nxtState = (self.state[agent_index][0], self.state[agent_index][1] - 1)
            else:
                nxtState = (self.state[agent_index][0], self.state[agent_index][1] + 1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS -1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS -1)):
                    if self.board[nxtState[0], nxtState[1]] != -1:
                        return nxtState
            return self.state[agent_index]

    def showBoard(self):
        self.board[self.state[0]] = 1
        self.board[self.state[FLEE_IDX]]=2
        for i in range(0, BOARD_ROWS):
            print('-------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i,j] == 2:
                    token='#'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('---------------------------')
        self.board[self.state[0]] = 0
        self.board[self.state[FLEE_IDX]]=0


# end of gridworld setup code from MJeremy2017


class Agent:
    def __init__(self):
        self.Q = np.zeros([BOARD_ROWS, BOARD_COLS, BOARD_ROWS, BOARD_COLS, len(ACTIONS)])
        self.eTraces = np.zeros([BOARD_ROWS, BOARD_COLS, BOARD_ROWS, BOARD_COLS, len(ACTIONS)])
        self.prvState = [CHASE_START, FLEE_START]
        self.step = 1
        self.epStep = 1
        self.randSteps = 0
        
       
        
    def policy(self, MDP):
        epsilon = 200/MDP.epNum
        #epsilon=1/math.log(self.step+1,1.3)
        #removed for speed
        #print("epsilon: ",epsilon)  
        #random or greedy
        if random() <= epsilon:
            #then choose randomly
            #removed for speed print("Random step!")
            nxtAction = randint(0,len(ACTIONS)-1)
            self.randSteps+=1
        else:    
            #choose greedily
            nxtAction = UP;
            for i in range(1,len(ACTIONS)):
                Q_i= self.Q[MDP.state[CHASE_IDX][0],MDP.state[CHASE_IDX][1],MDP.state[FLEE_IDX][0],MDP.state[FLEE_IDX][1],i]
                Q_nxtAction = self.Q[MDP.state[CHASE_IDX][0],MDP.state[CHASE_IDX][1],MDP.state[FLEE_IDX][0],MDP.state[FLEE_IDX][1],nxtAction]
                if  Q_i > Q_nxtAction:
                    nxtAction = i
                elif self.Q[MDP.state[CHASE_IDX][0],MDP.state[CHASE_IDX][1],MDP.state[FLEE_IDX][0],MDP.state[FLEE_IDX][1],i] == self.Q[MDP.state[CHASE_IDX][0],MDP.state[CHASE_IDX][1],MDP.state[FLEE_IDX][0],MDP.state[FLEE_IDX][1],nxtAction]:
                    if randint(0,1):
                        nxtAction = i
        return nxtAction

        
    def update(self, action, reward, MDP):
        #Sarsa
        #chase prev position
        cPrv=self.prvState[CHASE_IDX]
        #flee prev position
        fPrv=self.prvState[FLEE_IDX]
        #chase current position
        cPos=MDP.state[CHASE_IDX]
        fPos=MDP.state[FLEE_IDX]
        #debug
        
        oldQ = self.Q[cPrv[0],cPrv[1],fPrv[0],fPrv[1],action]
        delta=reward+GAMMA*(self.Q[cPos[0], cPos[1],fPos[0],fPos[1],self.policy(MDP)])-oldQ
        self.sarsa(action, reward, MDP.state, oldQ, delta, cPrv, fPrv)
     
       
    def sarsa(self, action, reward, state, oldQ, delta, cPrv, fPrv):
        newQ   = oldQ + ALPHA*delta
        self.Q[cPrv[0],cPrv[1],fPrv[0],fPrv[1],action] = newQ
        
    
    
    
    def sarsaL(self, action, reward, state, delta, cPrv, fPrv):
        self.eTraces[cPrv[0],cPrv[1],fPrv[0],fPrv[1],action] += 1
        #print("eTrace of ",self.prvState,": ",self.eTraces[self.prvState[0],self.prvState[1],action])
        self.update_eTraces(len(ACTIONS))
       # print("Old Q for ",self.prvState,": ",self.Q[self.prvState[0],self.prvState[1]])
        self.updateQL(len(ACTIONS), delta)
       # print("delta: ",delta)
        #print("New Q for ",self.prvState,": ",self.Q[self.prvState[0],self.prvState[1]])   

    def update_eTraces(self, num_actions):
        #updates eligibility traces
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(BOARD_ROWS):
                    for l in range(BOARD_COLS):
                        for a in range(num_actions):
                                self.eTraces[i,j,k,l,a] *= LAMBDA*GAMMA
    
    def updateQL(self, num_actions, delta):
        #updates Q value for all elibible s,a pairs
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(BOARD_ROWS):
                    for l in range(BOARD_COLS):
                        for a in range(num_actions):
                            self.Q[i,j,k,l,a] += ALPHA*delta*self.eTraces[i,j,k,l,a]
    
def play(MDP, agent):
    record = []
    val = 1
    while val != 0:
        val = int(input("Go to episode: "))
        if val == 0:
            write_csv(record, APPROACH)
        while MDP.epNum <= val:
            print("Reward: ",MDP.chaseReward())
            MDP.state=[CHASE_START, FLEE_START]
            MDP.isEnd = False
            agent.epStep=1
            agent.randSteps = 0
            agent.eTraces = np.zeros([BOARD_ROWS, BOARD_COLS, BOARD_ROWS, BOARD_COLS, len(ACTIONS)])
            if MDP.epNum==val:
                episode(MDP,agent, True)
            else:
                episode(MDP,agent, False)
            reward=MDP.chaseReward()    
        
            print("This was episode #",MDP.epNum)
            print("Ep Steps: ", agent.epStep)   
            print("Rand Steps: ",agent.randSteps)
            print("Average: ",(agent.step/MDP.epNum))
            record.append([MDP.epNum, reward, agent.epStep])
            MDP.epNum+=1
        
def write_csv(record_list, approach):
    with open('rand-chase-'+approach+'-performance.csv', mode='w') as file:
        data_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(['Episode Number', 'reward','steps in episode'])
        for i in range(len(record_list)):
            data_writer.writerow([record_list[i][0], record_list[i][1], record_list[i][2]])
            
def episode(MDP, agent, graphical):
    #graphical turns board printing on or off (boolean)
    
    #removed for speed MDP.showBoard()
    MDP.isEndFunc()
    while not(MDP.isEnd):
        
        action = agent.policy(MDP)
        agent.prvState[CHASE_IDX]=MDP.state[CHASE_IDX]
        agent.prvState[FLEE_IDX]=MDP.state[FLEE_IDX] 
        
        
        MDP.state[CHASE_IDX]=MDP.nxtPosition(CHASE_IDX,action)
        MDP.state[FLEE_IDX]=MDP.nxtPosition(FLEE_IDX,randint(0,3))
        MDP.isEndFunc()
        reward = MDP.chaseReward()
        agent.step+=1
        agent.epStep+=1
        if graphical:
            MDP.showBoard()
        
        reward = MDP.chaseReward()
        # debug
        
        
        agent.update(action, reward, MDP)
        MDP.isEndFunc()
        

seed(3)
test_state=State()
test_state.showBoard()
test_agent=Agent()
print(test_agent.Q)
play(test_state, test_agent)



    