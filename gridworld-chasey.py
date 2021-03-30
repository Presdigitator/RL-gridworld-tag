#find and replace CHASE_IDX etc

# gridworld setup from
#MJeremy2017
 #https://github.com/MJeremy2017/Reinforcement-Learning-Implementation/blob/master/GridWorld/gridWorld.py

import numpy as np
import math
from random import seed
from random import random
from random import randint
import copy

import csv

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc



# global variables
BOARD_ROWS = 7
BOARD_COLS = 8
CHASE_START = (1,6)
FLEE_START = (6, 2)
HONEY=(9,9)
CHASE_IDX = 0
FLEE_IDX=1
NUM_AGENTS=2

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP,DOWN,LEFT,RIGHT]
BARRIER=-1
DETERMINISTIC = True
APPROACH="sarsa"

ALPHA=0.5
GAMMA=0.3
LAMBDA=0.5

#video
FPS=24

class State:
    def __init__(self, state=[CHASE_START, FLEE_START]):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        #blocks:
        
        self.board[1,2] = -1
        self.board[1,3] = -1
        self.board[1,4] = -1
        self.board[1,5]=-1
        self.board[1, 7] = -1
        
        
        
        self.state = state
        self.epNum=1
        #self.targetState=WIN_STATE
        self.isEnd = False
        self.determine = DETERMINISTIC
       

    def chaseReward(self):
        if self.state[CHASE_IDX] == self.state[FLEE_IDX]:
            return 1
        #these two elif set up a 'thief' scenario
        #elif self.state[FLEE_IDX] == HONEY:
            #return -0.02
        else:
            return 0
            
    def fleeReward(self):
        if self.state[CHASE_IDX] == self.state[FLEE_IDX]:
            return -1
        #this  elif set up a 'thief' scenario
        #elif self.state[FLEE_IDX] == HONEY:
            #return 0.1
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
                    if self.board[nxtState] != -1:
                        return nxtState
            return self.state[agent_index]

    def showBoard(self): 
        self.board[self.state[CHASE_IDX]] = 1
        self.board[self.state[FLEE_IDX]]=2
        for i in range(0, BOARD_ROWS):
            print('-------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '>'
                if self.board[i,j] == 2:
                    token='O'
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
        self.prvState = [(0,0), (0,0)] #[own position, other's position]
        self.step = 1
        self.epStep = 1
        self.randSteps = 0
        self.action=-1
        
        #index for this agent ('self index')
        self.sIDX=-1
        #index for other agent
        self.oIDX=-1
        
       
        
    def policy(self, MDP, performance):
        epsilon = 1000/MDP.epNum
        #epsilon = 30/(math.log(MDP.epNum,1.1)+1)
   
        if performance:
            epsilon = 0.05
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
            #print("setting action to UP.")
            
            nxtAction = UP;
            for i in range(1,len(ACTIONS)):
                Q_i= self.Q[MDP.state[self.sIDX][0],MDP.state[self.sIDX][1],MDP.state[self.oIDX][0],MDP.state[self.oIDX][1],i]
                Q_nxtAction = self.Q[MDP.state[self.sIDX][0],MDP.state[self.sIDX][1],MDP.state[self.oIDX][0],MDP.state[self.oIDX][1],nxtAction]
                if  Q_i > Q_nxtAction:
                    #debug
                    ("changing action to ",ACTIONS[i])
                    
                    nxtAction = i
                elif Q_i == Q_nxtAction:
                    if randint(0,1):
                        nxtAction = i
        return nxtAction

        
    def update(self, reward, MDP):
        #Sarsa, but kind of Q learning?
        
       
        #self prev position
        sPrv=self.prvState[0]
        #other agent's prev position
        oPrv=self.prvState[1]
        #self current position
        sPos=MDP.state[self.sIDX]
        #other agent current position
        oPos=MDP.state[self.oIDX]

        oldQ = self.Q[sPrv[0],sPrv[1],oPrv[0],oPrv[1], self.action]
       
        
        
        
        delta=reward+GAMMA*(self.Q[sPos[0], sPos[1],oPos[0],oPos[1],self.policy(MDP, False)])-oldQ
        self.sarsa(reward, MDP.state, oldQ, delta, sPrv, oPrv)
     
       
    def sarsa(self, reward, state, oldQ, delta, sPrv, oPrv):
        newQ   = oldQ + ALPHA*delta
        #debug
        #print("New Q=", newQ)
        
        self.Q[sPrv[0],sPrv[1],oPrv[0],oPrv[1],self.action] = newQ
        
    
    
    
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
    
    def turn(self, MDP, greedy):
        self.prvState[0]=MDP.state[self.sIDX]
        self.prvState[1]=MDP.state[self.oIDX]
        self.action = self.policy(MDP, greedy)
        return MDP.nxtPosition(self.sIDX,self.action)
    
# End of agent #
    

def play(MDP, chase_agent, flee_agent):
    record = []
    val = 1
    while val != 0:
        val = int(input("Go to episode: "))
        if val == 0:
            write_csv(record, APPROACH)
        while MDP.epNum <= val:
            print("Reward: ",MDP.chaseReward())
            
            #reset:
            MDP.state=[CHASE_START, FLEE_START]
            MDP.isEnd = False
            chase_agent.epStep=1
            flee_agent.epStep=1
            chase_agent.randSteps = 0
            flee_agent.randSteps=0
            chase_agent.eTraces = np.zeros([BOARD_ROWS, BOARD_COLS, BOARD_ROWS, BOARD_COLS, len(ACTIONS)])
            flee_agent.eTraces = np.zeros([BOARD_ROWS, BOARD_COLS, BOARD_ROWS, BOARD_COLS, len(ACTIONS)])
           
            if MDP.epNum==val:
                states=episode(MDP,chase_agent, flee_agent, True, True)
                print("Ep Steps: ", chase_agent.epStep)
                response =input("Write to video? y/n\n")
                if response == 'y':
                    print("Writing video...")
                    write_video(states, MDP.board)
                    print("Done.")
                else:
                    print("Skipping video.")
                
            else:
                episode(MDP,chase_agent, flee_agent, False, False)
            reward=MDP.chaseReward()    
        
            print("\n--This was episode #",MDP.epNum,"--")
            print("Ep Steps: ", chase_agent.epStep)   
            print("Rand Steps chase: ",chase_agent.randSteps)
            print("Rand Steps flee: ",flee_agent.randSteps)
            print("Average: ",(chase_agent.step/MDP.epNum))
            record.append([MDP.epNum, reward, chase_agent.epStep])
            MDP.epNum+=1
        
def write_csv(record_list, approach):
    with open('rand-chase-'+approach+'-performance.csv', mode='w') as file:
        data_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(['Episode Number', 'reward','steps in episode'])
        for i in range(len(record_list)):
            data_writer.writerow([record_list[i][0], record_list[i][1], record_list[i][2]])
            
def write_video(states, board):
    #can increase blowup/radius for larger vids/characters and vice versa
    blowUp=100
    radius=30
    height=BOARD_ROWS*blowUp
    width=BOARD_COLS*blowUp
    secsPerState=0.2
    #adjust state coordinates for video size:
    BUStates = blownUp(states, height, blowUp)
   
    
    room=np.full((height, width, 3), 255, dtype=np.uint8)
   
    
    
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r,c] == -1:
                for i in range(r*blowUp, (r+1)*blowUp):
                    for j in range(c*blowUp, (c+1)*blowUp):
                        room[i,j]=0
            #paint the honeypot            
            elif (r,c)==HONEY:
                print("honey")
                for i in range(r*blowUp, (r+1)*blowUp):
                    for j in range(c*blowUp, (c+1)*blowUp):
                        room[i,j]=[0,50,0]
                        #room[i,j]=50
                        
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter('./chasey-vid-vanilla.avi',fourcc,float(FPS), 
									(width, height))
                            
    for k in range(len(BUStates)):
        
        frame=copy.deepcopy(room)
        
        #if flee is on honey
        if states[k][FLEE_IDX]==HONEY:
                for i in range(HONEY[0]*blowUp, (HONEY[0]+1)*blowUp):
                    for j in range(HONEY[1]*blowUp, (HONEY[1]+1)*blowUp):
                        frame[i,j]=[0,100,250]
                        
                        
        #draw flee agent
        cv2.circle(frame, BUStates[k][FLEE_IDX], radius, (0,200,0), -1)
        
        #draw chase agent
        cv2.circle(frame, BUStates[k][CHASE_IDX], radius, (0,200,200), -1)
        

        
        for f in range(int(FPS*secsPerState)):
            video.write(frame)
    video.release()
    
def blownUp(states, height, blowUpFactor):
    #adjust state coordinates for video size, and correct orientation:
    blownStates=[]
    for i in range(len(states)):
        
        state=[]
        for j in range(NUM_AGENTS):
            #0 and 1 for 2 dimensions
  
            state.append((int((states[i][j][1]+(1/2))*blowUpFactor),int((states[i][j][0]+(1/2))*blowUpFactor)))
      
        blownStates.append(state)
    return blownStates
        
            
def episode(MDP, chase_agent, flee_agent, greedy, graphical):
    #graphical turns board printing on or off (boolean)
    
    states=[]
    #removed for speed MDP.showBoard()
    MDP.isEndFunc()
    fReward=0
    while not(MDP.isEnd):
        
   
        #chase turn
        MDP.state[chase_agent.sIDX]=chase_agent.turn(MDP, greedy)
        #record state (for monitoring data-- not used in the RL itself)
        states.append(MDP.state.copy())
      
       
        
        if graphical:
            MDP.showBoard()
        
        cReward = MDP.chaseReward()
        
        if flee_agent.epStep != 0:
            flee_agent.update(fReward, MDP)
        
        MDP.isEndFunc()
        
        if not(MDP.isEnd):
            #flee turn
           MDP.state[flee_agent.sIDX]=flee_agent.turn(MDP, greedy)
           states.append(MDP.state.copy())
           
           if graphical:
            MDP.showBoard()
        
        fReward = MDP.fleeReward()
            
        chase_agent.update(cReward, MDP)
            
        chase_agent.step+=1
        flee_agent.step+=1
        chase_agent.epStep+=1
        flee_agent.epStep +=1
        
 
        # debug
        
        
        MDP.isEndFunc()
    flee_agent.update(fReward, MDP)
   
    return states
 

seed(5)
test_state=State()
test_state.showBoard()
chase_agent=Agent()
flee_agent=Agent()

chase_agent.sIDX=flee_agent.oIDX=CHASE_IDX
chase_agent.oIDX=flee_agent.sIDX=FLEE_IDX


print('Initialised... \n')
play(test_state, chase_agent, flee_agent)



    