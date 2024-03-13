# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        # Start from Pacman's turn (index 0)
        # Disregards minimax value from the minimax function while keeping optimal action
        action, _ = self.minimax(gameState, 0, self.depth)
        return action


    # Defines the minimax algorithm
    def minimax(self, gameState, agentIndex, depth):
        
        # Returns the best action for the given agent and the corresponding value.
        
        # Check terminal condition
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return None, self.evaluationFunction(gameState)

        # Get legal actions for the agent
        legalActions = gameState.getLegalActions(agentIndex)

        # If there are no legal actions, return the evaluation
        if not legalActions:
            return None, self.evaluationFunction(gameState)

        # If it's Pacman's turn (agentIndex == 0), maximize the score from the maximize function
        if agentIndex == 0:
            return self.maximize(gameState, agentIndex, depth)
        # If it's a Ghost's turn, minimize the score from the minimize function
        else:
            return self.minimize(gameState, agentIndex, depth)

    def maximize(self, gameState, agentIndex, depth):
        
        # Returns the best action for Pacman and the corresponding maximum value
        
        # Sets value to negative infinity in order to find maximum value
        value = float('-inf')
        # Initialize the best action to None. This will store the action associated with the best value
        bestAction = None
        # Iterate over all possible legal actions the agent can take in the current game state
        for action in gameState.getLegalActions(agentIndex):
            # Generate the successor state resulting from the agent taking the current action
            successorState = gameState.generateSuccessor(agentIndex, action)
            # Use the minimax method recursively to get the value of the resulting state
            _, nextValue = self.minimax(successorState, agentIndex + 1, depth)
            
            # If the value from this action is greater than the best found so far, update the best value and action
            if nextValue > value:
                value = nextValue
                bestAction = action
        # Return the best action found and its associated value        
        return bestAction, value

    def minimize(self, gameState, agentIndex, depth):
        
        # Returns the best action for a Ghost and the corresponding minimum value
        
        # Searches for the minimum value by setting it to infinity first
        value = float('inf')
        bestAction = None
        # Determine the next agent. If it's the last ghost, set to Pacman and decrease depth
        # Otherwise, just increment the agent index
        nextAgent = agentIndex + 1 if agentIndex + 1 < gameState.getNumAgents() else 0
        # If the next agent is Pacman (with index 0), we're starting a new depth level, so decrease the depth counter
        if nextAgent == 0:
            depth -= 1
        # Iterate over all possible legal actions the current agent can take in the current game state
        for action in gameState.getLegalActions(agentIndex):
            # Generate the successor state resulting from the agent taking the current action
            successorState = gameState.generateSuccessor(agentIndex, action)
            _, nextValue = self.minimax(successorState, nextAgent, depth)
            if nextValue < value:
                value = nextValue
                bestAction = action
        return bestAction, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Define the alpha-beta function for the maximizer (Pacman)
        def maxValue(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Any real value returned will be larger than negative infinity, so initialising with -infinity is nice 
            v = float('-inf')
            """ 
            Looks for all legal actions for the selected agent
            Acts as the logic for the Pacman agent in the minimax algorithm
            """
            for action in gameState.getLegalActions(agentIndex):
                # Get the game state that results from the agent taking the current action.
                successorState = gameState.generateSuccessor(agentIndex, action)
                # V is set to the maximum value between its current and the value of a successor state 
                # The value function will recursively check the value of a state considering the future actions of all agents
                v = max(v, value(successorState, depth, agentIndex + 1, alpha, beta))
                # Alpha-beta pruning step which checks if v is greater than beta. If so further actions are not evaluated for the agent.
                # The minimizing agent will avoid this branch  ensuring the game dosent reach this state
                if v > beta:
                    return v
                # Updates the alpha value to the largest value the maximising agent has found
                alpha = max(alpha, v)
            # returns final maximum value of v
            return v

        # Define the alpha-beta function for the minimizer (Ghosts)
        def minValue(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Any real value returned will be less than infinity, so initialising with infinity is nice 
            v = float('inf')
            
            """ 
            Looks for all legal actions for the selected agent. 
            Acts as the logic for the Ghost agent in the minimax algorithm
            """
            for action in gameState.getLegalActions(agentIndex):
                # Resulting game state from the latest action of the agent
                successorState = gameState.generateSuccessor(agentIndex, action)
                # V is set to the minimum value between its current and the value of a successor state
                v = min(v, value(successorState, depth, agentIndex + 1, alpha, beta))
                # If v is less than alpha the step stops evaluating. Maximising agent will prohibit this state from occuring this V is returned early
                if v < alpha:
                    return v
                # Beta value is updated to the best value the agent has found so far
                beta = min(beta, v)
            # Final minimum value of V is returned after evaluating all actions    
            return v

        # Value function to determine whether to call maxValue or minValue
        def value(gameState, depth, agentIndex, alpha, beta):
            if agentIndex == gameState.getNumAgents():
                return maxValue(gameState, depth + 1, 0, alpha, beta)
            elif agentIndex == 0:  # Pacman's turn
                return maxValue(gameState, depth, agentIndex, alpha, beta)
            else:  # Ghost's turn
                return minValue(gameState, depth, agentIndex, alpha, beta)

        # Main function body
        alpha = float('-inf')
        beta = float('inf')
        bestAction = Directions.STOP
        bestValue = float('-inf')

        # Iterate over all possible actions that Pacman can take from the current state
        for action in gameState.getLegalActions(0):
    
            # Generate the game state resulting from Pacman taking the current action
            successorState = gameState.generateSuccessor(0, action)
        
            # Store the current best value before evaluating this action's outcome
            prevMax = bestValue
        
            # Evaluate the successor state, and update bestValue if this action leads to a better state
            bestValue = max(bestValue, value(successorState, 0, 1, alpha, beta))
        
            # If the current action leads to a state with a higher value than previously known,
            # update the best action to the current action
            if bestValue > prevMax:
                bestAction = action
        
            # Alpha-beta pruning: If the best value found is greater than beta, we can prune 
            # the rest of the successors as the minimizer (ghost) will not choose this branc
            if bestValue > beta:
                return bestAction
        
            # Update the alpha value, which represents the best (maximum) value Pacman can 
            # guarantee to get from the current state onwards
            alpha = max(alpha, bestValue)    
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
