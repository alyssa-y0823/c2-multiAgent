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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] # bc there might be multiple best scores
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # if pacman takes this action right now, how good is the resulting state?
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
        newFood = successorGameState.getFood().asList() # grid showing where food still exists
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # how long ghosts remained scared for (after power pellet eaten)

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        if newFood:
            distances = [manhattanDistance(newPos, food) for food in newFood]
            score += 1.0 / min(distances) # closer food is better
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDist = manhattanDistance(newPos, ghost.getPosition())
            if scaredTime > 0:
                score += 2.0 / ghostDist # closer scared ghost is better (can eat it)
            else:
                if ghostDist < 2: # too close to non-scared ghost is bad
                    score -= 500
                else:
                    score -= 1.0 / ghostDist # farther from non-scared ghost is better

        if action == Directions.STOP:
            score -= 10 # discourage stopping

        return score

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
        "*** YOUR CODE HERE ***"
        
        numAgents = gameState.getNumAgents()

        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP
        
        def minimax(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # if we've gone through all agents, we go to the next depth level
            if agentIndex == numAgents:
                depth += 1
                agentIndex = 0
            
            # reached the requested (maximum) depth, stop expanding & evaluate this state
            if depth == self.depth:
                return self.evaluationFunction(state)
            
            actions = state.getLegalActions(agentIndex)
            if not actions: # no legal actions
                return self.evaluationFunction(state)
            if agentIndex == 0: # pacman's turn (maximizer)
                best = float('-inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, agentIndex + 1, depth)
                    if val > best:
                        best = val
                return best
            else: # ghost's turn (minimizer)
                best = float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, agentIndex + 1, depth)
                    if val < best:
                        best = val
                return best
            
        bestAction = None
        bestValue = float('-inf')
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            val = minimax(successor, 1, 0) # start with first ghost and depth 0
            if val > bestValue:
                bestValue = val
                bestAction = action

        return bestAction
            
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def alphabeta(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # if we've gone through all agents, we go to the next depth level
            if agentIndex == numAgents:
                depth += 1
                agentIndex = 0
            
            # reached the requested (maximum) depth, stop expanding & evaluate this state
            if depth == self.depth:
                return self.evaluationFunction(state)
            
            actions = state.getLegalActions(agentIndex)
            if not actions: # no legal actions
                return self.evaluationFunction(state)
            
            # max node (pacman)
            if agentIndex == 0:
                value = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(
                        value,
                        alphabeta(successor, agentIndex + 1, depth, alpha, beta)
                    )
                    if value > beta:   # prune (not on equality)
                        return value
                    alpha = max(alpha, value)
                return value

            # min node (ghost)
            else:
                value = float("inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(
                        value,
                        alphabeta(successor, agentIndex + 1, depth, alpha, beta)
                    )
                    if value < alpha:  # prune)
                        return value
                    beta = min(beta, value)
                return value
            
        # root call for pacman (maximizer)
        bestValue = float("-inf")
        bestAction = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 1, 0, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
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
        numAgents = gameState.getNumAgents()

        def expectimax(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex >= numAgents:
                agentIndex = 0
                depth += 1

            if depth == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            # MAX node (pacman)
            if agentIndex == 0:
                value = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, agentIndex + 1, depth))
                return value

            # CHANCE node (ghost)
            else:
                total = 0
                probability = 1.0 / len(actions)
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    total += probability * expectimax(successor, agentIndex + 1, depth)
                return total

        # at root (MAX), pacman chooses action with highest expected value
        bestValue = float("-inf")
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # base score from the game
    score = currentGameState.getScore()

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    pellets = currentGameState.getCapsules()

    # food: closer food is better, and fewer total food pellets is better
    if foodList:
        foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
        closestFoodDist = min(foodDistances)
        score += 2.0 / closestFoodDist   # strong pull toward nearby food: prevents wandering
        score -= 4 * len(foodList)       # fewer food pellets is better: tries to finish the board

    # pellets: makes it go eat pellets earlier
    score -= 20 * len(pellets)

    # ghost
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        dist = manhattanDistance(pacmanPos, ghostPos)

        if ghost.scaredTimer > 0:
            # eat scared ghosts, but not too much that it ignores food
            score += 5.0 / dist
        else:
            # really really avoids active ghosts
            if dist < 2:
                score -= 500
            else:
                score -= 2.0 / dist

    return score

# Abbreviation
better = betterEvaluationFunction
