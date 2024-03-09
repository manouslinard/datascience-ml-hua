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
import random
import util

from game import Agent


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (exercise 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getPossibleActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateNextState(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWinningState():
        Returns whether or not the game state is a winning state

        gameState.isLosingState():
        Returns whether or not the game state is a losing state
        """
        # starts at depth = 0
        result = self.value(gameState, self.index, 0)
        # Return action
        return result[1]

    def value(self, gameState, index, depth):
        # Terminal States:
        if gameState.isWinningState() or gameState.isLosingState() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if index == 0:  # == Pacman -> max (pacman wants to get the highest score)
            return self.max_value(gameState, index, depth)
        else:   # == Ghosts -> min (ghosts want to lower pacman's score)
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        # pacman uses this function...
        max_value = float("-inf")
        max_action = None

        for action in gameState.getPossibleActions(index):
            successor = gameState.generateNextState(index, action)

            # calculates successor's index:
            successor_index = (index + 1) % gameState.getNumAgents()

            # successors are only ghosts, depth (ply) does not change...
            current_value = self.value(successor, successor_index, depth)[0]    # gets the value

            # keeps the max value and its action.
            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_value, max_action

    def min_value(self, gameState, index, depth):
        # ghosts use this function...
        min_value = float("inf")
        min_action = None

        for action in gameState.getPossibleActions(index):
            successor = gameState.generateNextState(index, action)
            # calculates successor's index:
            successor_index = (index + 1) % gameState.getNumAgents()

            # if successor is pacman (0) -> increase successor depth (a ply is over - all ghosts and pacman have done an action).
            successor_depth = depth + 1 if not successor_index else depth

            current_value = self.value(successor, successor_index, successor_depth)[0]  # gets the value

            if current_value < min_value:
                min_value = current_value
                min_action = action

        return min_value, min_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (exercise 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # starts at depth = 0
        result = self.value(gameState, self.index, 0, float("-inf"), float("inf"))
        # Return action
        return result[1]

    def value(self, gameState, index, depth, a, b):
        # Terminal States:
        if gameState.isWinningState() or gameState.isLosingState() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if index == 0:  # == Pacman -> max (pacman wants to get the highest score)
            return self.max_value(gameState, index, depth, a ,b)
        else:   # == Ghosts -> min (ghosts want to lower pacman's score)
            return self.min_value(gameState, index, depth, a, b)

    def max_value(self, gameState, index, depth, a, b):
        # pacman uses this function...
        max_value = float("-inf")
        max_action = None

        for action in gameState.getPossibleActions(index):
            successor = gameState.generateNextState(index, action)

            # calculates successor's index:
            successor_index = (index + 1) % gameState.getNumAgents()

            # successors are only ghosts, depth (ply) does not change...
            current_value = self.value(successor, successor_index, depth, a, b)[0]    # gets the value

            # keeps the max value and its action.
            if current_value > max_value:
                max_value = current_value
                max_action = action

            # in exercise, v = max_value, so:
            if max_value > b:   # prunes
                return max_value, max_action

            a = max(a, max_value)

        return max_value, max_action


    def min_value(self, gameState, index, depth, a, b):
        # ghosts use this function...
        min_value = float("inf")
        min_action = None

        for action in gameState.getPossibleActions(index):
            successor = gameState.generateNextState(index, action)
            # calculates successor's index:
            successor_index = (index + 1) % gameState.getNumAgents()

            # if successor is pacman (0) -> increase successor depth (a ply is over - all ghosts and pacman have done an action).
            successor_depth = depth + 1 if not successor_index else depth

            current_value = self.value(successor, successor_index, successor_depth,a,b)[0]  # gets the value

            if current_value < min_value:
                min_value = current_value
                min_action = action

            # in exercise, v = min_value, so:
            if min_value < a:   # prunes
                return min_value, min_action

            b = min(b, min_value)

        return min_value, min_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (exercise 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # starts at depth = 0
        result = self.value(gameState, self.index, 0)
        # Return action
        return result[1]

    def value(self, gameState, index, depth):
        # Terminal States:
        if gameState.isWinningState() or gameState.isLosingState() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if index == 0:  # == Pacman -> max (pacman wants to get the highest score)
            return self.max_value(gameState, index, depth)
        else:   # == Ghosts -> min (ghosts want to lower pacman's score)
            return self.exp_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        # pacman uses this function...
        max_value = float("-inf")
        max_action = None

        for action in gameState.getPossibleActions(index):
            successor = gameState.generateNextState(index, action)

            # calculates successor's index:
            successor_index = (index + 1) % gameState.getNumAgents()

            current_value = self.value(successor, successor_index, depth)[0]    # gets the value

            # keeps the max value and its action.
            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_value, max_action

    def exp_value(self, gameState, index, depth):
        # ghosts use this function...
        exp_value = 0
        actions = gameState.getPossibleActions(index)
        p = 1 / len(actions)    # propability of each successor to occur.

        for action in actions:
            successor = gameState.generateNextState(index, action)
            # calculates successor's index:
            successor_index = (index + 1) % gameState.getNumAgents()

            # if successor is pacman (0) -> increase successor depth (a ply is over - all ghosts and pacman have done an action).
            successor_depth = depth + 1 if not successor_index else depth

            value = self.value(successor, successor_index, successor_depth)[0]

            exp_value += p * value  # gets the expected value

        # returns only the expected value, the action is decided by MAX.
        return exp_value, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (exercise 4).

    DESCRIPTION:
    This evaluation function takes into account the score
    and distances between pacman and game's ghosts, remaining food and capsules.
    Also, the evaluation is better when pacman is closer to food.
    It is also logical for the evaluation to be worse when there is much foods and many capsules remaining,
    because that means that pacman still needs time to finish the game. Thus, we substract the food count and
    capsule count from final sum value.
    Also, game score should be taken into consideration (a higher game score is better) and thus will be be adding it to
    the sum. When player is closer to food, the evaluation is better.
    Lastly, if ghosts are close to the player, we prioritize on escaping and not eating the nearest food.
    """
    pacman_pos = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()
    food_list = currentGameState.getFood().asList()
    game_score = currentGameState.getScore()

    # sets initial closest_food position:
    closest_food = min([manhattanDistance(pacman_pos, food_position) for food_position in food_list], default=1)

    for ghost_pos in ghost_positions:
        ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
        if ghost_distance <= 2:  # ghost is close...
            closest_food = float('inf')

    # invertion of closest_food -> higher evaluation for minimum distance.
    return (1 / closest_food) + game_score - len(food_list) - len(currentGameState.getCapsules())

# Abbreviation
better = betterEvaluationFunction
