# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getInitialState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isFinalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getNextStates(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getActionCost(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# Initial Documentation for DFS:
# Your search algorithm needs to return a list of actions that reaches the
# goal. Make sure to implement a graph search algorithm.

# To get started, you might want to try some of these simple commands to
# understand the search problem that is being passed in:

# print("Start:", problem.getInitialState())
# print("Is the start a goal?", problem.isFinalState(problem.getInitialState()))
# print("Start's successors:", problem.getNextStates(problem.getInitialState()))

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    searchStack = util.Stack()
    visited = set()  # used because graph search - contains only keys not values.
    a = []
    searchStack.enqueue((problem.getInitialState(), a))    # stack contains the state and the actions to get to it.
    while not searchStack.isEmpty():
        # s = state, a = action to get to state
        s, a = searchStack.dequeue()
        if problem.isFinalState(s):
            # print(a)  # used for debugging.
            return a
        if s not in visited:    # we append only not visited states.
            visited.add(s)
            for successor, action, cost in problem.getNextStates(s):
                # a + [action] = the actions to get to current successor.
                searchStack.enqueue((successor, a + [action]))
    return a

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    searchQueue = util.Queue()  # same as dfs, only this time we use a queue.
    visited = set()  # used because graph search.
    a = []
    searchQueue.enqueue((problem.getInitialState(), a))    # queue contains the state and the actions to get to it.
    while not searchQueue.isEmpty():
        # s = state, a = action to get to state
        s, a = searchQueue.dequeue()
        if problem.isFinalState(s):
            return a
        if s not in visited:
            visited.add(s)
            for successor, action, cost in problem.getNextStates(s):
                # a + [action] = the actions to get to current successor.
                searchQueue.enqueue((successor, a + [action]))
    return a

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    searchPQueue = util.PriorityQueue()
    visited = set()  # used because graph search.
    a = []
    searchPQueue.enqueue((problem.getInitialState(), a, 0), 0)    # queue contains the state, the actions to get to it and the cost (same as priority).
    while not searchPQueue.isEmpty():
        # s = state, a = action to get to state
        s, a, c = searchPQueue.dequeue()
        if problem.isFinalState(s):
            return a
        if s not in visited:  # goes here if not visited or current cost smaller than state with smallest cost.
            visited.add(s)
            for successor, action, cost in problem.getNextStates(s):
                # a + [action] = the actions to get to current successor.
                newCost = c + cost
                searchPQueue.enqueue((successor, a + [action], newCost), newCost)
    return a

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    searchPQueue = util.PriorityQueue()
    visited = set()  # used because graph search.
    a = []
    searchPQueue.enqueue((problem.getInitialState(), a, 0), 0)    # queue contains the state, the actions to get to it and the cost (same as priority).
    while not searchPQueue.isEmpty():
        # s = state, a = action to get to state
        s, a, c = searchPQueue.dequeue()
        if problem.isFinalState(s):
            return a
        if s not in visited :  # goes here if not visited or current cost smaller than state with smallest cost.
            visited.add(s)
            for successor, action, cost in problem.getNextStates(s):
                # a + [action] = the actions to get to current successor.
                newCost = c + cost
                searchPQueue.enqueue((successor, a + [action], newCost), newCost + heuristic(successor, problem))
    return a

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
