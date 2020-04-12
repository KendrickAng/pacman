# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # keys: states, values: dict of action:q-value pairs
        # E.g { (0,0) : { NORTH: 1.0, SOUTH: 0.1, EAST: 0.05, WEST: 0.2, STOP: 0.5 }}
        self.qLearningTable = {}
        # explicitly override superclass's attributes for clarity
        self.lastAction = None
        self.lastState = None

    ################################
    # KENDRICK'S UTILITY FUNCTIONS #
    ################################
    def getStateCDictAfterCheck(self, state):
        """
        Returns the dict of action:value pairs for a given (x, y) state. If the state is new, all legal actions
        are added to the dict, initialised to 0.0.
        """
        # Initialise all state-action pairs for a given state
        if state not in self.qLearningTable:
            actions_values = [(a, 0.0) for a in self.getLegalActions(state)]
            self.qLearningTable[state] = Counter(actions_values)
        return self.qLearningTable[state]

    def isTerminalState(self, state):
        return len(self.getLegalActions(state)) == 0

    ##########################
    # ACTUAL AGENT FUNCTIONS #
    ##########################

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Guaranteed to return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # Counter class returns 0 if action doesn't exist
        return self.getStateCDictAfterCheck(state)[action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # No legal actions - return 0
        if self.isTerminalState(state):
            return 0.0

        # Get maximum q value from all possible state-action pairs
        max_action_q = float('-inf')
        # legal_actions = [a for a in self.getLegalActions(state) if a != Directions.STOP]
        for action in self.getLegalActions(state):
            value = self.getQValue(state, action)
            max_action_q = max(max_action_q, value)

        return max_action_q

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.isTerminalState(state):
            return None

        legal_actions = self.getLegalActions(state)
        # While training, legal moves don't include stopping or turning back (unless the ghost is trapping us or it's a dead end)
        # Also, if within distance 2, run away from the ghost
        if self.episodesSoFar * 1.0 / self.numTraining < 0.5:
        # if self.isInTraining() and flipCoin(0.5):
            pac_xy = nearestPoint(state.getPacmanPosition())

            for ghostIndex in range(1, state.getNumAgents()):
                ghost_xy = nearestPoint(state.getGhostPosition(ghostIndex))
                pac_ghost_dist = manhattanDistance(pac_xy, ghost_xy)

                if pac_ghost_dist > 2:
                    # explore
                    if len(legal_actions) > 1:
                        if Directions.STOP in legal_actions:
                            legal_actions.remove(Directions.STOP)

                    if self.lastAction is not None:
                        forbidden_act = Directions.REVERSE[self.lastAction]
                        if forbidden_act in legal_actions and len(legal_actions) > 1:
                            legal_actions.remove(forbidden_act)
                else:
                    # escape
                    escape_acts = []
                    for action in legal_actions:
                        next_state = state.generatePacmanSuccessor(action)
                        next_pac_xy = nearestPoint(next_state.getPacmanPosition())
                        if manhattanDistance(next_pac_xy, ghost_xy) >= pac_ghost_dist:
                            escape_acts.append(action)
                    if len(escape_acts) > 0:
                        legal_actions = escape_acts

        # legal_actions = [a for a in self.getLegalActions(state) if a != Directions.STOP]
        # if not self.isInTraining():
        #     print(self.getStateCDictAfterCheck(state))

        # If all actions are negative values, randomly pick an unseen action
        unseen_actions = []
        all_seen_neg = True
        for a in legal_actions:
            if self.getQValue(state, a) == 0:
                unseen_actions.append(a)
            elif self.getQValue(state, a) > 0:
                all_seen_neg = False

        if all_seen_neg and len(unseen_actions) > 0:
            return random.choice(unseen_actions)

        action = legal_actions[0]
        for a in legal_actions:
            if self.getQValue(state, a) > self.getQValue(state, action):
                action = a
        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        # legalActions = [a for a in self.getLegalActions(state) if a != Directions.STOP]
        action = None
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        isRandomAction = flipCoin(self.epsilon)
        if isRandomAction:
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # update last action and state
        self.lastAction = action
        self.lastState = state

        # SARSA
        q_predict = self.getQValue(state, action)
        q_target = reward + self.discount * self.getValue(nextState)
        if self.isTerminalState(nextState):
            q_target = reward

        self.getStateCDictAfterCheck(state).incrementAll([action], self.alpha * (q_target - q_predict))

    def getPolicy(self, state):
        # TODO: DEBUG
        # print("state {0}: {1}".format(state.getPacmanPosition(), self.getStateCDictAfterCheck(state)))
        # print(state.__class__.__name__)
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        """
        Computes max q-value from a state, returns 0.0 for terminal state
        """
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
