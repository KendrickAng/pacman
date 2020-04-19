# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we worked on in the search project; here its all in one place
    Basically BFS to find the closest food, returns distance to closest food, and None if nothing found.
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class NewExtractor(FeatureExtractor):
    """
    Design you own feature extractor here. You may define other helper functions you find necessary.
    """
    # Need some way to have memory
    LAST_ACTION = 'lastAction'
    history = { LAST_ACTION: None }

    def getFeatures(self, state, action):
        "*** YOUR CODE HERE ***"
        features = util.Counter()

        # First feature is always the bias term - 1.0
        features["bias"] = 1.0

        # 1. Follow Last Action - when 2 actions are equally good, continue in the same direction
        self.addLastActionFeature(features, action)

        # 2. Chase Closest Pill - pacman needs to know closest food to make progress
        self.addClosestFoodFeature(features, state, action)

        # 3. Avoid Closest Ghost - pacman should know where the closest ghost is to avoid it
        # self.add

        # update history
        self.setLastAction(action)

        # prevent too wide of divergence while training
        features.divideAll(10.0)
        #print(features)
        return features

    def addClosestFoodFeature(self, features, state, action):
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        food, walls = state.getFood(), state.getWalls()

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update will diverge wildly
            max_path_length = walls.width * walls.height
            features["chase_closest_food"] = (max_path_length - float(dist)) / max_path_length

    def addLastActionFeature(self, features, action):
        """
        Feature value = 1.0 if last action = current action, else 0.0
        """
        lastAction = self.getLastAction()
        if lastAction is not None:
            features["follow_last_action"] = 1.0 if lastAction == action else 0.0
        else:
            features["follow_last_action"] = 0

    def getLastAction(self):
        return NewExtractor.history[NewExtractor.LAST_ACTION]

    def setLastAction(self, action):
        NewExtractor.history[NewExtractor.LAST_ACTION] = action
        
