# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:52:02 2021

@author: Lei Yang
"""


from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

ml, data, rankings = LoadMovieLensData()

evalData = EvaluationData(data, rankings)

# Train on leave-One-Out train set
trainSet = evalData.GetLOOCVTrainSet()
sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

leftOutTestSet = evalData.GetLOOCVTestSet()




# Build up dict to lists of (int(movieID), predictedrating) pairs
topN = defaultdict(list)
k = 10
###############new code#####################################
for uiid in range(trainSet.n_users):
    #Get top k items rated for this user
    UserRatings=trainSet.ur[uiid]
    kNeighbors = heapq.nlargest(k,  UserRatings, key=lambda t: t[1])
    # Get similar items to stuff this user liked (weighted by rating)
    candidates = defaultdict(float)
    for itemID, rating in kNeighbors:
        similarityRow=simsMatrix[itemID]

        for innerID, score in enumerate(similarityRow):
            candidates[innerID] += score*(rating/5.0)
    
    
    

    # Build a dictionary of stuff the user has already seen
    watched = {}
    for itemID, rating in trainSet.ur[uiid]:
        watched[itemID] = 1
        
    # Get top-rated items from similar users:
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            movieID = trainSet.to_raw_iid(itemID)
            topN[int(trainSet.to_raw_uid(uiid))].append( (int(movieID), 0.0) )
            pos += 1
            if (pos > 40):
                break
    
# Measure
print("HR", RecommenderMetrics.HitRate(topN, leftOutTestSet))   