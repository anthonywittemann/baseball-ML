"""
	on sample hitting data from 2012 - 2014
	- performing regression of age vs stats

"""


import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

# constants dictionary
from outlier_cleaner import outlierCleaner

statTypes = {'STD-BATTING': 0, 'STD-PITCHING': 1, 'ADV-BATTING': 2, 'ADV-PITCHING': 3}


##########################################################
#############          PREPROCESSING        ##############
##########################################################
def parseSTD_Batting(dataset):
    # print type(dataset)
    # have had to hard code number of players - FIX LATER
    playerStats = -1.0 * np.ones((1600,25))
    playerInfo = np.empty((1600,5), dtype='|S30') # hopefully no players with names > 30 chars
	#print 'playerStats: {0}'.format(playerStats)
	#print 'playerInfo: {0}'.format(playerInfo)
    statLabels = []
    infoLabels = []

	# Rk,Name,Age,Tm,Lg,G,PA,AB,R,H,2B,3B,HR,RBI,SB,CS,BB,SO,BA,OBP,SLG,OPS,OPS+,TB,GDP,HBP,SH,SF,IBB,Pos Summary
    statIndices = [0,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    infoIndices = [1,3,4,5,29]

    playerNum = 0
    for i in range(len(dataset)):
		#print dataset[i]

		if len(dataset[i]) != 0: # skip over blank lines

			# grab the labels
			if dataset[i][0] == 'Rk' and len(statLabels) == 0:
				statLabels = [str(dataset[i][j]) for j in statIndices]
				infoLabels = [str(dataset[i][j]) for j in infoIndices]
				#print 'statLabels: {0}'.format(statLabels)
				#print 'infoLabels: {0}'.format(infoLabels)

			#grab player info and stats only for players with full stats
			elif dataset[i][0] != 'Rk' and len(dataset[i]) == 30:
				playerInfo[playerNum] = [str(dataset[i][j]) for j in infoIndices]
				#print 'playerInfo: {0}'.format(playerInfo)
				
				# tried assignment with list compression but wasn't able to hadle blank entries
				#playerStats[playerNum] = [float(dataset[i][j]) for j in statIndices if dataset[i][j].isdigit()]
				
 				for j in range(len(statIndices)):
 					if dataset[i][statIndices[j]].isdigit():
 						playerStats[playerNum][j] = dataset[i][statIndices[j]]

 				playerNum += 1
 				#print 'playerStats: {0}'.format(playerStats)



    return playerStats, statLabels, playerInfo, infoLabels

# TODO - implement
def parseSTD_Pitching(dataset): pass

# TODO - implement
def parseADV_Batting(dataset): pass

# TODO - implement
def parseADV_Pitching(dataset): pass


"""
	loads a csv matrix into 2d np array

"""
def loadCsv(filename, statType=0):
	dataset = np.array(list(csv.reader(open(filename,'rb'),delimiter=',')))
	#print dataset

	if statType == statTypes['STD-BATTING']: return parseSTD_Batting(dataset)
	elif statType == statTypes['STD-PITCHING']: return parseSTD_Pitching(dataset)
	elif statType == statTypes['ADV-BATTING']: return parseADV_Batting(dataset)
	elif statType == statTypes['ADV-PITCHING']: return parseADV_Pitching(dataset)




filename = 'data/std-batting-2014.csv'
playerStats, statLabels, playerInfo, infoLabels = loadCsv(filename, 0)
print('Loaded data file {0} with {1} players').format(filename, len(playerStats))

#print 'playerStats: {0}'.format(playerStats)
#print 'playerStatLabels: {0}'.format(statLabels)

filename1 = 'data/std-batting-2013.csv'
playerStats1, statLabels1, playerInfo1, infoLabels1 = loadCsv(filename1, 0)
print('Loaded data file {0} with {1} players').format(filename1, len(playerStats1))

filename2 = 'data/std-batting-2012.csv'
playerStats2, statLabels2, playerInfo2, infoLabels2 = loadCsv(filename2, 0)
print('Loaded data file {0} with {1} players').format(filename2, len(playerStats2))

# stack arrays vertically
playerStats = np.vstack((playerStats, playerStats1))
playerStats = np.vstack((playerStats, playerStats2))
statLabels = np.vstack((statLabels, statLabels1))
statLabels = np.vstack((statLabels, statLabels2))
print len(playerStats[:, 0])

"""
    compare the age vs batting average - do regression, plot
"""

# select the age and BA data
batter_ABs = playerStats[:, 3]
batter_Hs = playerStats[:, 5]
batter_ages = playerStats[:, 1]

# FILTER out na batting avg, AB < 30
import itertools
selector = filter(lambda x: not np.isnan(x) and x > 60, batter_ABs)
print 'number of filtered players: {0}'.format(len(selector))

batter_ABs = np.array(selector)
batter_Hs = np.array(list(itertools.compress(batter_Hs, selector)))

batter_ages = np.array(list(itertools.compress(batter_ages, selector)))
batting_avgs = np.array(map(lambda x,y: x/y, batter_Hs, batter_ABs))


#print 'filtered ages: {0}'.format(batter_ages)
#print 'filtered BAs: {0}'.format(batting_avgs)


# SPLIT into training, testing - 80:20 split
batter_ages_flat = np.reshape(np.array(batter_ages), (len(batter_ages), 1))
batting_avgs_flat = np.reshape(np.array(batting_avgs), (len(batting_avgs), 1))
batter_ages_train, batter_ages_test, batting_avgs_train, batting_avgs_test = \
    train_test_split(batter_ages_flat, batting_avgs_flat, test_size=0.2, random_state=42)



##########################################################
#############    TRAINING & PLOTTING        ##############
##########################################################

reg = linear_model.LinearRegression()
reg.fit(batter_ages_train, batting_avgs_train)
print 'slope:', reg.coef_
print 'score on test data:', reg.score(batter_ages_test, batting_avgs_test)


# PLOT the linear regression

try:
    plt.plot(batter_ages_flat, reg.predict(batter_ages_flat), color="blue")
except NameError:
    pass
plt.scatter(batter_ages_flat, batting_avgs_flat)
plt.xlabel("ages")
plt.ylabel("batting averages")
plt.show()


# identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(batter_ages_train)
    cleaned_data = outlierCleaner(predictions, batter_ages_train, batting_avgs_train)
except NameError:
    print "can't make predictions to use in identifying outliers"


# only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, avgs, errors = zip(*cleaned_data)
    ages = np.reshape(np.array(ages), (len(ages), 1))
    avgs = np.reshape(np.array(avgs), (len(avgs), 1))

    # refit the data
    try:
        reg.fit(ages, avgs)
        print 'slope after outlier removal:', reg.coef_
        print 'score on test data after outlier removal:', reg.score(batter_ages_test, batting_avgs_test)
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(ages, avgs)
    plt.xlabel("ages")
    plt.ylabel("BA")
    plt.show()


else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"