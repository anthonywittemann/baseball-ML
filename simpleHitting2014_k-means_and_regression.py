"""
	on sample hitting data from 2014
	- performing k-means clustering and visualize
	- regression of age vs stats

"""


import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

# constants dictionary
statTypes = {'STD-BATTING': 0, 'STD-PITCHING': 1, 'ADV-BATTING': 2, 'ADV-PITCHING': 3}


##########################################################
#############          PREPROCESSING        ##############
##########################################################
def parseSTD_Batting(dataset):
	# arrays to 
	#print type(dataset)
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
def parseSTD_Pitching(dataset):
	pass

# TODO - implement
def parseADV_Batting(dataset):
	pass

# TODO - implement
def parseADV_Pitching(dataset):
	pass


"""
	loads a csv matrix into 2d np array

"""
def loadCsv(filename, statType=0):
	dataset = np.array(list(csv.reader(open(filename,'rb'),delimiter=',')))
	#print dataset

	if statType == statTypes['STD-BATTING']:
		return parseSTD_Batting(dataset)
	elif statType == statTypes['STD-PITCHING']:
		return parseSTD_Pitching(dataset)
	elif statType == statTypes['ADV-BATTING']:
		return parseADV_Batting(dataset)
	elif statType == statTypes['ADV-PITCHING']:
		return parseADV_Pitching(dataset)





filename = 'data/std-batting-2014.csv'
playerStats, statLabels, playerInfo, infoLabels = loadCsv(filename, 0)
print('Loaded data file {0} with {1} players').format(filename, len(playerStats))

#print 'playerStats: {0}'.format(playerStats)
print 'playerStatLabels: {0}'.format(statLabels)

"""
	compare the age vs batting average - do regression, plot
"""

# select the age and BA data
batter_ages = playerStats[:, 1]
batting_avgs = playerStats[:, 14]
print 'ages: {0}'.format(batter_ages)
print 'BAs: {0}'.format(batting_avgs)

# TODO filter out -1 data



# split into training, testing - 80:20 split
### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
#batter_ages_flat = numpy.reshape( numpy.array(batter_ages), (len(batter_ages), 1))
#batting_avgs_flat = numpy.reshape( numpy.array(batter_avgs), (len(batter_avgs), 1))
batter_ages_train, batter_ages_test, batting_avgs_train, batting_avgs_test = train_test_split(batter_ages, batting_avgs, test_size=0.2, random_state=42)

reg = linear_model.LinearRegression()
reg.fit( batter_ages_train, batting_avgs_train )
print 'slope:', reg.coef_
print 'score on test data:', reg.score( batter_ages_test, batting_avgs_test )

# plot the linear regression

try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()
