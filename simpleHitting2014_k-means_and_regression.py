"""
	on sample hitting data from 2014
	- performing k-means clustering and visualize
	- regression of age vs stats

"""
import csv
import numpy as np

def loadCsv(filename):
	dataset = np.array(list(csv.reader(open(filename,"rb"),delimiter=',')))
	#print dataset

	# arrays to return
	playerStats = []
	playerInfo = []
	statLabels = []
	infoLabels = []
	
	# Rk,Name,Age,Tm,Lg,G,PA,AB,R,H,2B,3B,HR,RBI,SB,CS,BB,SO,BA,OBP,SLG,OPS,OPS+,TB,GDP,HBP,SH,SF,IBB,Pos Summary
	statIndices = [0,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
	infoIndices = [1,3,4,5,29]

	playerNum = 0
	for i in range(len(dataset)):
		print dataset[i]

		if len(dataset[i]) != 0: # skip over blank lines

			# grab the labels
			if dataset[i][0] == 'Rk' and len(statLabels) == 0:
				statLabels = [str(x) for x in dataset[i][statIndices]]
				infoLabels = [str(x) for x in dataset[i][infoIndices]]
				print 'statLabels {0}'.format(statLabels)
				print 'infoLabels {0}'.format(infoLabels)

			#grab player info and stats
			elif dataset[i][0] != 'Rk':
				playerStats[playerNum] = [float(x) for x in dataset[i][statIndices]]
				playerInfo[playerNum] = [float(x) for x in dataset[i][infoIndices]]
 				playerNum += 1
		

	return playerStats, statLabels, playerInfo, infoLabels



filename = 'data/std-batting-2014.csv'
playerStats, statLabels, playerInfo, infoLabels = loadCsv(filename)
print('Loaded data file {0} with {1} players').format(filename, len(playerStats))