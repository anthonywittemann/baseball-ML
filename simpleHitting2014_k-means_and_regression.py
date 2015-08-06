"""
	on sample hitting data from 2014
	- performing k-means clustering and visualize
	- regression of age vs stats

"""
import csv
import math
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	dataset = [dataset[x].split(',') for x in range(len(dataset))]
	players = []
	statLabels = []
	# Rk,Name,Age,Tm,Lg,G,PA,AB,R,H,2B,3B,HR,RBI,SB,CS,BB,SO,BA,OBP,SLG,OPS,OPS+,TB,GDP,HBP,SH,SF,IBB,Pos Summary
	playerNum = 0
	for i in range(len(dataset)):
		if math.isnan( dataset[i][0] ):
			if dataset[i][0] == 'Rk':
				statLabels = [str(x) for x in dataset[i]]
		else:
			players[playerNum] = [float(x) for x in dataset[i]]
			players[playerNum][1] = [str(dataset[i][1])] # name
			players[playerNum][3] = [str(dataset[i][3])] # age
			players[playerNum][4] = [str(dataset[i][4])] # team
			players[playerNum][5] = [str(dataset[i][5])] # league
			players[playerNum][29] = [str(dataset[i][29])] # pos summary
 			playerNum += 1

	return players, statLabels



filename = 'data/std-batting-2014.csv'
players, statLabels = loadCsv(filename)
print('Loaded data file {0} with {1} rows').format(filename, len(dataset))