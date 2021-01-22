""" A module with functions to read a standard oxcal output
------------------------------------------

It contains functions for reading and plotting age distribution
functions, filled summed PDFs, and Kernel Density Estimates 

It can be loaded in a Jupyter Notebook using the code:

 % import OxcalReader

 ------------------------------------------"""
try:
	import simplejson as json
except (ImportError,):
	import json

import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def ReadFile(fp):

	"""
		Reads a file in JSON notation, taking the filepath as argument
		Returns a nested Python Dictionary object, with the structure of the Oxcal 
		Model
	"""

	with open(fp, 'r') as f:
		data = json.load(f)

	return data


def FillBetween(ax,dat,color,median = False,prob = 'posterior'):
	
	s  = dat['posterior']['start']
	r = dat['posterior']['resolution']
	l = len(dat['posterior']['prob'])
	xi = np.linspace(s,s+(l-1)*r,l)
	med = dat['posterior']['median']

	ax.fill_between(xi,y1=np.ones(l)*0.2,y2 = np.array(dat[prob]['prob'])+0.2,alpha= 0.75,color = color)
	if median == True:
		ax.scatter(x =med,y=0.2,marker = "D",zorder= 10,alpha = 1,color = 'white',edgecolor =color)
		
	
	return ax,med

def MakeRects(ax,dat,n,**kwargs):

	"""
		Updates a matplotlib ax object, taking the Oxcal radiocarbon date object
		and a height component (n, lower bound of the rectangle)
		Drawing some rectangles for each radiocarbon date
		Returns the updated ax object
	"""

	# shorten some variable names
	s  = dat['posterior']['start']
	r = dat['posterior']['resolution']
	l = len(dat['posterior']['prob'])
	xi = np.linspace(s,s+(l-1)*r,l)
	
	# for each age increment, draw a rectangle with opacity proportional to the
	# probability density
	for xi,prob in zip(xi,dat['posterior']['prob']):
		rect = Rectangle((xi,n),width=r,height=1,alpha=prob,**kwargs)
		ax.add_table(rect)
	
	# return the artist
	return ax