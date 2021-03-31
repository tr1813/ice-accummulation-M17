""" A module with functions to read a standard oxcal output
------------------------------------------

It contains functions for reading and plotting age distribution
functions, filled summed PDFs, and Kernel Density Estimates 

It can be loaded in a Jupyter Notebook using the code:

 % import OxcalReader

 ------------------------------------------"""

import json
import subprocess
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def MakeJSON(file):
	"""
	Calls subprocess to run the .js file and make a stringified JSON object
    Outputs a file in JSON notation called 'output.txt'
	"""

	with open(file, 'r') as f:
		body = f.readlines()
		header = ["var ocd = [];\n","var calib = [];\n","var model=[];\n"]
		footer = ["const obj = {ocd, calib,model}\n", "const fs = require('fs');\n",
				"fs.writeFile(\"output.txt\", JSON.stringify(obj), function(err) {\n",
				"	if(err) {\n",
				"	return console.log(err);\n",
				"	}\n",
				"	console.log(\"The file was saved!\");\n",
				"});"]
				
		new = header + body + footer
		f.close()
        
	name = file[:-3] + '_export.js'
    
	with open(name, 'w+') as f2:
		for line in new:
			f2.write(str(line))
		f2.close()
        
	subprocess.call(["node","{}".format(name)])
	subprocess.call(["rm","{}".format(name)])

def ReadJSON():
	with open('output.txt', 'r') as f:
		model = json.load(f)
		f.close()
	subprocess.call(['rm','output.txt'])
    
	return model

def LoadFile(file):
	"""
		Makes an output file in JSON notation, taking the filepath as argument
		Returns a nested Python Dictionary object, with the structure of the Oxcal 
		Model
	"""
    
	MakeJSON(file)
	model = ReadJSON()
    
	return model

def FillBetween(ax,dat,color,median = False,prob = 'posterior',**kwargs):
	
	s  = dat['posterior']['start']
	r = dat['posterior']['resolution']
	l = len(dat['posterior']['prob'])
	xi = np.linspace(s,s+(l-1)*r,l)
	med = dat['posterior']['median']

	ax.fill_between(xi,y1=np.zeros(l),y2 = np.array(dat[prob]['prob']),color = color,**kwargs)
	if median == True:
		ax.scatter(x =med,y=0.2,marker = "D",alpha = 1,color = 'white',edgecolor =color,zorder = 101)
		
	
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