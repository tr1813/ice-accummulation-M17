""" A module with functions to read the GNIP dataset
------------------------------------------

It contains functions for reading the WISER csv files, 
computing the Precipitation Weighted Least Squares Regression (PWLSR)
and plotting of all sorts. 

It can be loaded in a Jupyter Notebook using the code:

 %aimport GNIP

 ------------------------------------------"""

import numpy as np
import pandas as pd
import scipy.odr as odr


def ReadArchive(filename):

	""" 
	Takes the filename and returns a clean pandas dataframe.
	"""

	df = pd.read_csv(filename,
		parse_dates=True,
		usecols=[2,5,6,7,12,13,14,16,18,23,24,25], # these columns correspond to Date, Begin of Period, O18, H2, Precipitation
		index_col=["Site","Date"])
	df.dropna(how= 'all',inplace=True)

	return df

def PWLSR(df,site):

	"""
	Computes the Precipitation Weighted Least Squares Regression on a standard monthly GNIP dataset.
	Based on the formulae by Hughes and Crawford.
	Returns a function and its coefficients
	"""

	xi = df.loc[site,["O18","H2","Precipitation"]].dropna()["O18"]
	yi = df.loc[site,["O18","H2","Precipitation"]].dropna()["H2"]
	pi = df.loc[site,["O18","H2","Precipitation"]].dropna()["Precipitation"]
	n = len(xi)

	# plug into formula by Hughes and Crawford (2012), equation (9) for the slope
	k1 = (pi*xi*yi).sum()-(((pi*xi).sum())*((pi*yi).sum())/(pi.sum())) 
	k2 = (pi*xi**2).sum()-(((pi*xi).sum())**2/(pi.sum()))
	a = k1/k2

	# intercept (equation (10))
	b = (((pi*yi).sum())-a*((pi*xi).sum()))/(pi.sum())

	# S_yx_w - equation 13.
	k3 = n/(n-2)
	k4 = ((pi*yi**2).sum()-b*(pi*yi).sum()-a*(pi*yi*xi).sum())/(pi.sum())
	S_yx_w = (k3*k4)**(0.5)

	# sigma_aw (equation (11))
	k5 = n/(pi.sum())
	k6 = (pi*xi**2).sum()-(((pi*xi).sum())**2)/(pi.sum())
	sigma_aw = S_yx_w/(k5*k6)**(0.5)

	# sigma_bw (equation (12))
	k7 = ((pi*xi**2).sum())/(n*(((pi*xi**2).sum())-(((pi*xi).sum())**2)/(pi.sum())))
	sigma_bw = S_yx_w*k7**(0.5)

	# r2, equation (14)
	k8 = (((pi*xi*yi).sum())-(((pi*xi).sum())*((pi*yi).sum())/(pi.sum())))**2
	k9 = ((pi*xi**2).sum()-((pi*xi).sum())**2/(pi.sum()))*((pi*yi**2).sum()-((pi*yi).sum())**2/(pi.sum()))
	r2 = k8/k9

	# this returns a dictionary with the computed variables. 
	coeffs = dict(zip(['slope','intercept',
					'$\sigma_{a(w)}$','$\sigma_{b(w)}$',
					'std. error','$R^2$','N'],
					[a,b,
					 sigma_aw,sigma_bw,
			 S_yx_w,r2,n]))

	def f(x):
		return a*x+b

	return f,coeffs

def SlopeEstimate(xi,yi):

	"""
	returns a slope estimate for the ordinary least squares regression
	"""

	n = len(xi)
	A = (1/(n-2))*np.sum((yi-yi.mean())**2)
	B = np.sum((xi-xi.mean())**2)
	sigma_a = np.sqrt(A/B)

	return sigma_a

def GetStats(df,site):
	O18 = dataframe.loc[site]["O18"]
	H2 = dataframe.loc[site]["H2"]
	d_ex = H2-8*O18

	for i in (O18,H2,d_ex):
		
		return i.min(),i.max(),i.mean(),i.median(),i.std()

def ReadCustom(filename,**kwargs):

	"""
	Reads a custom excel file containing isotopic data from the studied ice caves
	"""

	columns=dict(zip(["cave","samplename","transect","column height (cm)","s.d. height","protocol","d18O","s.d. d18O","d2H","s.d. d2H","layer","layer type"],
		[0,1,2,3,4,5,7,8,9,10,14,15,16]))

	header=np.arange(0,15)

	df = pd.read_csv(filename,
		skiprows=header,
		usecols=[columns[i] for i in columns],
		index_col = [0,5,2,1,3])
	
	df2 = df.sort_index()
	df2["d-excess"] = df2["d2H"]-8*df2["d18O"]
	df2["s.d. d-excess"] = (df2["s.d. d2H"]**2 + df2["s.d. d18O"]**2 ) ** (1/2)

	# get rid of lines where the woody macrofossils were sampled
	df_iso =df2.query('protocol != "CARBON"').copy()
	df_iso.dropna(inplace=True)

	return df_iso

def GetODR(xi,yi):

	x = np.arange(-30,0,0.1)

	def g(B,x):
		return B[0]*x + B[1]

	linear = odr.Model(g)
	mydata = odr.Data(xi, yi, wd=1./xi.std()**2, we=1./yi.std()**2)
	myodr = odr.ODR(mydata, linear, beta0=[1., 2.])
	out = myodr.run()

	coeffs = {'slope': out.beta[0],
			  'sd_slope': out.sd_beta[0],
		  'intercept': out.beta[1],
		  'sd_intercept': out.sd_beta[1],
		  'min_xi': min(xi)-0.1, 
		  'max_xi': max(xi)+0.1}
	
	def f(x):
		y=x*coeffs["slope"]+coeffs["intercept"]
		return y

	return f,coeffs

def PlotLMWL(ax,df,site,x,y,label,posxy,**kwargs):
	
	AX = ax

	## plotting the LMWL
	# define xi
	xi = np.linspace(-14,-4,100)
	# return function and coeffs
	lmwl,coeffs = PWLSR(df,site)

	# plot xi vs yi, and label
	AX.plot(xi,lmwl(xi),lw=1.25,color = 'firebrick',label="Local Meteoric Water Line (PWLSR)")


	## plotting the data
	AX.plot(x,y,'o',color = "white",markersize=5, label =label,markeredgecolor="black")

	## plotting a linear regression on the data
	f2,coeffs2 = GetODR(x,y)
	# define xi
	x_odr = np.linspace(coeffs2["min_xi"],coeffs2["max_xi"],50)

	AX.plot(xi,f2(xi),'--',linewidth=0.6,**{"color":"black"})

	AX.text(posxy[0],posxy[1],size='smaller',
		s = "y={:.2f} ($\pm${:.2f}) x {:+.2f} ($\pm${:.2f})".format(coeffs2["slope"],
			coeffs2["sd_slope"],
			coeffs2["intercept"],
			coeffs2["sd_intercept"]),**kwargs)

	AX.text(posxy[0],posxy[1]+4,label,size='smaller',**kwargs)



	# Setting the limits
	AX.set_xlim(-14,-4)
	AX.set_ylim(-112,-20)
	# and spines visibility
	AX.spines["bottom"].set_visible(False)
	AX.spines["left"].set_visible(False)
	AX.yaxis.set_label_position("right")
	AX.yaxis.tick_right()
	AX.xaxis.set_label_position("top")
	AX.xaxis.tick_top()
	# Setting the labels
	AX.set_xlabel("$\delta^{18}$O [‰] (VSMOW)")
	AX.set_ylabel("$\delta^{2}$H [‰] (VSMOW)")
	
	return AX,coeffs,coeffs2