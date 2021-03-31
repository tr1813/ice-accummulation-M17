import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
colors = sns.palettes.color_palette('colorblind')

# plot of relative humidity - temperature - temperature variation.

def CavePlot(t_in,t_out,rh_in,s=None,e=None,path= None,offset = 0,o_c = None,**kwargs):
    '''function for plotting T, RH
    
    t_in, t_out, rh_in: identically indexed DataFrames or Series
    the index should be a datetime index
    s,e: Datetime indices of format "YYYY-MM-DD"
    
    returns a figure
    '''

    # check the longest record.
    resampler_1D = t_in[s:e].rolling('D')# make  two-daily resmpler object
    var_1D = resampler_1D.std().to_numpy()
    
    ## figure
    fig, axT = plt.subplots(figsize=(13,7))

    ## plot temperature
    axT.plot(t_out[s:e].index.to_numpy(), t_out[s:e].to_numpy() -offset,color = 'lightgray',lw = 0.5, label = 'T$_{vogel_{corr}}$ (°C)')
    axT.plot(t_in[s:e].index.to_numpy(),t_in[s:e].to_numpy(),label = 'T$_{L2}$ (°C)')
    
    ## plot relative humidity
    axRH = axT.twinx()
    axRH.plot(rh_in[s:e].index.to_numpy(),rh_in[s:e].to_numpy(),color = colors[2],label = 'RH$_{L2}$ (%)')


    ## plot the criterion fields
        
    # calculate where to plot the criterion fields
    t_min,t_max = round(t_out.min()-2),round(t_out.max()+1)
    k = (t_max-t_min) / 30 # five increments covering a sixth of the plot extent.
    
    c1 = t_out[s:e].to_numpy().astype(float) + offset <t_in[s:e]
    c1b = t_in[s:e].to_numpy() < 0
    c2 = rh_in[s:e].to_numpy() < 98
    c3 = (var_1D > 0.25)
    
    axT.fill_between(y1=t_min-k,y2=t_min,x=t_out[s:e].index.to_numpy(),where=c1,label = '(a) T$_{vogel_{corr}}$ < T$_{L2}$')
    
    x1 = pd.to_datetime("14-08-2018")
    axT.text(x= x1, y= t_min-k/2,s = '(a)', color = colors[0])
    axT.text(x= x1, y= t_min-5*k/2,s = '(b)', color = colors[3])
    axT.text(x= x1, y= t_min-9*k/2,s = '(c)',color = colors[2] )
    axT.fill_between(y1=t_min-3*k,y2=t_min-2*k,x=resampler_1D.max().index.to_numpy(),where=c3, label = '(b) daily $\sigma_{T_{L2}}$ >0.25°C',color = colors[3])
    axT.fill_between(y1=t_min-5*k,y2=t_min-4*k,x=t_out[s:e].index.to_numpy(),where=c2, label = '(c) RH < 98%',color = colors[2])


    

    # plot a horizontal line at 0
    axT.axhline(0, color = 'r',ls = '--',lw = 0.5)
    
    # tidy up axT
    axT.set_xlim(s,e)
    axT.set_ylim(t_min-8*k,t_max)
    axT.set_xlabel("Date")
    axT.set_ylabel("Hourly air temperature (°C)",position=(0,0.64))
    axT.set_yticks([-10,-5,0,5,10,15,20,25])


    # plot open and closed
    for date,stat in o_c.items():
        s = pd.to_datetime(stat[1])
        e = pd.to_datetime(stat[2])


        if stat[0] == "closed":
            returnFullRect(axT,s,e,mid = pd.to_datetime(date),y = t_min-9,l = stat[3] ,**{'color':colors[7]})
            axT.text(pd.to_datetime(date),t_min-11,s =stat[0], ha = stat[4],va ='bottom',color=colors[7]) 

        else:
            returnFullRect(axT,s,e,mid =pd.to_datetime(date),y =t_min-9, l = stat[3], **{'color':colors[9]})
            axT.text(pd.to_datetime(date),t_min-11,s =stat[0], ha = stat[4],va ='bottom',color=colors[9]) 

    # tidy up axRH
    axRH.set_ylim(30,100)
    axRH.set_yticks([60,70,80,90,100])
    axRH.set_ylabel("Hourly relative humidity (%)",color = colors[2],position=(1,0.72))
    axRH.yaxis.label.set_color(colors[2])
    axRH.tick_params(axis='y', colors=colors[2])

    # gather up the legend handles and labels from axT and axRH
    handles,labels = [(pp+t) for pp,t in zip(axRH.get_legend_handles_labels(),axT.get_legend_handles_labels())]
    
    # make it so it appears on the top left corner, out of the plotting area
    axRH.legend(handles,labels,bbox_to_anchor=(0., 1.02, 1., .102),ncol = 2)
 
    plt.tight_layout()
    
    # save the figure if you want, passing additional keyword arguments as needed
    if path is not None:
        plt.savefig(path,**kwargs)

def returnFullRect(ax,s,e,style = 'full',y =0,l =13, mid=0,**kwargs):
    
    start = pd.Timestamp(s)
    end = pd.Timestamp(e)
    
    t = np.linspace(start.value, end.value, 60)
    t = pd.to_datetime(t)
    
    y1 = y-0.5*np.exp(-((mid-t)/pd.Timedelta("{} days".format(l)))**2)
    y2 = y+0.5*np.exp(-((mid-t)/pd.Timedelta("{} days".format(l)))**2)
    ax.fill_between(x = t,y1 = y1,y2 = y2,**kwargs)
        

    return ax,t
