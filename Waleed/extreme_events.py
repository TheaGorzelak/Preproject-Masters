# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:19:31 2022

@author: arsha
"""
import pandas as pd
import itertools
import matplotlib.pyplot as plt

country_codes = ['AL','AT','BA','BE','BG','BY','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LV','MD','ME','MK','MT','NL','NO','PL','PT','RO','RS','RU','SE','SI','SK','TR','UA']

wind_ensemble_BOC =  pd.read_csv("wind_ensemble_BOC_ssp585.csv",delimiter=',')
wind_ensemble_BOC = wind_ensemble_BOC.drop(["time","time.1"],axis=1)

wind_ensemble_EOC =  pd.read_csv("wind_ensemble_EOC_ssp585.csv",delimiter=',')
wind_ensemble_EOC = wind_ensemble_EOC.drop(["time","time.1"],axis=1)
countries = country_codes
def extreme_events_count(country_codes,scen):
    import numpy as np
    import itertools
    mean_dur_boc = [0]*len(country_codes)
    mean_nb_seq_boc = [0]*len(country_codes)
    mean_dur_eoc = [0]*len(country_codes)
    mean_nb_seq_eoc = [0]*len(country_codes)
    for c in range(len(country_codes)):
        infl_BOC_d = wind_ensemble_BOC
        infl_EOC_d = wind_ensemble_EOC
        infl_BOC_d_array = np.array(infl_BOC_d)
        infl_EOC_d_array = np.array(infl_EOC_d)
        if scen == 'drought':
            extreme_BOC = np.percentile(infl_BOC_d,10,axis=0)
            # extreme_BOC = pd.DataFrame(extreme_BOC).T
            
            condition = infl_EOC_d_array <= extreme_BOC
            condition = condition[:,[c]]
            condition_boc = infl_BOC_d_array <= extreme_BOC
            condition_boc = condition_boc[:,[c]]
        else:
            extreme_BOC = np.percentile(infl_BOC_d_array,90,axis=0)
            condition = infl_EOC_d_array >= extreme_BOC
            condition = condition[:,[c]]
            condition_boc = infl_BOC_d_array >= extreme_BOC
            condition_boc = condition_boc[:,[c]]

        cons_days_eoc = [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
        cons_days_boc = [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition_boc ) if key ]
        
        if scen == 'drought':
            consec_days_eoc = infl_EOC_d[infl_EOC_d < extreme_BOC]
        else:
            consec_days_eoc = infl_EOC_d[infl_EOC_d > extreme_BOC]
        cons_groups = [0]*len(cons_days_eoc)
        j = 0
        for i in range(len(cons_days_eoc)):
            if i == 0:
                cons_groups[i] = consec_days_eoc.iloc[0:cons_days_eoc[i]]
            else:
                cons_groups[i] = consec_days_eoc.iloc[j:j+cons_days_eoc[i]]
            j += cons_days_eoc[i]
        mean_dur_boc[c] = sum(cons_days_boc)/len(cons_days_boc) # Mean drought duration
        mean_nb_seq_boc[c] = len(np.array(cons_days_boc)[np.array(cons_days_boc) > 1])/len(infl_BOC_d_array)*365
        mean_dur_eoc[c] = sum(cons_days_eoc)/len(cons_days_eoc) # Mean drought duration
        mean_nb_seq_eoc[c] = len(np.array(cons_days_eoc)[np.array(cons_days_eoc) > 1])/len(infl_EOC_d_array)*365
        
    return mean_dur_boc,mean_nb_seq_boc,mean_dur_eoc,mean_nb_seq_eoc
      
def extreme_events_plot(country_codes,countries,scen,fig,ax,mean_dur_boc,mean_dur_eoc,mean_nb_seq_boc,mean_nb_seq_eoc):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import matplotlib.ticker as plticker
    import numpy as np
    from AUcolor import AUcolor
    colors,color_names = AUcolor()
    from matplotlib import font_manager as fm
    ticks_font = fm.FontProperties(fname='AUfonts/AUPassata_Rg.ttf',size=20)
    color_BOC = 'k'
    color_EOC = 'r'
    ax[0,0].scatter(-1,-1,label='EOC SSP5-8.5',zorder=1,color=color_EOC,marker='s')
    ax[0,0].scatter(-1,-1,label='BOC',zorder=1,color=color_BOC,marker='s')
    z = np.array(mean_dur_eoc)*np.array(mean_nb_seq_eoc)
    # ax[0,0].set_ylim([min(mean_nb_seq_eoc),max(mean_nb_seq_eoc)*1.4])
    # ax[0,0].set_xlim([min(mean_dur_eoc),max(mean_dur_eoc)*1.3])
    ax[0,0].set_xlim([2.08,37.54])
    ax[0,0].set_ylim([0.23,21.91])
    x1 = np.arange(0,40,0.01)
    y1 = np.arange(0,40,0.01)
    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('log')
    xv, yv = np.meshgrid(x1, y1)
    zv = xv*yv
    #zv[zv > z.max()] = z.max()
    if scen == 'drought':
        IPCCpreccolors = [(245/255,245/255,245/255),(84/255,48/255,5/255)]
        colormap = LinearSegmentedColormap.from_list(
        'IPCCprec', IPCCpreccolors, N=50)
        #colormap = plt.cm.Reds
    else:
        #colormap = plt.cm.Blues
        IPCCpreccolors = [(245/255,245/255,245/255),(0,60/255,48/255)]
        colormap = LinearSegmentedColormap.from_list(
        'IPCCprec', IPCCpreccolors, N=50)
    normalize = Normalize(vmin=0, vmax=100) #(vmin=z.min(), vmax=z.max())
    plt.contourf(xv,yv,zv,cmap=colormap,norm=normalize,levels=[10,25,50,75,100,150],zorder=0)#,levels=10)
    for i in range(len(countries)):
        ax[0,0].annotate(country_codes[i], xy=(mean_dur_boc[i],mean_nb_seq_boc[i]),color='w',
                        bbox=dict(boxstyle="circle", fc=color_BOC),fontsize=15,zorder=4)
        
        ax[0,0].annotate(country_codes[i], xy=(mean_dur_eoc[i],mean_nb_seq_eoc[i]),color='w',
                        bbox=dict(boxstyle="round", fc=color_EOC),fontsize=15,zorder=5)
        
    ax[0,0].legend(prop=ticks_font)
    #ax[0,0].grid(linestyle='-', linewidth='0.75', color='darkgrey',which='both',axis='both')  
    cbar = plt.colorbar(ax=ax[0,0],boundaries=np.linspace(0, 60, 6)) 
    cbar.ax.set_ylabel('Number of days with ' + scen + ' in a year',fontproperties=ticks_font)
    cbar.ax.tick_params(labelsize=20)
    ax[0,0].yaxis.set_minor_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    plt.setp(ax[0,0].get_yminorticklabels(), visible=False)
    #ax[0,0].tick_params(axis='both', which='minor', labelsize=20)
    #ax[0,0].set_yticks([2,3,4,5,6,7,8,9,10,20])
    #ax[0,0].set_xticks([3,4,5,6,7,8,9,10,20,30])
    # loc = plticker.MultipleLocator(base=2)
    # ax[0,0].yaxis.set_minor_locator(loc)
    # ax.xaxis.set_minor_locator(MultipleLocator(5))
    return fig

# np.count_nonzero(condition)

scen = 'drought'
mean_dur_boc,mean_nb_seq_boc,mean_dur_eoc,mean_nb_seq_eoc = extreme_events_count(country_codes,scen)    
countries = country_codes
f4, ax4 = plt.subplots(1,squeeze=False,figsize=(15,9))
f4 = extreme_events_plot(country_codes,countries,scen,f4,ax4,mean_dur_boc,mean_dur_eoc,mean_nb_seq_boc,mean_nb_seq_eoc)
ax4[0,0].grid(linestyle='-', linewidth='0.75', color='darkgrey',which='both',axis='both')   
plt.margins(0,0)
ax4[0,0].set_xticklabels([2, 5, 10, 20, 30])
ax4[0,0].set_yticklabels([0.5, 1, 2, 5, 10, 20, 30])
ax4[0,0].set_xticks([2, 5, 10, 20, 30])
ax4[0,0].set_yticks([0.5, 1, 2, 5, 10, 20, 30])
ax4[0,0].set_xlabel('Mean duration [days]', fontsize = 14, fontweight ='bold')
ax4[0,0].set_ylabel('Mean number of sequences [/year]', fontsize = 14, fontweight ='bold')
f4.savefig('Extreme_events_droughts_wind_ssp585' + '.png',bbox_inches='tight') 

# res_list = [mean_dur_eoc[i] * mean_nb_seq_eoc[i] for i in range(len(mean_nb_seq_eoc))]

scen = 'overflow'
mean_dur_boc,mean_nb_seq_boc,mean_dur_eoc,mean_nb_seq_eoc = extreme_events_count(country_codes,scen)    
countries = country_codes
f2, ax2 = plt.subplots(1,squeeze=False,figsize=(15,9))
f2 = extreme_events_plot(country_codes,countries,scen,f2,ax2,mean_dur_boc,mean_dur_eoc,mean_nb_seq_boc,mean_nb_seq_eoc)
ax2[0,0].grid(linestyle='-', linewidth='0.75', color='darkgrey',which='both',axis='both')   
plt.margins(0,0)
ax2[0,0].set_xticklabels([2, 3, 4, 6, 10,20])
ax2[0,0].set_yticklabels([0.5, 1, 2, 5, 10, 20])
ax2[0,0].set_xticks([2, 3, 4, 6, 10, 20])
ax2[0,0].set_yticks([0.5, 1, 2, 5, 10, 20])
ax2[0,0].set_xlabel('Mean duration [days]', fontsize = 14, fontweight ='bold')
ax2[0,0].set_ylabel('Mean number of sequences [/year]', fontsize = 14, fontweight ='bold')
ax2[0,0].set_xlim([2,20])
ax2[0,0].set_ylim([0,25.91])
f2.savefig('Extreme_events_overflow_wind_ssp585' + '.png',bbox_inches='tight') 
