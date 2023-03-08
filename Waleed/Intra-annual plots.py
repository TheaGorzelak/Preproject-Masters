# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:43:23 2022

@author: arsha
"""

# ___________________________#

wind_ssp245_cp_Earth3_BOC_245_djf = wind_ssp245_cp_Earth3_BOC_245.sel(time=wind_ssp245_cp_Earth3_BOC_245.time.dt.month.isin([1, 2, 12]))

wind_ssp245_cp_Earth3_BOC_245_mam = wind_ssp245_cp_Earth3_BOC_245.sel(time=wind_ssp245_cp_Earth3_BOC_245.time.dt.month.isin([3, 4, 5]))

wind_ssp245_cp_Earth3_BOC_245_jja = wind_ssp245_cp_Earth3_BOC_245.sel(time=wind_ssp245_cp_Earth3_BOC_245.time.dt.month.isin([6, 7, 8]))

wind_ssp245_cp_Earth3_BOC_245_son = wind_ssp245_cp_Earth3_BOC_245.sel(time=wind_ssp245_cp_Earth3_BOC_245.time.dt.month.isin([9, 10, 11]))

wind_ssp245_cp_Earth3_EOC_245_djf = wind_ssp245_cp_Earth3_EOC_245.sel(time=wind_ssp245_cp_Earth3_EOC_245.time.dt.month.isin([1, 2, 12]))

wind_ssp245_cp_Earth3_EOC_245_mam = wind_ssp245_cp_Earth3_EOC_245.sel(time=wind_ssp245_cp_Earth3_EOC_245.time.dt.month.isin([3, 4, 5]))

wind_ssp245_cp_Earth3_EOC_245_jja = wind_ssp245_cp_Earth3_EOC_245.sel(time=wind_ssp245_cp_Earth3_EOC_245.time.dt.month.isin([6, 7, 8]))

wind_ssp245_cp_Earth3_EOC_245_son = wind_ssp245_cp_Earth3_EOC_245.sel(time=wind_ssp245_cp_Earth3_EOC_245.time.dt.month.isin([9, 10, 11]))

# ___________________________#

wind_ssp245_cp_ESM2_BOC_245_djf = wind_ssp245_cp_ESM2_BOC_245.sel(time=wind_ssp245_cp_ESM2_BOC_245.time.dt.month.isin([1, 2, 12]))

wind_ssp245_cp_ESM2_BOC_245_mam = wind_ssp245_cp_ESM2_BOC_245.sel(time=wind_ssp245_cp_ESM2_BOC_245.time.dt.month.isin([3, 4, 5]))

wind_ssp245_cp_ESM2_BOC_245_jja = wind_ssp245_cp_ESM2_BOC_245.sel(time=wind_ssp245_cp_ESM2_BOC_245.time.dt.month.isin([6, 7, 8]))

wind_ssp245_cp_ESM2_BOC_245_son = wind_ssp245_cp_ESM2_BOC_245.sel(time=wind_ssp245_cp_ESM2_BOC_245.time.dt.month.isin([9, 10, 11]))

wind_ssp245_cp_ESM2_EOC_245_djf = wind_ssp245_cp_ESM2_EOC_245.sel(time=wind_ssp245_cp_ESM2_EOC_245.time.dt.month.isin([1, 2, 12]))

wind_ssp245_cp_ESM2_EOC_245_mam = wind_ssp245_cp_ESM2_EOC_245.sel(time=wind_ssp245_cp_ESM2_EOC_245.time.dt.month.isin([3, 4, 5]))

wind_ssp245_cp_ESM2_EOC_245_jja = wind_ssp245_cp_ESM2_EOC_245.sel(time=wind_ssp245_cp_ESM2_EOC_245.time.dt.month.isin([6, 7, 8]))

wind_ssp245_cp_ESM2_EOC_245_son = wind_ssp245_cp_ESM2_EOC_245.sel(time=wind_ssp245_cp_ESM2_EOC_245.time.dt.month.isin([9, 10, 11]))

# ___________________________#

wind_ssp245_cp_CM2_BOC_245_djf = wind_ssp245_cp_CM2_BOC_245.sel(time=wind_ssp245_cp_CM2_BOC_245.time.dt.month.isin([1, 2, 12]))

wind_ssp245_cp_CM2_BOC_245_mam = wind_ssp245_cp_CM2_BOC_245.sel(time=wind_ssp245_cp_CM2_BOC_245.time.dt.month.isin([3, 4, 5]))

wind_ssp245_cp_CM2_BOC_245_jja = wind_ssp245_cp_CM2_BOC_245.sel(time=wind_ssp245_cp_CM2_BOC_245.time.dt.month.isin([6, 7, 8]))

wind_ssp245_cp_CM2_BOC_245_son = wind_ssp245_cp_CM2_BOC_245.sel(time=wind_ssp245_cp_CM2_BOC_245.time.dt.month.isin([9, 10, 11]))

wind_ssp245_cp_CM2_EOC_245_djf = wind_ssp245_cp_CM2_EOC_245.sel(time=wind_ssp245_cp_CM2_EOC_245.time.dt.month.isin([1, 2, 12]))

wind_ssp245_cp_CM2_EOC_245_mam = wind_ssp245_cp_CM2_EOC_245.sel(time=wind_ssp245_cp_CM2_EOC_245.time.dt.month.isin([3, 4, 5]))

wind_ssp245_cp_CM2_EOC_245_jja = wind_ssp245_cp_CM2_EOC_245.sel(time=wind_ssp245_cp_CM2_EOC_245.time.dt.month.isin([6, 7, 8]))

wind_ssp245_cp_CM2_EOC_245_son = wind_ssp245_cp_CM2_EOC_245.sel(time=wind_ssp245_cp_CM2_EOC_245.time.dt.month.isin([9, 10, 11]))

# ___________________________#
#Percentage intra-annual plot

wind_ssp245_cp_ensemble_BOC_245_djf = (wind_ssp245_cp_CM2_BOC_245_djf+wind_ssp245_cp_ESM2_BOC_245_djf+wind_ssp245_cp_Earth3_BOC_245_djf)/3

wind_ssp245_cp_ensemble_BOC_245_mam = (wind_ssp245_cp_CM2_BOC_245_mam+wind_ssp245_cp_ESM2_BOC_245_mam+wind_ssp245_cp_Earth3_BOC_245_mam)/3

wind_ssp245_cp_ensemble_BOC_245_jja = (wind_ssp245_cp_CM2_BOC_245_jja+wind_ssp245_cp_ESM2_BOC_245_jja+wind_ssp245_cp_Earth3_BOC_245_jja)/3

wind_ssp245_cp_ensemble_BOC_245_son = (wind_ssp245_cp_CM2_BOC_245_son+wind_ssp245_cp_ESM2_BOC_245_son+wind_ssp245_cp_Earth3_BOC_245_son)/3

wind_ssp245_cp_ensemble_EOC_245_djf = (wind_ssp245_cp_CM2_EOC_245_djf+wind_ssp245_cp_ESM2_EOC_245_djf+wind_ssp245_cp_Earth3_EOC_245_djf)/3

wind_ssp245_cp_ensemble_EOC_245_mam = (wind_ssp245_cp_CM2_EOC_245_mam+wind_ssp245_cp_ESM2_EOC_245_mam+wind_ssp245_cp_Earth3_EOC_245_mam)/3

wind_ssp245_cp_ensemble_EOC_245_jja = (wind_ssp245_cp_CM2_EOC_245_jja+wind_ssp245_cp_ESM2_EOC_245_jja+wind_ssp245_cp_Earth3_EOC_245_jja)/3

wind_ssp245_cp_ensemble_EOC_245_son = (wind_ssp245_cp_CM2_EOC_245_son+wind_ssp245_cp_ESM2_EOC_245_son+wind_ssp245_cp_Earth3_EOC_245_son)/3


wind_ssp245_cp_ensemble_BOC_245_djf = (wind_ssp245_cp_CM2_BOC_245_djf+wind_ssp245_cp_ESM2_BOC_245_djf)/2

wind_ssp245_cp_ensemble_BOC_245_mam = (wind_ssp245_cp_CM2_BOC_245_mam+wind_ssp245_cp_ESM2_BOC_245_mam)/2

wind_ssp245_cp_ensemble_BOC_245_jja = (wind_ssp245_cp_CM2_BOC_245_jja+wind_ssp245_cp_ESM2_BOC_245_jja)/2

wind_ssp245_cp_ensemble_BOC_245_son = (wind_ssp245_cp_CM2_BOC_245_son+wind_ssp245_cp_ESM2_BOC_245_son)/2

wind_ssp245_cp_ensemble_EOC_245_djf = (wind_ssp245_cp_CM2_EOC_245_djf+wind_ssp245_cp_ESM2_EOC_245_djf)/2

wind_ssp245_cp_ensemble_EOC_245_mam = (wind_ssp245_cp_CM2_EOC_245_mam+wind_ssp245_cp_ESM2_EOC_245_mam)/2

wind_ssp245_cp_ensemble_EOC_245_jja = (wind_ssp245_cp_CM2_EOC_245_jja+wind_ssp245_cp_ESM2_EOC_245_jja)/2

wind_ssp245_cp_ensemble_EOC_245_son = (wind_ssp245_cp_CM2_EOC_245_son+wind_ssp245_cp_ESM2_EOC_245_son)/2


wind_ssp245_cp_ensemble_BOC_245_djf = wind_ssp245_cp_Earth3_BOC_245_djf

wind_ssp245_cp_ensemble_BOC_245_mam = wind_ssp245_cp_Earth3_BOC_245_mam

wind_ssp245_cp_ensemble_BOC_245_jja = wind_ssp245_cp_Earth3_BOC_245_jja

wind_ssp245_cp_ensemble_BOC_245_son = wind_ssp245_cp_Earth3_BOC_245_son

wind_ssp245_cp_ensemble_EOC_245_djf = wind_ssp245_cp_Earth3_EOC_245_djf

wind_ssp245_cp_ensemble_EOC_245_mam = wind_ssp245_cp_Earth3_EOC_245_mam

wind_ssp245_cp_ensemble_EOC_245_jja = wind_ssp245_cp_Earth3_EOC_245_jja

wind_ssp245_cp_ensemble_EOC_245_son = wind_ssp245_cp_Earth3_EOC_245_son


cmap = mpl.cm.coolwarm
norm = mpl.colors.Normalize(vmin=-20, vmax=20)
normBOC = mpl.colors.Normalize(vmin=0.06, vmax=0.14)
                            
fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6), (ax7, ax8)) = plt.subplots(4,2,subplot_kw=dict(projection=ccrs.PlateCarree()),figsize=(8,16))
fig.subplots_adjust(hspace=0.2)


europe['mean_cf_wind_ensemble_BOC_245_djf'] = wind_ssp245_cp_ensemble_BOC_245_djf.mean(dim='time')
europe.plot(column='mean_cf_wind_ensemble_BOC_245_djf', ax=ax1,  legend=True)
ax1.set_title('Ensemble BOC DJF')


europe['mean_cf_wind_ensemble_EOC_245_djf'] = wind_ssp245_cp_ensemble_EOC_245_djf.mean(dim='time')
europe['mean_cf_winddelta_ensemble_djf']=(europe['mean_cf_wind_ensemble_EOC_245_djf']-europe['mean_cf_wind_ensemble_BOC_245_djf'])*100/europe['mean_cf_wind_ensemble_BOC_245_djf']
europe.plot(column='mean_cf_winddelta_ensemble_djf', ax=ax2,cmap=cmap,norm=norm, legend=True)
ax2.set_title('Ensemble EOC DJF')

europe['mean_cf_wind_ensemble_BOC_245_mam'] = wind_ssp245_cp_ensemble_BOC_245_mam.mean(dim='time')
europe.plot(column='mean_cf_wind_ensemble_BOC_245_mam', ax=ax3,  legend=True)
ax3.set_title('Ensemble BOC mam')


europe['mean_cf_wind_ensemble_EOC_245_mam'] = wind_ssp245_cp_ensemble_EOC_245_mam.mean(dim='time')
europe['mean_cf_winddelta_ensemble_mam']=(europe['mean_cf_wind_ensemble_EOC_245_mam']-europe['mean_cf_wind_ensemble_BOC_245_mam'])*100/europe['mean_cf_wind_ensemble_BOC_245_mam']
europe.plot(column='mean_cf_winddelta_ensemble_mam', ax=ax4,cmap=cmap,norm=norm, legend=True)
ax4.set_title('Ensemble EOC mam')

europe['mean_cf_wind_ensemble_BOC_245_jja'] = wind_ssp245_cp_ensemble_BOC_245_jja.mean(dim='time')
europe.plot(column='mean_cf_wind_ensemble_BOC_245_jja', ax=ax5,  legend=True)
ax5.set_title('Ensemble BOC jja')


europe['mean_cf_wind_ensemble_EOC_245_jja'] = wind_ssp245_cp_ensemble_EOC_245_jja.mean(dim='time')
europe['mean_cf_winddelta_ensemble_jja']=(europe['mean_cf_wind_ensemble_EOC_245_jja']-europe['mean_cf_wind_ensemble_BOC_245_jja'])*100/europe['mean_cf_wind_ensemble_BOC_245_jja']
europe.plot(column='mean_cf_winddelta_ensemble_jja', ax=ax6,cmap=cmap,norm=norm, legend=True)
ax6.set_title('Ensemble EOC jja')

europe['mean_cf_wind_ensemble_BOC_245_son'] = wind_ssp245_cp_ensemble_BOC_245_son.mean(dim='time')
europe.plot(column='mean_cf_wind_ensemble_BOC_245_son', ax=ax7,  legend=True)
ax7.set_title('Ensemble BOC son')


europe['mean_cf_wind_ensemble_EOC_245_son'] = wind_ssp245_cp_ensemble_EOC_245_son.mean(dim='time')
europe['mean_cf_winddelta_ensemble_son']=(europe['mean_cf_wind_ensemble_EOC_245_son']-europe['mean_cf_wind_ensemble_BOC_245_son'])*100/europe['mean_cf_wind_ensemble_BOC_245_son']
europe.plot(column='mean_cf_winddelta_ensemble_son', ax=ax8,cmap=cmap,norm=norm, legend=True)
ax8.set_title('Ensemble EOC son')

plt.show()


cmap = mpl.cm.coolwarm
norm = mpl.colors.Normalize(vmin=-30, vmax=30)
normBOC = mpl.colors.Normalize(vmin=0.06, vmax=0.14)
                            
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,subplot_kw=dict(projection=ccrs.PlateCarree()),figsize=(8,16))
fig.subplots_adjust(hspace=-0.3)


europe['mean_cf_wind_ensemble_EOC_245_djf'] = wind_ssp245_cp_ensemble_EOC_245_djf.mean(dim='time')
europe['mean_cf_winddelta_ensemble_djf']=(europe['mean_cf_wind_ensemble_EOC_245_djf']-europe['mean_cf_wind_ensemble_BOC_245_djf'])*100/europe['mean_cf_wind_ensemble_BOC_245_djf']
europe.plot(column='mean_cf_winddelta_ensemble_djf', ax=ax1,cmap=cmap,norm=norm, legend=False)
ax1.set_title('Winter')


europe['mean_cf_wind_ensemble_EOC_245_mam'] = wind_ssp245_cp_ensemble_EOC_245_mam.mean(dim='time')
europe['mean_cf_winddelta_ensemble_mam']=(europe['mean_cf_wind_ensemble_EOC_245_mam']-europe['mean_cf_wind_ensemble_BOC_245_mam'])*100/europe['mean_cf_wind_ensemble_BOC_245_mam']
europe.plot(column='mean_cf_winddelta_ensemble_mam', ax=ax2,cmap=cmap,norm=norm, legend=False)
ax2.set_title('Spring')

europe['mean_cf_wind_ensemble_EOC_245_jja'] = wind_ssp245_cp_ensemble_EOC_245_jja.mean(dim='time')
europe['mean_cf_winddelta_ensemble_jja']=(europe['mean_cf_wind_ensemble_EOC_245_jja']-europe['mean_cf_wind_ensemble_BOC_245_jja'])*100/europe['mean_cf_wind_ensemble_BOC_245_jja']
europe.plot(column='mean_cf_winddelta_ensemble_jja', ax=ax3,cmap=cmap,norm=norm, legend=False)
ax3.set_title('Summer')


europe['mean_cf_wind_ensemble_EOC_245_son'] = wind_ssp245_cp_ensemble_EOC_245_son.mean(dim='time')
europe['mean_cf_winddelta_ensemble_son']=(europe['mean_cf_wind_ensemble_EOC_245_son']-europe['mean_cf_wind_ensemble_BOC_245_son'])*100/europe['mean_cf_wind_ensemble_BOC_245_son']
europe.plot(column='mean_cf_winddelta_ensemble_son', ax=ax4,cmap=cmap,norm=norm, legend=False)
ax4.set_title('Autumn')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.3, 0.18, 0.4, 0.02])
# Draw the colorbar
cbar=fig.colorbar(sm, cax=cbar_ax,ticks=[-30, -20, -10, 0, 10, 20, 30],orientation='horizontal')
cbar.ax.set_xticklabels(['-30%','-20%', '-10%', '0%', '10%', '20%','30%'])
fig.suptitle('SSP2-4.5',y=0.82,weight='bold')
plt.show()