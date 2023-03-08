# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 08:53:19 2022

@author: arsha
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Solar

cf_ssp585_pv_ESM2_BOC_585 = pv_ssp585_cp_ESM2_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_ESM2_EOC_585 = pv_ssp585_cp_ESM2_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_CM2_BOC_585 = pv_ssp585_cp_CM2_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_CM2_EOC_585 = pv_ssp585_cp_CM2_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_Earth3_BOC_585 = pv_ssp585_cp_Earth3_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_Earth3_EOC_585 = pv_ssp585_cp_Earth3_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_MPI_BOC_585 = pv_ssp585_cp_MPI_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_MPI_EOC_585 = pv_ssp585_cp_MPI_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_MIROC6_BOC_585 = pv_ssp585_cp_MIROC6_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('D').mean().fillna(0)

cf_ssp585_pv_MIROC6_EOC_585 = pv_ssp585_cp_MIROC6_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('D').mean().fillna(0)


# Drop 29th Feb

cf_ssp585_pv_ESM2_BOC_585 = cf_ssp585_pv_ESM2_BOC_585[~((cf_ssp585_pv_ESM2_BOC_585.index.month == 2) & (cf_ssp585_pv_ESM2_BOC_585.index.day == 29))]

cf_ssp585_pv_ESM2_EOC_585 = cf_ssp585_pv_ESM2_EOC_585[~((cf_ssp585_pv_ESM2_EOC_585.index.month == 2) & (cf_ssp585_pv_ESM2_EOC_585.index.day == 29))]

cf_ssp585_pv_CM2_BOC_585 = cf_ssp585_pv_CM2_BOC_585[~((cf_ssp585_pv_CM2_BOC_585.index.month == 2) & (cf_ssp585_pv_CM2_BOC_585.index.day == 29))]

cf_ssp585_pv_CM2_EOC_585 = cf_ssp585_pv_CM2_EOC_585[~((cf_ssp585_pv_CM2_EOC_585.index.month == 2) & (cf_ssp585_pv_CM2_EOC_585.index.day == 29))]

cf_ssp585_pv_Earth3_BOC_585 = cf_ssp585_pv_Earth3_BOC_585[~((cf_ssp585_pv_Earth3_BOC_585.index.month == 2) & (cf_ssp585_pv_Earth3_BOC_585.index.day == 29))]

cf_ssp585_pv_Earth3_EOC_585 = cf_ssp585_pv_Earth3_EOC_585[~((cf_ssp585_pv_Earth3_EOC_585.index.month == 2) & (cf_ssp585_pv_Earth3_EOC_585.index.day == 29))]

cf_ssp585_pv_MPI_BOC_585 = cf_ssp585_pv_MPI_BOC_585[~((cf_ssp585_pv_MPI_BOC_585.index.month == 2) & (cf_ssp585_pv_MPI_BOC_585.index.day == 29))]

cf_ssp585_pv_MPI_EOC_585 = cf_ssp585_pv_MPI_EOC_585[~((cf_ssp585_pv_MPI_EOC_585.index.month == 2) & (cf_ssp585_pv_MPI_EOC_585.index.day == 29))]

cf_ssp585_pv_MIROC6_BOC_585 = cf_ssp585_pv_MIROC6_BOC_585[~((cf_ssp585_pv_MIROC6_BOC_585.index.month == 2) & (cf_ssp585_pv_MIROC6_BOC_585.index.day == 29))]

cf_ssp585_pv_MIROC6_EOC_585 = cf_ssp585_pv_MIROC6_EOC_585[~((cf_ssp585_pv_MIROC6_EOC_585.index.month == 2) & (cf_ssp585_pv_MIROC6_EOC_585.index.day == 29))]


fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharey = False,figsize=(12,6))
fig.subplots_adjust(hspace=0.2)

sns.histplot(data=cf_ssp585_pv_ESM2_BOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_ESM2_EOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_CM2_BOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_Earth3_BOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MPI_BOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MIROC6_BOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_CM2_EOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_Earth3_EOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MPI_EOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MIROC6_EOC_585['ES']*1.33, ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})

# Set title and labels for axes
ax1.set(xlabel="Capacity Factor",
       ylabel="PDF",
       title="Spain",
       )
       
sns.histplot(data=cf_ssp585_pv_ESM2_BOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_CM2_BOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_Earth3_BOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MPI_BOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MIROC6_BOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_ESM2_EOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_CM2_EOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_Earth3_EOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MPI_EOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MIROC6_EOC_585['NO']*1.4, ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})

# Set title and labels for axes
ax2.set(xlabel="Capacity Factor",
       ylabel="PDF",
       title="Norway",
       )

sns.histplot(data=cf_ssp585_pv_ESM2_BOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_CM2_BOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_Earth3_BOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MPI_BOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MIROC6_BOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_ESM2_EOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_CM2_EOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_Earth3_EOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MPI_EOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_pv_MIROC6_EOC_585['DE']*1.13, ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})

# Set title and labels for axes
ax3.set(xlabel="Capacity Factor",
       ylabel="PDF",
       title="Germany",
       )

fig.legend(['BOC','EOC'],frameon=False,loc='lower center',ncol=2,bbox_to_anchor=(0.5, -0.04),fancybox=True)
plt.tight_layout()
plt.rcParams['patch.linewidth'] = 0
plt.show()


# Wind

cf_ssp585_wind_ESM2_BOC_585 = wind_ssp585_cp_ESM2_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_ESM2_EOC_585 = wind_ssp585_cp_ESM2_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_CM2_BOC_585 = wind_ssp585_cp_CM2_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_CM2_EOC_585 = wind_ssp585_cp_CM2_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_Earth3_BOC_585 = wind_ssp585_cp_Earth3_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_Earth3_EOC_585 = wind_ssp585_cp_Earth3_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_MPI_BOC_585 = wind_ssp585_cp_MPI_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_MPI_EOC_585 = wind_ssp585_cp_MPI_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_MIROC6_BOC_585 = wind_ssp585_cp_MIROC6_BOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2000-01-01':'2029-12-31'].resample('Y').mean().fillna(0)

cf_ssp585_wind_MIROC6_EOC_585 = wind_ssp585_cp_MIROC6_EOC_585.sel(dim_0=['DE','ES','NO','GB']).to_series().unstack()['2070-01-01':'2099-12-31'].resample('Y').mean().fillna(0)

# Drop 29th Feb

cf_ssp585_wind_ESM2_BOC_585 = cf_ssp585_wind_ESM2_BOC_585[~((cf_ssp585_wind_ESM2_BOC_585.index.month == 2) & (cf_ssp585_wind_ESM2_BOC_585.index.day == 29))]

cf_ssp585_wind_ESM2_EOC_585 = cf_ssp585_wind_ESM2_EOC_585[~((cf_ssp585_wind_ESM2_EOC_585.index.month == 2) & (cf_ssp585_wind_ESM2_EOC_585.index.day == 29))]

cf_ssp585_wind_CM2_BOC_585 = cf_ssp585_wind_CM2_BOC_585[~((cf_ssp585_wind_CM2_BOC_585.index.month == 2) & (cf_ssp585_wind_CM2_BOC_585.index.day == 29))]

cf_ssp585_wind_CM2_EOC_585 = cf_ssp585_wind_CM2_EOC_585[~((cf_ssp585_wind_CM2_EOC_585.index.month == 2) & (cf_ssp585_wind_CM2_EOC_585.index.day == 29))]

cf_ssp585_wind_Earth3_BOC_585 = cf_ssp585_wind_Earth3_BOC_585[~((cf_ssp585_wind_Earth3_BOC_585.index.month == 2) & (cf_ssp585_wind_Earth3_BOC_585.index.day == 29))]

cf_ssp585_wind_Earth3_EOC_585 = cf_ssp585_wind_Earth3_EOC_585[~((cf_ssp585_wind_Earth3_EOC_585.index.month == 2) & (cf_ssp585_wind_Earth3_EOC_585.index.day == 29))]

cf_ssp585_wind_MPI_BOC_585 = cf_ssp585_wind_MPI_BOC_585[~((cf_ssp585_wind_MPI_BOC_585.index.month == 2) & (cf_ssp585_wind_MPI_BOC_585.index.day == 29))]

cf_ssp585_wind_MPI_EOC_585 = cf_ssp585_wind_MPI_EOC_585[~((cf_ssp585_wind_MPI_EOC_585.index.month == 2) & (cf_ssp585_wind_MPI_EOC_585.index.day == 29))]

cf_ssp585_wind_MIROC6_BOC_585 = cf_ssp585_wind_MIROC6_BOC_585[~((cf_ssp585_wind_MIROC6_BOC_585.index.month == 2) & (cf_ssp585_wind_MIROC6_BOC_585.index.day == 29))]

cf_ssp585_wind_MIROC6_EOC_585 = cf_ssp585_wind_MIROC6_EOC_585[~((cf_ssp585_wind_MIROC6_EOC_585.index.month == 2) & (cf_ssp585_wind_MIROC6_EOC_585.index.day == 29))]


fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharey = False,figsize=(12,6))
fig.subplots_adjust(hspace=0.2)

sns.histplot(data=cf_ssp585_wind_ESM2_BOC_585['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_ESM2_EOC_585['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_CM2_BOC_585['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_Earth3_BOC_585['ES']*0.6, ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MPI_BOC_585['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MIROC6_BOC_585['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_CM2_EOC_585['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_Earth3_EOC_585['ES']*0.6, ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MPI_EOC_585['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MIROC6_EOC_585['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})

# Set title and labels for axes
ax1.set(xlabel="Capacity Factor",
       ylabel="PDF",
       title="Spain",
       )
       
sns.histplot(data=cf_ssp585_wind_ESM2_BOC_585['NO'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_CM2_BOC_585['NO'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_Earth3_BOC_585['NO']*0.83, ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MPI_BOC_585['NO'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MIROC6_BOC_585['NO'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_ESM2_EOC_585['NO'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_CM2_EOC_585['NO'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_Earth3_EOC_585['NO']*0.83, ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MPI_EOC_585['NO'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MIROC6_EOC_585['NO'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})

# Set title and labels for axes
ax2.set(xlabel="Capacity Factor",
       ylabel="PDF",
       title="Norway",
       )


sns.histplot(data=cf_ssp585_wind_ESM2_BOC_585['DE'], ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_ESM2_EOC_585['DE'], ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_CM2_BOC_585['DE'], ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_Earth3_BOC_585['DE']*0.64, ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MPI_BOC_585['DE'], ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MIROC6_BOC_585['DE'], ax=ax3,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_CM2_EOC_585['DE'], ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_ssp585_wind_Earth3_EOC_585['DE']*0.64, ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MPI_EOC_585['DE'], ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_ssp585_wind_MIROC6_EOC_585['DE'], ax=ax3,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})

# Set title and labels for axes
ax3.set(xlabel="Capacity Factor",
       ylabel="PDF",
       title="Germany",
       )

fig.legend(['BOC','EOC'],frameon=False,loc='lower center',ncol=2,bbox_to_anchor=(0.5, -0.04),fancybox=True)
plt.tight_layout()
plt.rcParams['patch.linewidth'] = 0
plt.show()


# Hydro

cf_hydro_ESM2_BOC_585_df = cf_hydro_ESM2_BOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_CM2_BOC_585_df = cf_hydro_CM2_BOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_Earth3_BOC_585_df = cf_hydro_Earth3_BOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_MPI_BOC_585_df = cf_hydro_MPI_BOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_MIROC6_BOC_585_df = cf_hydro_MIROC6_BOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_ESM2_EOC_585_df = cf_hydro_ESM2_EOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_CM2_EOC_585_df = cf_hydro_CM2_EOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_Earth3_EOC_585_df = cf_hydro_Earth3_EOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_MPI_EOC_585_df = cf_hydro_MPI_EOC_585_df.resample('Y').mean().fillna(0)

cf_hydro_MIROC6_EOC_585_df = cf_hydro_MIROC6_EOC_585_df.resample('Y').mean().fillna(0)

fig, (ax1, ax2) = plt.subplots(1,2,sharey = False,figsize=(12,6))
fig.subplots_adjust(hspace=0.2)

# sns.histplot(data=cf_hydro_ESM2_BOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_hydro_ESM2_EOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_hydro_CM2_BOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_Earth3_BOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_Earth3_EOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_MPI_BOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_MIROC6_BOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_hydro_CM2_EOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_MPI_EOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_MIROC6_EOC_585_df['ES'], ax=ax1,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})

# Set title and labels for axes
ax1.set(xlabel="Runoff",
       ylabel="PDF",
       title="Spain",
       )
       
# sns.histplot(data=cf_hydro_ESM2_BOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_hydro_CM2_BOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_Earth3_BOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_MPI_BOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_MIROC6_BOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='royalblue', line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_hydro_ESM2_EOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
# sns.histplot(data=cf_hydro_CM2_EOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_Earth3_EOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_MPI_EOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})
sns.histplot(data=cf_hydro_MIROC6_EOC_585_df['DE'], ax=ax2,stat='probability',kde=True, color='orange', alpha=0.3,line_kws={'lw': 1.5},kde_kws={'cut': 5})

# Set title and labels for axes
ax2.set(xlabel="Runoff",
       ylabel="PDF",
       title="Germany",
       )

fig.suptitle('Hydro Analysis',y=0.95,weight='bold')
fig.legend(['BOC','EOC'],frameon=False,loc='lower center',ncol=2)
plt.tight_layout()
plt.rcParams['patch.linewidth'] = 0
plt.show()
