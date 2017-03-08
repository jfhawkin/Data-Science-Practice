# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:44:04 2016

@author: jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

birddata = pd.read_csv('bird_tracking.csv')
bird_names = pd.unique(birddata.bird_name)
plt.figure(figsize=(7,7))
for bird_name in bird_names:
    ix = birddata.bird_name==bird_name
    x,y = birddata.longitude[ix],birddata.latitude[ix]
    plt.plot(x,y,".",label=bird_name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="lower right")
plt.savefig("3trj.pdf")

plt.figure(figsize=(8,4))
ix = birddata.bird_name=="Eric"
speed = birddata.speed_2d[ix]
ind = np.isnan(speed)
plt.hist(speed[~ind], bins=np.linspace(0,30,20),normed=True)
plt.xlabel("2D speed (m/s)")
plt.ylabel("Frequency")
plt.savefig("hist.pdf")

birddata.speed_2d.plot(kind="hist",range=[0,30])
plt.xlabel("2D speed")
plt.savefig("pd_hist.pdf")

import datetime
date_str=birddata.date_time[0]
datetime.datetime.strptime(date_str[:-3],'%Y-%m-%d %H:%M:%S')

timestamp = []
for k in range(len(birddata)):
    print()
    timestamp.append(datetime.datetime.strptime\
    (birddata.date_time.iloc[k][:-3],'%Y-%m-%d %H:%M:%S'))
    
birddata["timestamp"] = pd.Series(timestamp,index=birddata.index)

data = birddata[birddata.bird_name=="Eric"]
times = birddata.timestamp[birddata.bird_name=="Eric"]
elapsed_time = [time-times[0] for time in times]

plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1))
plt.xlabel("Observation")
plt.ylabel("Elapsed Time (days)")
plt.savefig("timeplot.pdf")


data = birddata[birddata.bird_name=="Eric"]
times = data.timestamp
elapsed_time = [time-times[0] for time in times]
elapsed_days = np.array(elapsed_time)/datetime.timedelta(days=1)
next_day = 1
inds = []
daily_mean_speed=[]
for (i,t) in enumerate(elapsed_days):
    if t<next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day+=1
        inds=[]

plt.figure(figsize=(8,6))
plt.plot(daily_mean_speed)
plt.xlabel("Day")
plt.ylabel("Mean Speed (m/s)")

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()
plt.figure(figsize=(10,10))
ax = plt.axes(projection=proj)
ax.set_extent((-25.0,20.0,52.0,10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS,linestyle=':')
for name in bird_names:
    ix = birddata["bird_name"] == name
    x,y = birddata.longitude[ix],birddata.latitude[ix]
    ax.plot(x,y,".",transform=ccrs.Geodetic(), label=name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="upper left")
plt.savefig("map.pdf")
