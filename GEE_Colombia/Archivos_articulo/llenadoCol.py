#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 19:29:43 2023

@author: bojack
"""
from satelite import ColSatellite
from satelite import csv_from_sat
from satelite import statAnalisisData


import sys
import subprocess
import io
from contextlib import redirect_stdout
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
import ee
import requests
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import shape
from numpy.ma.core import sqrt
from math import ceil

#ee.Authenticate()
#ee.Initialize()




satelliteName = 'COPERNICUS/S5P/OFFL/L3_CH4'
layer = 'CH4_column_volume_mixing_ratio_dry_air_bias_corrected'
initDate = '2018-05-31'
finalDate = '2022-12-31'





methaneSat = ColSatellite(satelliteName, layer) # crea el objeto
methaneImageCollection = methaneSat.get_image_collection(initDate, finalDate) # obtiene datos para las fechas especificadas
methaneSat.get_available_regions() # Muestra las regiones disponibles
fronterasMaritimasCol = methaneSat.get_roi('fronterasMaritimasCol') # selecciona la region que ya trae el objeto, la entrada es el ID
methaneImageClip = methaneSat.clip_image(methaneImageCollection, fronterasMaritimasCol) # Recorta la imagen a una región
methaneImageCollectionClip = ee.ImageCollection.fromImages([methaneImageClip]) # from Image to ImageCollection (just contains 1 image in this case)

# Obtener las regiones de colombia para hacer el llenado


RegionesCol = ee.FeatureCollection('projects/ee-hides/assets/zonas_colombia')
methaneSat.add_region('Zonas de similar comportamiento Colombia', 'RegionesCol', RegionesCol) # Se agrega la región al objeto
RegionesCol = RegionesCol.geometry() # de featureCollection a geometry
coordinates = RegionesCol.coordinates() # Obtiene las coordenadas del geometry
coordinates = coordinates.getInfo() 

"""
El objeto geometries a continuación tendrá las zonas de colombia individualizadas ya que el featureCollection 
que se importa viene con todas ellas en conjunto.
geometries = [
0: N/A
1: medellin
2: cuenca cauca
3: Guajira
4: Cuenca magdalena 
5: Amazonia
6: llanos orinetales
7: costa y mar adentro occidental
8: región caribe 
9: mar adentro zona norte
]
"""
geometries_dict = {
0: 'N/A',
1: 'medellin',
2: 'cuenca cauca',
3: 'Guajira',
4: 'Cuenca magdalena ',
5: 'Amazonia',
6: 'llanos orinetales',
7: 'costa y mar adentro occidental',
8: 'región caribe', 
9: 'mar adentro zona norte'
}
g=4 # Valor dependiendo a la region especifica de colombia seleccionada del diccionario de arriba.

geometries = [] # Lista con las zonas de colombia como poligonos
for i in coordinates:
    for j in i:
      geometries.append(ee.Geometry.Polygon(j)) # individualizo cada una de las regiones 
      
      
"""
A continuación se leen los datos obtenidos del satelite en las fechas especificadas los cuales fueron pormediados
para los años desde el 2019 hasta el 2023.
"""     
r = np.loadtxt('2019-02-09_2023-05-05_prom.csv',skiprows=1, usecols=(1,2,3),delimiter=',') # se carga el archivo .csv


speciicRegionPoly = geometries[g]
satData = csv_from_sat(r)      
xr, yr, fieldr = satData.getVal_in_sahpe(geometries[g]) # devuelve las coordenadas y el valor en dicho punto


"""
A continuación se seleccionan el 10% de los datos de forma aleatoria 
y se almacenan en las variables x, y, field como np.arrays

"""
ind = np.random.choice(len(xr),int(len(xr)*0.1)) # Seecciona aleatoriamente el indice del 10% de los datos
xr = np.array(xr)
yr = np.array(yr)
fieldr = np.array(fieldr)
x = xr[ind]
y = yr[ind]
field = fieldr[ind]


"""
A continación se prueban diferentes modelos para los semivariogramas
"""

statAnalisis = statAnalisisData(r)
scores = statAnalisis.getVariogramScores(x, y, field)
#----

#--------


model_k = statAnalisis.getBestVariogram()
#----

# Parecen haber dos tipos de modelos en el semivariograma por lo que se evalúa la suposición de isotropía
#### ¿ porque se puede hacer la suposición de isotropia ?

"""
Nota: Isotropy refers to the property of a spatial process where the spatial dependence or correlation 
between two locations depends only on the distance between them, and not on their direction. 
If the assumption of isotropy is not met, then anisotropic models may be more appropriate to use. 
"""

angle = np.pi/8 # Sets the angle for the main axes of the variogram 

bins = np.arange(0,200,4)/100 # Creates an array of bins for the variogram.

bin_center, dir_vario, counts = gs.vario_estimate(
    *((x, y), field, bins),
    direction=gs.rotated_main_axes(dim=2, angles=angle),
    angles_tol=np.pi / 16,
    bandwidth=8,
    return_counts=True,
)

#---

model = gs.SuperSpherical(dim=2, angles=angle)
main_axes=model.main_axes()
print("Original:")
print(model)
model.fit_variogram(bin_center, dir_vario)
print("Fitted:")
print(model)

"""
RESULT:
The fitting process has adjusted the parameters of the model to better match 
the variogram of the data. The fitted model has a higher variance, a shorter 
range, a different shape, and is anisotropic compared to the original model. 
"""
#--

fig, ax1 = plt.subplots(figsize=[10, 5])

ax1.scatter(bin_center, dir_vario[0], label="emp. vario: pi/8")
ax1.scatter(bin_center, dir_vario[1], label="emp. vario: pi*5/8")

# Create a legend on the subplot in the lower right corner
ax1.legend(loc="lower right")

# Set the title of the subplot
ax1.set_title(f"Modelo anisotrópico {geometries_dict[g]}")

# Display the figure
plt.show()

#----

#---

cor=speciicRegionPoly.getInfo()['coordinates'][0]
xmax=cor[0][1]
ymax=cor[0][0]
xmin=cor[0][1]
ymin=cor[0][0]
for i in range(len(cor)):
  xc=cor[i][1]
  yc=cor[i][0]
  #print(xc)
  #print(x>xmax)
  if xc>xmax:
    xmax=xc
  if yc>ymax:
    ymax=yc
  if xc<xmin:
    xmin=xc
  if yc<ymin:
    ymin=yc


print(xmax,ymax,xmin,ymin)

#------------------------------------------------------- codigo primo juli -----------

cond_xs = []
cond_ys = []
cond_vals = []
gridxs = []
gridys = []
divs = ceil(sqrt(len(xr)/200))
x_len = (xmax - xmin) / divs
y_len = (ymax - ymin) / divs
for i in range(divs):
  indx = np.argwhere((xmin+x_len*i)<=xr)
  indx2 = np.argwhere(xr<(xmin+(x_len*(i+1))))
  for j in range(divs):
    indy = np.argwhere((ymin+y_len*j)<=yr)
    indy2 = np.argwhere(yr<ymin+(y_len*(j+1)))
    ind = indx[np.in1d(indx, indx2)]
    ind = ind[np.in1d(ind, indy)]
    ind = ind[np.in1d(ind, indy2)]
    cond_xs.append(xr[ind])
    cond_ys.append(yr[ind])
    cond_vals.append(fieldr[ind])
    gridxs.append(np.arange(xmin+x_len*i, xmin+(x_len*(i+1)), 0.005))
    gridys.append(np.arange(ymin+y_len*j, ymin+(y_len*(j+1)), 0.005))


# data = np.array([[0, 0, 0, 0]])

# for i in range(len(cond_xs)):
#   OK2 = gs.krige.Ordinary(model_k, [cond_xs[i], cond_ys[i]], cond_vals[i], exact=True)
#   OK2.structured([gridxs[i], gridys[i]])
#   #ax = OK2.plot()
  
#   xx, yy = np.meshgrid(gridys[i], gridxs[i])
#   z = OK2.field.copy()
#   w=OK2.krige_var.copy()
#   #z=z.reshape(len(gridy),len(gridx))
#   data_i = np.array([xx, yy, z, w]).reshape(4, -1).T
#   data = np.concatenate((data, data_i), axis=0)
  
# data = np.delete(data, 0, 0)

field_data = []
variance_field_data = []
x_data = [] 
y_data = []
for i in range(len(cond_xs)):
 
  OK2 = gs.krige.Ordinary(model_k, [cond_xs[i], cond_ys[i]], cond_vals[i], exact=True)
  OK2.structured([gridxs[i], gridys[i]])
  
  xx, yy = np.meshgrid(gridys[i], gridxs[i])
  x_data.append(xx)
  y_data.append(yy)
  
  z = OK2.field.copy()
  w=OK2.krige_var.copy()
  field_data.append(z)
  variance_field_data.append(w)
  


#print(data)
# -------------------------------------------------------------------------------------
#------
#xx, yy = np.meshgrid(gridy, gridx)
#z = OK2.field.copy()
#w=OK2.krige_var.copy()
#z=z.reshape(len(gridy),len(gridx))
#data = np.array([xx, yy, z, w]).reshape(4, -1).T




# plt.figure(1)
# fig, ax1 = plt.subplots(layout='constrained')

# #myPoly.boundary.plot(edgecolor='red')
# CS1=plt.contourf(xx, yy, w,100)
# fig.colorbar(CS1)
# ax1.set_title('Varianza en ' + geometries_dict[g])
# ax1.set_xlabel('Longitud')
# ax1.set_ylabel('Latitud')
# plt.show()
# plt.figure(2)

# #myPoly.boundary.plot(edgecolor='red')
# fig1, ax2 = plt.subplots(layout='constrained')
# CS=ax2.contourf(xx, yy, z,100)
# ax2.set_title('Valor en ' + geometries_dict[g])
# ax2.set_xlabel('Longitud')
# ax2.set_ylabel('Latitud')
# fig1.colorbar(CS)
# plt.show()

#---

cor_col=fronterasMaritimasCol.getInfo()['features'][0]['geometry']['coordinates'][0]
#--
# Recorte del cuadrado en la zona correspondiente


speciicRegionPoly = Polygon( cor )
myPoly = gpd.GeoSeries([speciicRegionPoly])

poly_col = Polygon( cor_col )
myPoly_col = gpd.GeoSeries([poly_col])

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



# for i in range(np.shape(data)[0]):
  
#     point = Point(xx[i,j],yy[i,j])
    
#     if not speciicRegionPoly.contains(point) and  not myPoly_col.contains(point):
#         data[i, 2] = np.nan
#         data[i, 3] = np.nan
        

plt.figure(1)
fig, ax1 = plt.subplots(layout='constrained')

for z in range(len(field_data)):
    xx = x_data[z]
    yy = y_data[z]
    
    
    for i in range(np.shape(field_data[z])[0]):
        for j in range(np.shape(field_data[z])[1]):
            
            point = Point(xx[i,j],yy[i,j])
            
            if not speciicRegionPoly.contains(point):
                field_data[z][i, j] = np.nan
                variance_field_data[z][i, j] = np.nan
             
            if not myPoly_col.contains(point)[0]:
                field_data[z][i, j] = np.nan
                variance_field_data[z][i, j] = np.nan
            
    
    
    #myPoly.boundary.plot(edgecolor='red')
    CS1=plt.contourf(xx, yy, field_data[z], 100)

    print(z) 
    
    
fig.colorbar(CS1)
ax1.set_title('Valor en zona' + geometries_dict[g])
ax1.set_xlabel('Longitud')
ax1.set_ylabel('Latitud')        
plt.show()
input('doneeeeee')
#df = pd.DataFrame(data,columns=['lon','lat','value','var'])
#df = df[df['value'].notna()]
#df.to_csv('prueba_medallo.csv', sep=',', encoding='utf-8', index=False)

#---

plt.figure(1)
fig, ax1 = plt.subplots(layout='constrained')

#myPoly.boundary.plot(edgecolor='red')
CS1=plt.contourf(xx, yy, w,100)
fig.colorbar(CS1)
ax1.set_title('Varianza en zona' + geometries_dict[g])
ax1.set_xlabel('Longitud')
ax1.set_ylabel('Latitud')
plt.savefig('varianza'+ geometries_dict[g] + '.png', dpi=600)
plt.show()

plt.figure(2)

#myPoly.boundary.plot(edgecolor='red')
fig1, ax2 = plt.subplots(layout='constrained')
CS=ax2.contourf(xx, yy, z,100)
ax2.set_title('Valor en zona' + geometries_dict[g])
ax2.set_xlabel('Longitud')
ax2.set_ylabel('Latitud')
fig1.colorbar(CS)
plt.savefig('valor '+ geometries_dict[g] +'.png', dpi=600)
plt.show()

#---
