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

g = 3 # Valor dependiendo a la region especifica de colombia seleccionada del diccionario de arriba.

geometries = [] # Lista con las zonas de colombia como poligonos
for i in coordinates:
    for j in i:
      geometries.append(ee.Geometry.Polygon(j)) # individualizo cada una de las regiones 
   
specificRegionPoly = geometries[g]

"""
A continuación se leen los datos obtenidos del satelite en las fechas especificadas los cuales fueron pormediados
para los años desde el 2019 hasta el 2023.
"""     
r = np.loadtxt('2019-02-09_2023-05-05_prom.csv',skiprows=1, usecols=(1,2,3),delimiter=',') # se carga el archivo .csv

statAnalisis = statAnalisisData(r) 

xr, yr, fieldr = statAnalisis.getVal_in_sahpe(geometries[g]) # devuelve las coordenadas y el valor en dicho punto


"""
A continuación se seleccionan el 10% de los datos de forma aleatoria 
y se almacenan en las variables x, y, field como np.arrays

"""

x, y, field = statAnalisis.get_random_sample(xr, yr, fieldr, 0.1)




"""
A continación se prueban diferentes modelos para los semivariogramas
"""


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

model = gs.Linear(dim=2, angles=angle)
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

xmax,ymax,xmin,ymin = methaneSat.get_region_limits(specificRegionPoly)
limits = [xmax, ymax, xmin, ymin]

print('The limits of the región are: ', xmax,ymax,xmin,ymin)

# Paritición e interpolación

cond_xs, cond_ys, cond_vals, gridxs, gridys = statAnalisis.partition_geometry(limits, 500, 0.005)
    
x_data, y_data, field_data, variance_field_data = statAnalisis.get_interpolation(cond_xs, cond_ys, cond_vals, gridxs, gridys, model_k)





#--
# Recorte del cuadrado en la zona correspondiente


specific_region_poligon = Polygon(specificRegionPoly.getInfo()['coordinates'][0])
specific_region_poligon = gpd.GeoSeries([specific_region_poligon])

cor_col = fronterasMaritimasCol.getInfo()['features'][0]['geometry']['coordinates'][0]
poly_col = Polygon( cor_col )
col_polygon = gpd.GeoSeries([poly_col])

#from shapely.geometry import Point
#from shapely.geometry.polygon import Polygon



# plt.figure(1)
# fig, ax1 = plt.subplots(layout='constrained')

# for z in range(len(field_data)):
#     xx = x_data[z]
#     yy = y_data[z]
    
    
#     for i in range(np.shape(field_data[z])[0]):
#         for j in range(np.shape(field_data[z])[1]):
            
#             point = Point(xx[i,j],yy[i,j])
            
#             if not specific_region_poligon.contains(point)[0]:
#                 field_data[z][i, j] = np.nan
#                 variance_field_data[z][i, j] = np.nan
             
#             if not col_polygon.contains(point)[0]:
#                 field_data[z][i, j] = np.nan
#                 variance_field_data[z][i, j] = np.nan
            
    
    
#     #myPoly.boundary.plot(edgecolor='red')
#     CS1=plt.contourf(xx, yy, field_data[z], 100)

#     print(f"geometria: {geometries_dict[g]}, partición: {z}") 
    
    
# fig.colorbar(CS1)
# ax1.set_title('Valor en zona' + geometries_dict[g])
# ax1.set_xlabel('Longitud')
# ax1.set_ylabel('Latitud')        
# plt.show()

plt.figure(1)
fig, ax1 = plt.subplots(layout='constrained')
#---------------------------------
for z in range(len(field_data)):
    xx = x_data[z]
    yy = y_data[z]
    field_data_i = field_data[z]
    variance_field_data_i = variance_field_data[z]
    
    field_data_div, variance_field_data_div = statAnalisis.clip_subdivition(xx, yy, field_data_i, variance_field_data_i, specific_region_poligon, col_polygon)
    
    CS1=plt.contourf(xx, yy, field_data[z], 100)

    print(f"geometria: {geometries_dict[g]}, partición: {z}") 
    
    
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

# plt.figure(1)
# fig, ax1 = plt.subplots(layout='constrained')

# #myPoly.boundary.plot(edgecolor='red')
# CS1=plt.contourf(xx, yy, w,100)
# fig.colorbar(CS1)
# ax1.set_title('Varianza en zona' + geometries_dict[g])
# ax1.set_xlabel('Longitud')
# ax1.set_ylabel('Latitud')
# plt.savefig('varianza'+ geometries_dict[g] + '.png', dpi=600)
# plt.show()

# plt.figure(2)

# #myPoly.boundary.plot(edgecolor='red')
# fig1, ax2 = plt.subplots(layout='constrained')
# CS=ax2.contourf(xx, yy, z,100)
# ax2.set_title('Valor en zona' + geometries_dict[g])
# ax2.set_xlabel('Longitud')
# ax2.set_ylabel('Latitud')
# fig1.colorbar(CS)
# plt.savefig('valor '+ geometries_dict[g] +'.png', dpi=600)
# plt.show()

#---
