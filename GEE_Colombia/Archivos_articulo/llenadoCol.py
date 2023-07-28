#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 19:29:43 2023

@author: bojack
"""
from satelite import colSatellite
from satelite import csv_from_sat
from satelite import statAnalisisData

while True:
    try:
        import sys
        import subprocess
        from google.colab import drive
        import io
        from contextlib import redirect_stdout
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
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
        break
    
    except ModuleNotFoundError as e:
        try: 
            subprocess.run([sys.executable, "-m", "pip", "install", e.name])
        except:
            raise Exception(f"Sorry, the library {e.name} could not be installed")


#ee.Authenticate()
ee.Initialize()




satelliteName = 'COPERNICUS/S5P/OFFL/L3_CH4'
layer = 'CH4_column_volume_mixing_ratio_dry_air_bias_corrected'
initDate = '2018-05-31'
finalDate = '2022-12-31'





methaneSat = colSatellite(satelliteName, layer) # crea el objeto
methaneImageCollection = methaneSat.getImageCollection(initDate, finalDate) # obtiene datos para las fechas especificadas
methaneSat.getAviableRegions() # Muestra las regiones disponibles
fronterasMaritimasCol = methaneSat.getROI('fronterasMaritimasCol') # selecciona la region que ya trae el objeto
methaneImageClip = methaneSat.clipImage(methaneImageCollection, fronterasMaritimasCol) # Recorta la imagen a una región
methaneImageCollectionClip = ee.ImageCollection.fromImages([methaneImageClip]) # from Image to ImageCollection (just contains 1 image in this case)


# Obtener las regiones de colombia para hacer el llenado


RegionesCol = ee.FeatureCollection('projects/ee-hides/assets/zonas_colombia')
methaneSat.addRegion('Zonas de similar comportamiento Colombia', 'RegionesCol', RegionesCol) # Se agrega la región al objeto
RegionesCol = RegionesCol.geometry() # de featureCollection a geometry
coordinates = RegionesCol.coordinates() # Obtiene las coordenadas del geometry
coordinates = coordinates.getInfo() 

"""
El objeto geometries a continuación tendrá las zonas de colombia individualizadas ya que el featureCollection 
que se importa viene con tollas ellas en conjunto.
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

geometries = [] # Lista con las zonas de colombia como poligonos
for i in coordinates:
    for j in i:
      geometries.append(ee.Geometry.Polygon(j)) # individualizo cada una de las regiones 
      
      
"""
A continuación se leen los datos obtenidos del satelite en las fechas especificadas los cuales fueron pormediados
para los años desde el 2019 hasta el 2023.
"""     
r = np.loadtxt('2019-02-09_2023-05-05_prom.csv',skiprows=1, usecols=(1,2,3),delimiter=',') # se carga el archivo .csv

g=9 # Valor dependiendo a la region especifica de colombia
speciicRegionPoly = geometries[g]
satData = csv_from_sat(r)      
xr, yr, fieldr = satData.getVal_in_sahpe(geometries[g])


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


##### ¡¡¡¡REVISADO!!!! ###########
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

angle = np.pi/8
bins = np.arange(0,200,4)/100
bin_center, dir_vario, counts = gs.vario_estimate(
    *((x, y), field, bins),
    direction=gs.rotated_main_axes(dim=2, angles=angle),
    angles_tol=np.pi / 16,
    bandwidth=8,
    return_counts=True,
)

#---
model = gs.Integral(dim=2, angles=angle)
main_axes=model.main_axes()
print("Original:")
print(model)
model.fit_variogram(bin_center, dir_vario)
print("Fitted:")
print(model)

#--

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 5])

ax1.scatter(bin_center, dir_vario[0], label="emp. vario: pi/8")
ax1.scatter(bin_center, dir_vario[1], label="emp. vario: pi*5/8")

ax1.legend(loc="lower right")

#model.plot("vario_axis", axis=0, ax=ax1, label="fit on axis 0")
#model.plot("vario_axis", axis=1, ax=ax1, label="fit on axis 1")
ax1.set_title("Modelo anisotrópico Mar adentro")

#srf.plot(ax=ax2)
plt.show()

input('done')
#---

fc=geometries[g]

#---

cor=fc.getInfo()['coordinates'][0]
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

#print(cor)
# xmax = np.amax(cor,axis=0)
# ymax = np.amax(cor,axis=1)
# xmin = np.amin(cor,axis=0)
# ymin = np.amin(cor,axis=1)

print(xmax,ymax,xmin,ymin)
gridx = np.arange(xmin,xmax, 0.04)
gridy = np.arange(ymin, ymax, 0.04)

# model = gs.Integral(
#     dim=2, len_scale=1, anis=0.2, angles=-0.5, var=0.5, nugget=0.1
# )
#model_k=gs.Integral(latlon=True, var=3.19e+02, len_scale=6.33e+02, nugget=0.0, rescale=6.37e+03, nu=0.229)
#model=fit_model
#Si hay suficientes datos, se puede dividir los datos con el fin de tener un
#grupo de validación de la estimación
#ind=np.random.choice(len(xr),int(len(xr)*1))
cond_x=xr#[ind]#r[:,1]
cond_y=yr#[ind]#r[:,0]
cond_val=fieldr#[ind]#r[:,2]
OK2 = gs.krige.Ordinary(model_k, [cond_x, cond_y], cond_val, exact=True)
OK2.structured([gridx, gridy])
#ax = OK2.plot()

input('llego al krige-------------------------------------')

"""
#------
xx, yy = np.meshgrid(gridy, gridx)
z=OK2.field.copy()
w=OK2.krige_var.copy()
#z=z.reshape(len(gridy),len(gridx))
data = np.array([xx, yy, z, w]).reshape(4, -1).T
np.shape(data)


plt.figure(1)
fig, ax1 = plt.subplots(layout='constrained')

#myPoly.boundary.plot(edgecolor='red')
CS1=plt.contourf(xx, yy, w,100)
fig.colorbar(CS1)
ax1.set_title('Varianza en Guaviare')
ax1.set_xlabel('Longitud')
ax1.set_ylabel('Latitud')
plt.show()
plt.figure(2)

#myPoly.boundary.plot(edgecolor='red')
fig1, ax2 = plt.subplots(layout='constrained')
CS=ax2.contourf(xx, yy, z,100)
ax2.set_title('Valor en Guaviare')
ax2.set_xlabel('Longitud')
ax2.set_ylabel('Latitud')
fig1.colorbar(CS)
plt.show()

#---

cor_col=fronterasMaritimasCol.getInfo()['features'][0]['geometry']['coordinates'][0]
#--
# Recorte del cuadrado en la zona correspondiente

poly1 = geometries[g] # verificar y si no sirve borrar

poly1 = Polygon( cor )
myPoly = gpd.GeoSeries([poly1])
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
ind=[]
for i in range(np.shape(xx)[0]):
  for j in range(np.shape(xx)[1]):
    point = Point(xx[i,j],yy[i,j])
    if not poly1.contains(point):
      z[i,j]=np.nan
      w[i,j]=np.nan
    #else:

poly1 = Polygon( cor_col )
myPoly = gpd.GeoSeries([poly1])
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
ind=[]
for i in range(np.shape(xx)[0]):
  for j in range(np.shape(xx)[1]):
    point = Point(xx[i,j],yy[i,j])
    if not poly1.contains(point):
      z[i,j]=np.nan
      w[i,j]=np.nan
    #else:

df=pd.DataFrame(data,columns=['lon','lat','value','var'])

#---

plt.figure(1)
fig, ax1 = plt.subplots(layout='constrained')

#myPoly.boundary.plot(edgecolor='red')
CS1=plt.contourf(xx, yy, w,100)
fig.colorbar(CS1)
ax1.set_title('Varianza en zona antioqueña')
ax1.set_xlabel('Longitud')
ax1.set_ylabel('Latitud')
plt.show()
plt.figure(2)

#myPoly.boundary.plot(edgecolor='red')
fig1, ax2 = plt.subplots(layout='constrained')
CS=ax2.contourf(xx, yy, z,100)
ax2.set_title('Valor en zona antioqueña')
ax2.set_xlabel('Longitud')
ax2.set_ylabel('Latitud')
fig1.colorbar(CS)
plt.show()

#---
input()
"""