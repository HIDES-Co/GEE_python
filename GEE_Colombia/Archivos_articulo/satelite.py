#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:49:12 2023

@author: bojack
"""
import ee
from abc import ABC, abstractmethod
from shapely.geometry import Point
from shapely.geometry import shape
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
#------------

#ee.Authenticate()
ee.Initialize()
#------------

class satellite(ABC):
    
    
    def __init__(self, satName, layer):
        
        self.satellite = satName
        self.layer = layer
        self.regions = []
    
    def getImageCollection(self, initDate, finalDate):
        
        imageCollection = ee.ImageCollection(self.satellite).select(self.layer)
        imageCollection = imageCollection.filterDate(initDate, finalDate)
        return imageCollection
    
    @abstractmethod
    def getAviableRegions(self):
        
        pass
        
    def getROI(self, identifier):
        
        for i in range(len(self.regions)):
            if self.regions[i]['id'] == identifier:
                return self.regions[i]['object'] 
    
    def addRegion(self, tittle, identificator, regionObject):
        
        region = {'title': tittle,
                    'id': identificator,
                    'type': ee.Algorithms.ObjectType(regionObject),
                    'object': regionObject
        }
        self.regions.append(region)

    def clipImage(self, imageCollection, shape):
        
        meanImage = imageCollection.mean()
        meanImageClip = meanImage.clip(shape)
        return meanImageClip
    
    
class colSatellite(satellite):
    
    
    fronterasMaritimasCol = ee.FeatureCollection('projects/ee-jolejua/assets/EEZ_land_union_v3_202003')
    fronterasMaritimasCol = fronterasMaritimasCol.filter(ee.Filter.eq('UNION', 'Colombia'))
    fronterasMaritimasCol = {'title': 'Colombia con sus fronteras maritimas',
                'id': 'fronterasMaritimasCol',
                'type': type(fronterasMaritimasCol),
                'object': fronterasMaritimasCol
    }
    localRegions = [fronterasMaritimasCol]
    
    def getAviableRegions(self):
        
        for j in self.localRegions:
            self.regions.append(j)
        for i in range(len(self.regions)):
            print('---------- Region '+ str(i+1) + ' -----------')
            print('title: ' + self.regions[i]['title'])
            print('id: ' + self.regions[i]['id'])
            print('type: ', self.regions[i]['type'])
            print('--------------------------------')
            
            
            
class csv_from_sat(object):
    
    
    def __init__(self, satData):
        
        self.satData = satData
        
    def getVal_in_sahpe(self, shapefile):
        """
        Funcion que obtiene los valores que estan contenidos en 

        Parameters
        ----------
        shapefile : ee.Geometry.Polygon()
            DESCRIPTION.
            
        Returns
        -------
        xr : TYPE
            DESCRIPTION.
        yr : TYPE
            DESCRIPTION.
        fieldr : TYPE
            DESCRIPTION.

        """
        fieldr = [] #r[:,0] -----> valor de la medición satelital
        xr = [] #r[:,1] ---------> valor de coordenada longitud
        yr = [] #]r[:,2] --------> valor de coprdenada latitud
        shapely_polygon = shape(shapefile.getInfo()) # crea un poligono dadas unas coordenadas para el metodo shapely_polygon.contains
        for i in range(len(self.satData[:,1])):
            point = Point(self.satData[i,0], self.satData[i,1])
            if shapely_polygon.contains(point):
              fieldr.append(self.satData[i,2])
              xr.append(self.satData[i,1])
              yr.append(self.satData[i,0])
              
        return xr, yr, fieldr
    

class statAnalisisData(csv_from_sat):
    
    variogramModels = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Integral": gs.Integral,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Cubic" : gs.Cubic,
        "Linear" : gs.Linear,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
        "HyperSpherical": gs.HyperSpherical,
        "SuperSpherical": gs.SuperSpherical,
        "JBessel": gs.JBessel,
    }
    def __init__(self, satData):
        csv_from_sat.__init__(self, satData)
        self.scores = None
        self.x = None
        self.y = None
        self.field = None
        
    def getVariogramScores(self, x, y, field, graph=False):
        
        self.x = x
        self.y = y
        self.field = field
        bins = np.arange(0,150,3.5)/10000
        bin_center, gamma=gs.vario_estimate((x,y),field,bins,latlon=True)
        scores = {}
        
        # plot the estimated variogram
        if graph:
            plt.scatter(bin_center, gamma, color="k", label="data")
            ax = plt.gca()
        # fit all models to the estimated variogram
        for model in self.variogramModels:
            fit_model = self.variogramModels[model](dim=2,latlon=True, rescale=gs.EARTH_RADIUS)
            para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True,nugget=False)#sill=np.var(field))
            scores[model] = r2
            if graph:
                fit_model.plot(x_max=max(bin_center), ax=ax)
        
            if graph:
                plt.title('variograma para Mar adentro')
                plt.show()
        self.scores = scores                
        return scores
    
    def sortScores(self, scores):    
        
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print("RANKING by Pseudo-r2 score")
        for i, (model, score) in enumerate(ranking, 1):
            print(f"{i:>6}. {model:>15}: {score:.5}")

        return ranking
    
    def getBestVariogram(self):
        
        if self.scores == None:
            raise Exception('scores not found, please get variogram scores')
        ranking = self.sortScores(self.scores)
        model = ranking[0][0]
        fit_model = self.variogramModels[model](dim=2,latlon=True, rescale=gs.EARTH_RADIUS)
        bins = np.arange(0,150,3.5)/10000
        bin_center, gamma = gs.vario_estimate((self.x, self.y), self.field, bins, latlon=True)
        fit_model.fit_variogram(bin_center, gamma, nugget=False)
        ax = fit_model.plot(x_max=max(bin_center))
        ax.scatter(bin_center,gamma, color="k", label="data")
        model_k = fit_model
        
        return model_k