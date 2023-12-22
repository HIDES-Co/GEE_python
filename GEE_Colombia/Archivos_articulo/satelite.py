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

class Satellite(ABC):
    """
    Abstract base class for a satellite.
    """
    
    def __init__(self, sat_name, layer):
        """
        Initialize the Satellite object.
        
        Args:
            sat_name (str): The name of the satellite.
            layer (str): The layer of the satellite.
        """
        self.satellite = sat_name
        self.layer = layer
        self.regions = []
    
    def get_image_collection(self, init_date, final_date):
        """
        Get the image collection for the given date range.
        
        Args:
            init_date (str): The initial date.
            final_date (str): The final date.
            
        Returns:
            ee.ImageCollection: The image collection for the given date range.
        """
        image_collection = ee.ImageCollection(self.satellite).select(self.layer)
        image_collection = image_collection.filterDate(init_date, final_date)
        return image_collection
    
    @abstractmethod
    def get_available_regions(self):
        """
        Abstract method to get available regions.
        """
        pass
    
    @abstractmethod
    def get_roi(self, identifier):
        """
        Abstract method to get region of interest.
        
        Args:
            identifier (str): The identifier of the region.
        """
        pass
    
    def add_region(self, title, identifier, region_object):
        """
        Add a region to the regions list.
        
        Args:
            title (str): The title of the region.
            identifier (str): The identifier of the region.
            region_object (ee.Geometry): The region object.
        """
        region = {
            'title': title,
            'id': identifier,
            'type': ee.Algorithms.ObjectType(region_object),
            'object': region_object
        }
        self.regions.append(region)

    def clip_image(self, image_collection, shape):
        """
        Clip the image collection with the given shape.
        
        Args:
            image_collection (ee.ImageCollection): The image collection.
            shape (ee.Geometry): The shape to clip the image collection.
            
        Returns:
            ee.Image: The clipped image.
        """
        mean_image = image_collection.mean()
        mean_image_clip = mean_image.clip(shape.geometry())
        return mean_image_clip
    
    
class ColSatellite(Satellite):
    """
    Class for a Colombian satellite that inherits from the Satellite class.
    """
    
    fronteras_maritimas_col = ee.FeatureCollection('projects/ee-jolejua/assets/EEZ_land_union_v3_202003')
    fronteras_maritimas_col = fronteras_maritimas_col.filter(ee.Filter.eq('UNION', 'Colombia'))
    fronteras_maritimas_col = {
        'title': 'Colombia con sus fronteras maritimas',
        'id': 'fronterasMaritimasCol',
        'type': type(fronteras_maritimas_col),
        'object': fronteras_maritimas_col
    }
    local_regions = [fronteras_maritimas_col]
    
    def get_available_regions(self):
        """
        Get available regions and print them.
        """
        for region in self.local_regions:
            self.regions.append(region)
        print('Plotting available regions... \n')
        for i, region in enumerate(self.regions, 1):
            print(f'---------- Region {i} -----------')
            print('title: ' + region['title'])
            print('id: ' + region['id'])
            print('type: ', region['type'])
            print('--------------------------------')
        print('Done...')  
    
    def get_roi(self, identifier):
        """
        Get region of interest (ROI) by identifier.
        
        Args:
            identifier (str): The identifier of the region.
            
        Returns:
            ee.Geometry: The region of interest.
        """
        
        roi = None
        for region in self.regions:
            if region['id'] == identifier:
                print('Selecting ' + identifier + ' region')
                roi = region['object'] 
        if roi:
            return roi
        else: 
            print('No funciona')
            
    def get_clusters_data(self):
        pass
            
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
        plt.show()
        
        return model_k