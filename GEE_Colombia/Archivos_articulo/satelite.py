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
from math import ceil
from numpy.ma.core import sqrt

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
    
    def get_region_limits(self, specificRegionPoly):
        
        cor=specificRegionPoly.getInfo()['coordinates'][0]

        xmax=cor[0][1]
        ymax=cor[0][0]
        xmin=cor[0][1]
        ymin=cor[0][0]


        for i in range(len(cor)):
          xc=cor[i][1]
          yc=cor[i][0]
          if xc>xmax:
            xmax=xc
          if yc>ymax:
            ymax=yc
          if xc<xmin:
            xmin=xc
          if yc<ymin:
            ymin=yc
    
        return xmax, ymax, xmin, ymin
    
    
    
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
    
    def __init__(self, sat_name, layer):
        csv_from_sat.__init__(self, sat_name, layer)
    
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
        self.xr = None
        self.yr = None
        self.fieldr = None
        
    
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
        xr = np.array(xr)
        yr = np.array(yr)
        fieldr = np.array(fieldr)
        
        self.xr = xr
        self.yr = yr
        self.fieldr = fieldr
        
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
    
    
    
    def get_random_sample(self, xr, yr, fieldr, sample_size):
        
        ind = np.random.choice(len(xr),int(len(xr)*sample_size)) # Seecciona aleatoriamente el indice del 10% de los datos

        x_sample = xr[ind]
        y_sample = yr[ind]
        field_sample = fieldr[ind]
        
        return x_sample, y_sample, field_sample
    
    def partition_geometry(self, limits, div_factor, step):
        
        cond_xs = []
        cond_ys = []
        cond_vals = []
        gridxs = []
        gridys = []
        
        xmax = limits[0]
        ymax = limits[1]
        xmin = limits[2]
        ymin = limits[3]
        
        divs = ceil(sqrt(len(self.xr)/div_factor))
        
        print(f"partitioning geometry into {divs+divs} subdivitions...")
        
        x_len = (xmax - xmin) / divs
        y_len = (ymax - ymin) / divs


        for i in range(divs):
          indx = np.argwhere((xmin+x_len*i) <= self.xr)
          indx2 = np.argwhere(self.xr < (xmin+(x_len*(i+1))))
          for j in range(divs):
            indy = np.argwhere((ymin+y_len*j) <= self.yr)
            indy2 = np.argwhere(self.yr < ymin+(y_len*(j+1)))
            ind = indx[np.in1d(indx, indx2)]
            ind = ind[np.in1d(ind, indy)]
            ind = ind[np.in1d(ind, indy2)]
            cond_xs.append(self.xr[ind])
            cond_ys.append(self.yr[ind])
            cond_vals.append(self.fieldr[ind])
            gridxs.append(np.arange(xmin+x_len*i, xmin+(x_len*(i+1)), step))
            gridys.append(np.arange(ymin+y_len*j, ymin+(y_len*(j+1)), step))
            
            
        return cond_xs, cond_ys, cond_vals, gridxs, gridys
    
    def get_interpolation(self, cond_xs, cond_ys, cond_vals, gridxs, gridys, model_k):
        
        field_data = []
        variance_field_data = []
        x_data = [] 
        y_data = []
        print("Interpolating data...")
        for i in range(len(cond_xs)):
            print(f"interpolating partition {i+1}")
            OK2 = gs.krige.Ordinary(model_k, [cond_xs[i], cond_ys[i]], cond_vals[i], exact=True)
            OK2.structured([gridxs[i], gridys[i]])
            
            xx, yy = np.meshgrid(gridys[i], gridxs[i])
            x_data.append(xx)
            y_data.append(yy)
            
            z = OK2.field.copy()
            w=OK2.krige_var.copy()
            field_data.append(z)
            variance_field_data.append(w)
          
        return x_data, y_data, field_data, variance_field_data
          
    def clip_subdivition(self, xx, yy, field_data, variance_field_data, specific_region_poligon_1, specific_region_poligon_2):
        
        
        for i in range(np.shape(field_data)[0]):
            for j in range(np.shape(field_data)[1]):
                
                point = Point(xx[i,j],yy[i,j])
                
                if not specific_region_poligon_1.contains(point)[0]:
                    field_data[i, j] = np.nan
                    variance_field_data[i, j] = np.nan
                 
                if not specific_region_poligon_2.contains(point)[0]:
                    field_data[i, j] = np.nan
                    variance_field_data[i, j] = np.nan
                
        return field_data, variance_field_data
        
    
    
    
    