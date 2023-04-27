import os
os.environ['USE_PYGEOS'] = '0'
import ee
import geemap
import geopandas as gdp
from ipyleaflet import GeoJSON
from zipfile import ZipFile
from geeS2downloader import GEES2Downloader

# Initializate GEE API
ee.Initialize()

def getImageCollection(satellite, layer, initDate, finalDate):
    
    imageCollection = ee.ImageCollection(satellite).select(layer).filterDate(initDate, finalDate)
    return imageCollection



satellite = 'COPERNICUS/S5P/OFFL/L3_CH4'
layer = 'CH4_column_volume_mixing_ratio_dry_air_bias_corrected'
initDate = '2018-05-31'
finalDate = '2022-12-31'



fronterasMaritimasCol = ee.FeatureCollection('projects/ee-jolejua/assets/EEZ_land_union_v3_202003')
fronterasMaritimasCol = fronterasMaritimasCol.filter(ee.Filter.eq('UNION', 'Colombia'))


methaneImageCollection = getImageCollection(satellite, layer, initDate, finalDate)

methaneImageClip = methaneImageCollection.mean()
methaneImageClip = methaneImageClip.clip(fronterasMaritimasCol)


methaneImageCollectionClip = ee.ImageCollection.fromImages([methaneImageClip]) # from Image to ImageCollection

pixelsSample = methaneImageClip.sample(region = fronterasMaritimasCol.geometry(), scale=5000, factor = 1, geometries=True) # generate a sample of the pixels in the images

n_clusters = 3
clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(pixelsSample)
result = methaneImageClip.cluster(clusterer)

Map = geemap.Map(center=(4.6,-74),zoom=9)
Map.addLayer(result.randomVisualizer(), {}, 'clusters')
Map