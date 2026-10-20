// Source reconstruction helper. Run in an authenticated Earth Engine Code Editor.
// No observed burned-area label is fabricated or inferred from the desired IoU.
// Static exports only: time-window interpretation and classifier need author confirmation.
var aoi=ee.Geometry.Rectangle([129.29,37.03,129.356,37.075],null,false);
var size=50;
var cells=ee.FeatureCollection(ee.List.sequence(0,size-1).map(function(y){
  return ee.List.sequence(0,size-1).map(function(x){
    x=ee.Number(x);y=ee.Number(y);
    var west=x.multiply(129.356-129.29).divide(size).add(129.29);
    var south=y.multiply(37.075-37.03).divide(size).add(37.03);
    var east=west.add((129.356-129.29)/size);
    var north=south.add((37.075-37.03)/size);
    return ee.Feature(ee.Geometry.Rectangle([west,south,east,north],null,false),{x:x,y:y});
  });
}).flatten());
var s2=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi).filterDate('2021-01-01','2022-01-01')
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',20));
// Compute each scene's raw NDVI before annual averaging. No 0..100 scaling.
var annualNdvi=s2.map(function(image){
  return image.normalizedDifference(['B8','B4']).rename('ndvi');
}).mean();
// Dataset version is a reconstruction choice, not proof of the original version.
var dsmTiles=ee.ImageCollection('JAXA/ALOS/AW3D30/V4_1').filterBounds(aoi).select('DSM');
var dsm=dsmTiles.mosaic().setDefaultProjection(ee.Image(dsmTiles.first()).projection());
var terrain=ee.Terrain.products(dsm);
var aspectRadians=terrain.select('aspect').multiply(Math.PI/180);
var inputs=annualNdvi.addBands(terrain.select('slope'))
  .addBands(aspectRadians.sin().rename('aspect_sin'))
  .addBands(aspectRadians.cos().rename('aspect_cos'));
// Circular aspect averaging is reconstructed after export using atan2(sin,cos).
var staticTable=inputs.reduceRegions({collection:cells,reducer:ee.Reducer.mean(),scale:10,tileScale:4});
Export.table.toDrive({collection:staticTable,description:'uljin_static_reconstruction',fileFormat:'CSV',selectors:['x','y','ndvi','slope','aspect_sin','aspect_cos']});
var sceneManifest=ee.FeatureCollection(s2.toList(s2.size()).map(function(item){
  var image=ee.Image(item);
  return ee.Feature(null,{scene_id:image.id(),time_start:image.get('system:time_start'),cloud_percentage:image.get('CLOUDY_PIXEL_PERCENTAGE')});
}));
Export.table.toDrive({collection:sceneManifest,description:'uljin_ndvi_scene_manifest',fileFormat:'CSV'});
// Export original hourly vectors and temperatures. Half-hour interpolation happens
// explicitly in prepare_dataset.py; it is not new half-hour observational detail.
var weather=ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterBounds(aoi)
  .filterDate('2022-03-04T02:00:00Z','2022-03-06T05:00:00Z')
  .select(['temperature_2m','dewpoint_temperature_2m','u_component_of_wind_10m','v_component_of_wind_10m']);
var weatherTable=ee.FeatureCollection(weather.toList(weather.size()).map(function(item){
  var image=ee.Image(item);
  return image.reduceRegions({collection:cells,reducer:ee.Reducer.mean(),scale:11132,tileScale:4})
    .map(function(feature){return feature.set('time_ms',image.get('system:time_start'));});
})).flatten();
Export.table.toDrive({collection:weatherTable,description:'uljin_hourly_weather_reconstruction',fileFormat:'CSV',selectors:['time_ms','x','y','temperature_2m','dewpoint_temperature_2m','u_component_of_wind_10m','v_component_of_wind_10m']});
Map.centerObject(aoi,12);Map.addLayer(annualNdvi.clip(aoi),{min:-1,max:1,palette:['blue','white','green']},'2021 annual NDVI');
print('Original ERA5 half-hour interpolation, burn-mask classifier and scene IDs still require confirmation.');
