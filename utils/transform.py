'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import os
import numpy as np
import json
import glob
import pandas as pd
from pathlib import Path

from shapely import Polygon, MultiPolygon
from shapely.affinity import scale

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType as AV2ObjectType
from av2.map.lane_segment import LaneMarkType as AV2LaneMarkType
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario
from av2.map.map_api import ArgoverseStaticMap


from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track, ObjectState, RequiredPrediction
from waymo_open_dataset.protos.map_pb2 import Map, MapFeature, LaneCenter, Crosswalk, RoadEdge, RoadLine, MapPoint 

SCENE_LEN = 110
HIST_LEN = 50

TRACK_CATEGORIES =['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']

OBJECT_TYPES = ['vehicle', #0
                'pedestrian', #1
                'motorcyclist', #2
                'cyclist', #3
                'bus', #4
                'static', #5
                'background', #6
                'construction', #7
                'riderless_bicycle', #8
                'unknown' #9
               ] 


def av2_object_type_converter(av2_obj_type):
    '''
        convert agent type from argo2 to waymo, return default agent size in [m]
    '''
    if av2_obj_type == 0: # vehicle
        return Track.ObjectType.TYPE_VEHICLE, 4.8, 2.0, 1.
    elif av2_obj_type == 4: # bus
        return Track.ObjectType.TYPE_VEHICLE, 12, 2.5, 2.5  # 
    elif av2_obj_type == 1: # pedestrian
        return Track.ObjectType.TYPE_PEDESTRIAN, 0.6, 0.6, 1.7 # pedestrian
    elif av2_obj_type == 2 or av2_obj_type == 3: # cyclist
        return Track.ObjectType.TYPE_CYCLIST, 2.0, 0.6, 1.7 # cyclist
    elif av2_obj_type == 9: # unknown
        return Track.ObjectType.TYPE_UNSET, 0.5, 0.5, 1.0 # unknown
    else: # others
        return Track.ObjectType.TYPE_OTHER, 0.5, 0.5, 1.0 # other

def av2_lane_type_converter(av2_lane_type):
    '''
        convert lane type from argo2 to waymo
    '''
    if av2_lane_type == 0: # TYPE_UNDEFINED 
        return 0 # 'VEHICLE'
    elif av2_lane_type == 1: # TYPE_FREEWAY  
        return 0 # 'VEHICLE'
    elif av2_lane_type == 2: # TYPE_SURFACE_STREET   
        return 0 # 'VEHICLE'
    elif av2_lane_type == 3: # TYPE_BIKE_LANE    
        return 1 # 'BIKE'
    elif av2_lane_type == 4: # TYPE_CROSSWALK    
        return 3 # 'PEDESTRIAN'


def expand_polygon(polygon, distance):    
    # Calculate the polygon centroid
    centroid = polygon.centroid

    # Calculate scale factors; distance is applied relative to the polygon's bounding box size
    # You can calculate the factor based on the current polygon dimensions
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    current_width = bounds[2] - bounds[0]
    current_height = bounds[3] - bounds[1]
    
    # Scaling factors based on the desired expansion distance
    scale_x = (current_width + 2 * distance) / current_width
    scale_y = (current_height + 2 * distance) / current_height

    # Scale the polygon around its centroid
    expanded_polygon = scale(polygon, xfact=scale_x, yfact=scale_y, origin=centroid)
    
    # Return the expanded polygon's coordinates as a list of tuples
    return expanded_polygon


def union_polygon(polygon_1, polygon_2, expand_distance = 1.):
    return expand_polygon(expand_polygon(polygon_1,expand_distance).union(expand_polygon(polygon_2,expand_distance)),-expand_distance)
    
    
def parquet_to_protobuf(src_file: str):
    scenario_file = glob.glob(os.path.join(src_file, '*.parquet*'))[0]
    map_file = glob.glob(os.path.join(src_file, '*.json*'))[0]

    a2_scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_file)
    avm = ArgoverseStaticMap.from_json(Path(map_file))
    
    a2_df = pd.read_parquet(scenario_file)
    historical_df = a2_df[a2_df['timestep'] < HIST_LEN]
    
    actor_ids = list(historical_df['track_id'].unique())
    actor_ids = list(filter(lambda actor_id: np.sum(historical_df[historical_df['track_id'] == actor_id]['observed'])>=1, actor_ids))
    a2_df = a2_df[a2_df['track_id'].isin(actor_ids)]
    
    
    wo_scenario = process_track(a2_scenario, a2_df)
    wo_scenario = process_map(avm, wo_scenario)
    
    return wo_scenario
    
    
def process_track(a2_scenario: ArgoverseScenario, a2_df):
    actor_ids = list(a2_df['track_id'].unique())
    timesteps = list(np.sort(a2_df['timestep'].unique()))
    wo_scenario = Scenario()
    
    wo_scenario.scenario_id  = a2_scenario.scenario_id 
    wo_scenario.current_time_index = 49
    timestamps = a2_scenario.timestamps_ns/ 1e9
    for t in (timestamps - timestamps[0]):
        wo_scenario.timestamps_seconds.append(t) # shape (110)
    tracks_to_predict = []
    
    wo_tracks=[]
    
    for (actor_id, actor_df) in a2_df.groupby('track_id'):
    #for i_track, a2_track in enumerate(a2_scenario.tracks):
        i_track = actor_ids.index(actor_id)

        actor_steps = [timesteps.index(timestep) for timestep in a2_df[a2_df['track_id'] == actor_id]['timestep']]
        wo_track = Track()
        
        if actor_id == 'AV':
            wo_track.id = 0
            wo_scenario.sdc_track_index = i_track
        else:
            wo_track.id = int(actor_id)
            
        wo_track.object_type, l, w, h = av2_object_type_converter(OBJECT_TYPES.index(actor_df['object_type'].unique()[0]))
        
        if actor_df['object_category'].unique()[0] >= 2.: 
            to_pred = RequiredPrediction()
            to_pred.track_index = i_track 
            wo_scenario.tracks_to_predict.append(to_pred)
        
        wo_states = []
        
        x = np.zeros((SCENE_LEN), dtype=np.float64) 
        y = np.zeros((SCENE_LEN), dtype=np.float64) 
        z = np.zeros((SCENE_LEN), dtype=np.float64) 
        heading = np.zeros((SCENE_LEN), dtype=float)
        vx = np.zeros((SCENE_LEN), dtype=float)
        vy = np.zeros((SCENE_LEN), dtype=float)
        valid = np.zeros((SCENE_LEN), dtype=bool)
        
        
        x[actor_steps] = np.array(actor_df[:]['position_x'].values)
        y[actor_steps] = np.array(actor_df[:]['position_y'].values)
        heading[actor_steps] = np.array(actor_df[:]['heading'].values)
        vx[actor_steps] = np.array(actor_df['velocity_x'].values)
        vy[actor_steps] = np.array(actor_df['velocity_x'].values)
        valid[actor_steps] = True         
        
        for a2_state in zip(x,y,heading,vx,vy,valid):
            wo_state = ObjectState()
            wo_state.center_x = a2_state[0]
            wo_state.center_y = a2_state[1]
            wo_state.center_z = 0.
        
            wo_state.heading = a2_state[2]

            wo_state.velocity_x = a2_state[3]
            wo_state.velocity_y = a2_state[4]

            wo_state.valid = a2_state[5]
            wo_state.length, wo_state.width, wo_state.height = l, w, h
            wo_track.states.append(wo_state)
            
        wo_scenario.tracks.append(wo_track)
            
    return wo_scenario
    

def process_map(avm: ArgoverseStaticMap,
                wo_scenario: Scenario):    
    ### Crosswalk ###
    cross_walk_ids = list(avm.vector_pedestrian_crossings.keys())
    for crosswalk in avm.get_scenario_ped_crossings():
        wo_cw = Crosswalk()
        edge1 = crosswalk.edge1.xyz[:, :2]
        edge2 = crosswalk.edge2.xyz[:, :2]
        
        for cw_pt in [edge1[0], edge1[1], edge2[1], edge2[0], edge1[0]]:
            wo_cw_pt = MapPoint()
            wo_cw_pt.x = cw_pt[0]
            wo_cw_pt.y = cw_pt[1]
            wo_cw_pt.z = 0.
            wo_cw.polygon.append(wo_cw_pt)
        
        wo_map_feature = MapFeature(id=crosswalk.id, crosswalk=wo_cw)
        wo_scenario.map_features.append(wo_map_feature)
    
    lane_segment_ids = avm.get_scenario_lane_segment_ids()
    
    ### Lane Segment ###
    for lane_segment in avm.get_scenario_lane_segments():
        wo_lane_center = LaneCenter()
   
        
        lane_segment_idx = lane_segment_ids.index(lane_segment.id)
        left_bound = avm.vector_lane_segments[lane_segment.id].left_lane_boundary
        right_bound = avm.vector_lane_segments[lane_segment.id].right_lane_boundary
        
        for i_b, bound_line in enumerate([left_bound, right_bound]):
            road_line = RoadLine()

            for b_pt in bound_line.xyz:
                wo_b_pt = MapPoint()
                wo_b_pt.x = b_pt[0]
                wo_b_pt.y = b_pt[1]
                wo_b_pt.z = 0.
                road_line.polyline.append(wo_b_pt)

                
            wo_map_feature_road_line = MapFeature(id= lane_segment.id, road_line= road_line)
            
            wo_scenario.map_features.append(wo_map_feature_road_line)
    

    da_polygons = []
    for da_b in avm.get_scenario_vector_drivable_areas():
        coords = da_b.xyz[:, :2]
        da_polygons.append(Polygon(coords))
    
    da_polygon = da_polygons[0]
    if len(da_polygons) > 1:
        for other_polygon in da_polygons[1:]:
            da_polygon = union_polygon(da_polygon, other_polygon)
    
 
    road_edge = RoadEdge()
    x_list, y_list = [], []
    if not isinstance(da_polygon, Polygon):
        areas = np.zeros(len(da_polygon.geoms))
        for i, poly in enumerate(da_polygon.geoms):
            areas[i] = poly.area
        
        da_polygon=da_polygon.geoms[np.argmax(areas)]
        
    x_list, y_list = da_polygon.exterior.xy
        
    for x, y in zip(x_list[::-1], y_list[::-1]):

        wo_b_pt = MapPoint()
        wo_b_pt.x = x
        wo_b_pt.y = y
        wo_b_pt.z = 0.
        road_edge.polyline.append(wo_b_pt)

    wo_map_feature_road_edge = MapFeature(id= da_b.id, road_edge= road_edge)
    wo_scenario.map_features.append(wo_map_feature_road_edge)
            
    return wo_scenario