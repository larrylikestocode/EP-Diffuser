'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from shapely import LineString, Point, Polygon
import numpy as np
import warnings
import hashlib
import torch
from copy import deepcopy

from utils.spline import Spline
from utils.preprocess_utils.converter_utils import waymo_lane_type_converter, waymo_boundary_type_converter

class WOMapInterpreter():
    def __init__(self, 
                 scenario, 
                 mapel_deg):
        
        self.mapel_deg = mapel_deg
        self.map_infos = self.get_map_info(scenario)
        
        
    def get_map_info(self, scenario):
        map_infos = {
            'lane': {},
            'road_line': {},
            'crosswalk': {},
        }

        for cur_data in scenario.map_features:
            cur_info = {'id': cur_data.id}

            if cur_data.lane.ByteSize() > 0:
                cur_info['type'] = cur_data.lane.type  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane, 4: cross_walk

                cur_polyline = np.stack(
                    [np.array([point.x, point.y]) for point in cur_data.lane.polyline], axis=0)
                cur_info['points'] = cur_polyline

                cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
                cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)
                cur_info['left_neighbors'] = [left_neighbor.feature_id for left_neighbor in cur_data.lane.left_neighbors]
                cur_info['right_neighbors'] = [right_neighbor.feature_id for right_neighbor in cur_data.lane.right_neighbors]

                cur_info['left_boundary'] = [{
                        'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                        'feature_id': x.boundary_feature_id,
                        'boundary_type': x.boundary_type  # roadline type
                    } for x in cur_data.lane.left_boundaries
                ]
                cur_info['right_boundary'] = [{
                        'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                        'feature_id': x.boundary_feature_id,
                        'boundary_type': x.boundary_type  # roadline type
                    } for x in cur_data.lane.right_boundaries
                ]

                if cur_polyline.shape[0] > 1: # lane should have at least two points     
                    map_infos['lane'][cur_data.id] = cur_info

            elif cur_data.road_line.ByteSize() > 0:
                cur_info['type'] = cur_data.road_line.type

                cur_polyline = np.stack(
                    [np.array([point.x, point.y]) for point in cur_data.road_line.polyline], axis=0)
                cur_info['points']=cur_polyline

                map_infos['road_line'][cur_data.id] = cur_info

            elif cur_data.road_edge.ByteSize() > 0:
                cur_info['type'] = cur_data.road_edge.type

                cur_polyline = np.stack([np.array([point.x, point.y]) for point in cur_data.road_edge.polyline], axis=0)
                cur_info['points'] = cur_polyline

                map_infos['road_line'][cur_data.id] = cur_info

            elif cur_data.crosswalk.ByteSize() > 0:
                cur_info['type'] = 4 # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane, 4: cross_walk

                cur_polyline = np.stack([np.array([point.x, point.y]) for point in cur_data.crosswalk.polygon], axis=0)
                edge_1, edge_2 = find_edges(cur_polyline)
                cur_info['edge_1'] = edge_1
                cur_info['edge_2'] = edge_2

                map_infos['crosswalk'][cur_data.id] = cur_info

            else:
                continue

        return map_infos


    def get_map_features(self, origin, R, city_map = None):
        # initialization
        lane_ids = []
        lane_cps = [] #np.zeros((num_lanes, 3, 4, 2), dtype=float)
        lane_boundary_type = [] #np.zeros((num_lanes, 2), dtype=np.uint8)
        lane_type = [] #np.zeros((num_lanes), dtype=np.uint8)
        lane_is_intersection = [] #np.zeros(num_lanes, dtype=np.uint8)

        for lane_id, lane in self.map_infos['lane'].items():
            pts = lane['points']
            hash_id = hashlib.md5(pts.tostring()).hexdigest()

            cps_list = []

            if city_map is not None and hash_id in city_map.keys():
                cps_list = city_map[hash_id]
            else:
                recurrent_fit_line(pts=pts, cps_list=cps_list, degree=self.mapel_deg) # lane segments in Waymo are longer, we have break them into subsegments.
                city_map[hash_id] = deepcopy(cps_list)

            lane_cps = lane_cps + cps_list
            l_type = waymo_lane_type_converter(lane['type'])
            lane_type = lane_type + [l_type for _ in range(len(cps_list))]
            lane_ids = lane_ids + [lane_id for _ in range(len(cps_list))]
                
        num_lanes = len(lane_cps)

         # initialization
        cw_ids = []
        cw_cps = [] 
        cw_boundary_type = [] 
        cw_type = []                        

        for cw_id, cw in self.map_infos['crosswalk'].items():
            edge_1 = cw['edge_1']
            edge_2 = cw['edge_2']
            center = (edge_1 + edge_2)/2.

            cps = fit_line(center, degree = self.mapel_deg, use_borgespastva = False, num_sample_point=4)
            cps_reverse = cps[::-1]
            cw_cps.append(cps)
            cw_cps.append(cps_reverse)

            cw_type.append(waymo_lane_type_converter(cw['type']))
            cw_type.append(waymo_lane_type_converter(cw['type']))
            cw_ids.append(cw_id)

        num_cws = len(cw_cps)

        
        lane_cps = (np.array(lane_cps) - origin) @ R 
        lane_cps = np.concatenate([np.zeros((num_lanes, 2, self.mapel_deg+1, 2)), lane_cps[:, None, :, :]], axis = 1) # add dummy lane boundaries
        lane_boundary_type = np.zeros((num_lanes, 2), dtype=np.uint8)
        lane_type = np.array(lane_type, dtype=np.uint8)
        lane_is_intersection = np.ones(num_lanes, dtype=np.uint8) * 2

        
        if num_cws >0:
            cw_cps = (np.array(cw_cps)-origin)@R
            cw_cps = np.concatenate([np.zeros((num_cws, 2, self.mapel_deg+1, 2)), cw_cps[:, None, :, :]], axis = 1)
            cw_boundary_type = np.ones((num_cws, 2),  dtype=np.uint8) * 15 # 15 is the crosswalk boundary type
            cw_type = np.array(cw_type, dtype=np.uint8)

            mapel_ids = torch.tensor(np.array(lane_ids+cw_ids+cw_ids), dtype=torch.long)
            mapel_cps = torch.tensor(np.concatenate((lane_cps, cw_cps), axis=0), dtype=torch.float32)
            mapel_types = torch.tensor(np.concatenate((lane_type, cw_type), axis=0), dtype=torch.long)
            mapel_boundary_types = torch.tensor(np.concatenate((lane_boundary_type, cw_boundary_type), axis=0), dtype=torch.long)
            mapel_in_intersection = torch.tensor(np.concatenate((lane_is_intersection, np.ones(cw_type.shape)*2), axis=0), dtype=torch.long)
            num_mapels = num_lanes + num_cws
        else:
            mapel_ids = torch.tensor(np.array(lane_ids), dtype=torch.long)
            mapel_cps = torch.tensor(lane_cps , dtype=torch.float32)
            mapel_types = torch.tensor(lane_type, dtype=torch.long)
            mapel_boundary_types = torch.tensor(lane_boundary_type, dtype=torch.long)
            mapel_in_intersection = torch.tensor(lane_is_intersection, dtype=torch.long)
            num_mapels = num_lanes
        
        map_data = {
            'mapel_ids': mapel_ids,
            'mapel_cps': mapel_cps,
            'mapel_type': mapel_types,
            'mapel_boundary_type': mapel_boundary_types,
            'mapel_is_intersection': mapel_in_intersection,
            'num_nodes': num_mapels,
        }
            

        return map_data

            
################################ centerline ###########################################
def recurrent_fit_line(pts, cps_list, degree, current_iter =0, max_iter = 3):
    num_pts = pts.shape[0]
    lane_cps = fit_line(pts, degree=degree, use_borgespastva=True)

    if current_iter == max_iter:
        cps_list.append(lane_cps)
        return

    fit_error = np.linalg.norm(pts[[0, -1]] - lane_cps[[0, -1]], axis=-1)

    if np.max(fit_error) > 0.1 and num_pts >= 8:
        recurrent_fit_line(pts[:((num_pts // 2) +1)], cps_list, degree=degree, current_iter=current_iter + 1)
        recurrent_fit_line(pts[(num_pts // 2):], cps_list, degree=degree, current_iter=current_iter + 1)
    else:
        cps_list.append(lane_cps)
        return
    

def fit_line(line: np.ndarray,
             degree: int,
             use_borgespastva = False,
             num_sample_point = 12,
             maxiter = 2,
             no_clean = False):
    '''
    fit line and find control points.
    
    parameter:
        - line: [N, 2]
        
    return:
        - resampled (interpolated) line [deg + 1,2]
    '''
    if line.shape[0] == 2 or no_clean:
        l = resample_line(line, num_sample_point)
    else:
        l = resample_line(clean_lines(line)[0], num_sample_point) #av2.geometry.interpolate.interp_arc(num_sample_point, line)
        
    s = Spline(degree)
    cps = s.find_control_points(l)
    
    if use_borgespastva:
        t0 = s.initial_guess_from_control_points(l, cps)
        t, converged , errors = s.borgespastva(l, k = degree, t0 = t0, maxiter=maxiter)
        cps = s.find_control_points(l, t=t)
        
    return cps


def resample_line(line: np.ndarray, num_sample_point = 12):
    '''
    resample (interpolate) line with equal distance.
    
    parameter:
        - line: [N, 2]
        
    return:
        - resampled (interpolated) line [M,2]
    '''

    ls = LineString(line)
    s0 = 0
    s1 = ls.length
    
    return np.array([
        ls.interpolate(s).coords.xy
        for s in np.linspace(s0, s1, num_sample_point)
    ]).squeeze()


def clean_lines(lines):
    '''
    clean line points, which go backwards.
    
    parameter:
        - lines: list of lines with shape [N, 2]
        
    return:
        - cleaned list of lines with shape [M, 2]
    '''
    cleaned_lines = []
    if not isinstance(lines, list):
        lines = [lines]
    for candidate in lines:
        # remove duplicate points
        ds = np.linalg.norm(np.diff(candidate, axis=0), axis=-1) > 0.05
        keep = np.block([True, ds])
        
        cleaned = candidate[keep, :]
        
        # remove points going backward
        dx, dy = np.diff(cleaned, axis=0).T
        dphi = np.diff(np.unwrap(np.arctan2(dy, dx)))
        
        keep = np.block([True, dphi < (np.pi / 2), True])
        
        cleaned = cleaned[keep, :]
        
        cleaned_lines.append(cleaned)
        
    return cleaned_lines

def angle_between_vectors(v1, v2):
    return np.arccos(v1@v2.T / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

def side_to_directed_lineseg(
        query_point,
        start_point,
        end_point) -> str:
    cond = np.cross((end_point - start_point), (query_point - start_point))
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'
    

def line_string_to_xy(ls):
    x, y = ls.coords.xy
    return np.vstack([x,y]).T

def transform_cw_id(cw_id, additional_id):
    return int(str(cw_id) + str(cw_id) + str(additional_id))

def find_edges(xy):
    if xy.shape[0] != 4:
        polygon = Polygon(xy)
        rect = polygon.minimum_rotated_rectangle
        x, y = rect.boundary.xy
        xy = np.concatenate([np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)], axis=1)
    
    dist_1 = np.linalg.norm(xy[0] - xy[1], axis=-1)
    dist_2 = np.linalg.norm(xy[1] - xy[2], axis=-1)
    
    if dist_1 >= dist_2:
        return xy[:2], xy[[-1, -2]]
    else:
        return xy[[1,2]], xy[[0,3]]