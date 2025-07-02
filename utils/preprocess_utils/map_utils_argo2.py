'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from av2.map.map_api import ArgoverseStaticMap
from shapely import LineString, Point, Polygon
import numpy as np
from pathlib import Path
import hashlib
import torch

from utils.spline import Spline
from utils.preprocess_utils import data_types 


class AV2MapInterpreter():
    def __init__(self, 
                 map_file_path: str, 
                 mapel_deg: int):
        self.avm = ArgoverseStaticMap.from_json(Path(map_file_path))
        self.mapel_deg = mapel_deg
        self.s = Spline(self.mapel_deg)
        
        self.all_polygon_dict, self.vehicle_polygon_dict, self.bike_polygon_dict, self.bus_polygon_dict , self.pedes_polygon_dict = {}, {}, {}, {}, {}
        self.all_lane_ids, self.vehicle_lane_ids, self.bike_lane_ids, self.bus_lane_ids, self.pedes_lane_ids = [], [], [], [], []
        self.lane_center_dict = {}
        self.adjacent_lane_dict = {}
        
        self._generate_polygons_dict_()
        
    def _generate_polygons_dict_(self):
        self.all_lane_ids = self.avm.get_scenario_lane_segment_ids()
        self.vehicle_lane_ids = [l_id for l_id in self.all_lane_ids if data_types.POLYGON_TYPES.index(self.avm.vector_lane_segments[l_id].lane_type.value) == 0]
        self.bike_lane_ids = [l_id for l_id in self.all_lane_ids if data_types.POLYGON_TYPES.index(self.avm.vector_lane_segments[l_id].lane_type.value) == 1 ]
        self.bus_lane_ids = [l_id for l_id in self.all_lane_ids if data_types.POLYGON_TYPES.index(self.avm.vector_lane_segments[l_id].lane_type.value) == 2]
        self.pedes_lane_ids = [transform_cw_id(cw_id, 1) for cw_id in self.avm.vector_pedestrian_crossings.keys()] + [transform_cw_id(cw_id, 2) for cw_id in self.avm.vector_pedestrian_crossings.keys()]
        
        self.all_polygon_dict = {l_id: Polygon(self.avm.get_lane_segment_polygon(l_id)) for l_id in self.all_lane_ids}
        self.vehicle_polygon_dict = {l_id: Polygon(self.avm.get_lane_segment_polygon(l_id)) for l_id in self.vehicle_lane_ids}
        self.bike_polygon_dict = {l_id: Polygon(self.avm.get_lane_segment_polygon(l_id)) for l_id in self.bike_lane_ids}
        self.bus_polygon_dict = {l_id: Polygon(self.avm.get_lane_segment_polygon(l_id)) for l_id in self.bus_lane_ids}
        pedes_polygon_dict_1 = {transform_cw_id(cw_id, 1): Polygon(self.avm.vector_pedestrian_crossings[cw_id].polygon) for cw_id in self.avm.vector_pedestrian_crossings.keys()}
        pedes_polygon_dict_2 = {transform_cw_id(cw_id, 2): Polygon(self.avm.vector_pedestrian_crossings[cw_id].polygon) for cw_id in self.avm.vector_pedestrian_crossings.keys()}
        pedes_polygon_dict_1.update(pedes_polygon_dict_2)
        self.pedes_polygon_dict = pedes_polygon_dict_1
        
        self.lane_center_dict = {l_id: LineString(self.avm.get_lane_segment_centerline(l_id)[:, :2]) for l_id in self.all_lane_ids}
        
        cw_center_dict_1, cw_center_dict_2 = {}, {}
        for cw_id in self.avm.vector_pedestrian_crossings.keys():
            edge = np.mean(np.array(self.avm.vector_pedestrian_crossings[cw_id].get_edges_2d()), axis=0)
            cw_center_dict_1[transform_cw_id(cw_id, 1)] = LineString(edge)
            cw_center_dict_2[transform_cw_id(cw_id, 2)] = LineString(edge[::-1])
 
        self.lane_center_dict.update(cw_center_dict_1)
        self.lane_center_dict.update(cw_center_dict_2)
        
    
################################ Map Features ################################################
    def get_map_features(self, origin, R, city_map = None):
        lane_segment_ids = self.all_lane_ids
        num_lanes = len(lane_segment_ids)

        # initialization
        lane_cps = np.zeros((num_lanes, 3, self.mapel_deg+1, 2), dtype=float)
        lane_boundary_type = np.zeros((num_lanes, 2), dtype=np.uint8)
        lane_type = np.zeros((num_lanes), dtype=np.uint8)
        lane_is_intersection = np.zeros(num_lanes, dtype=np.uint8)
        
        for lane_segment in self.avm.get_scenario_lane_segments():
            lane_segment_idx = lane_segment_ids.index(lane_segment.id)
            left_bound = self.avm.vector_lane_segments[lane_segment.id].left_lane_boundary
            right_bound = self.avm.vector_lane_segments[lane_segment.id].right_lane_boundary
            
            hash_id = hashlib.md5(np.concatenate([left_bound.xyz, right_bound.xyz], axis = 0).tostring()).hexdigest()
            
            if city_map is None:
                lane_cps[lane_segment_idx, 0] = fit_line(left_bound.xyz[:, :2], degree = self.mapel_deg, use_borgespastva=True if left_bound.xyz.shape[0] > 6 else False)
                lane_cps[lane_segment_idx, 1] = fit_line(right_bound.xyz[:, :2], degree = self.mapel_deg, use_borgespastva=True if right_bound.xyz.shape[0] > 6 else False)
                lane_cps[lane_segment_idx, 2] = fit_line(self.avm.get_lane_segment_centerline(lane_segment.id)[:, :2], degree = self.mapel_deg, no_clean = True, use_borgespastva=True\
                                                                 if left_bound.xyz.shape[0] > 6 or right_bound.xyz.shape[0] > 6 else False)
            elif hash_id in city_map.keys():
                lane_cps[lane_segment_idx] = city_map[hash_id]
            else:
                lane_cps[lane_segment_idx, 0] = fit_line(left_bound.xyz[:, :2], degree = self.mapel_deg, use_borgespastva=True if left_bound.xyz.shape[0] > 6 else False)
                lane_cps[lane_segment_idx, 1] = fit_line(right_bound.xyz[:, :2], degree = self.mapel_deg, use_borgespastva=True if right_bound.xyz.shape[0] > 6 else False)
                lane_cps[lane_segment_idx, 2] = fit_line(self.avm.get_lane_segment_centerline(lane_segment.id)[:, :2], degree = self.mapel_deg, no_clean = True, use_borgespastva=True\
                                                                 if (left_bound.xyz.shape[0] > 6 or right_bound.xyz.shape[0] > 6) else False)
                city_map[hash_id] = lane_cps[lane_segment_idx]
            
            lane_boundary_type[lane_segment_idx, 0] = data_types.POINT_TYPES.index(self.avm.vector_lane_segments[lane_segment.id].left_mark_type.value)
            lane_boundary_type[lane_segment_idx, 1] = data_types.POINT_TYPES.index(self.avm.vector_lane_segments[lane_segment.id].right_mark_type.value)
            
            lane_type[lane_segment_idx] = data_types.POLYGON_TYPES.index(lane_segment.lane_type.value)
            lane_is_intersection[lane_segment_idx] = data_types.POLYGON_IS_INTERSECTIONS.index(lane_segment.is_intersection)
            
            # # Check curvature
            # if np.max(self.s.curvature(lane_cps[lane_segment_idx, 0], return_abs = True)) > self.curvature_tol or \
            #    np.max(self.s.curvature(lane_cps[lane_segment_idx, 1], return_abs = True)) > self.curvature_tol:
            #     outlier_lane_ids.append(lane_segment.id)
        
        
        cross_walk_ids = list(self.avm.vector_pedestrian_crossings.keys())
        num_cws = len(cross_walk_ids) * 2 # bi-directional
        
        # initialization
        cw_cps = np.zeros((num_cws, 3, self.mapel_deg+1, 2), dtype=np.float32)
        cw_boundary_type = np.zeros((num_cws, 2), dtype=np.uint8)
        cw_type = np.zeros((num_cws), dtype=np.uint8)

        
        for crosswalk in self.avm.get_scenario_ped_crossings():
            crosswalk_idx = cross_walk_ids.index(crosswalk.id)
            edge1 = crosswalk.edge1.xyz[:, :2]
            edge2 = crosswalk.edge2.xyz[:, :2]
            
            edge1_cps = fit_line(edge1, degree = self.mapel_deg, use_borgespastva=False, num_sample_point=4)
            edge2_cps = fit_line(edge2, degree = self.mapel_deg, use_borgespastva=False, num_sample_point=4)
            
            edge1_cps_reverse = edge1_cps[::-1, :]
            edge2_cps_reverse = edge2_cps[::-1, :]
            
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            
            if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
                cw_cps[crosswalk_idx, 0] = edge1_cps
                cw_cps[crosswalk_idx, 1] = edge2_cps
                cw_cps[crosswalk_idx, 2] = (edge1_cps + edge2_cps) / 2.

                cw_cps[crosswalk_idx + len(cross_walk_ids), 0] = edge2_cps_reverse
                cw_cps[crosswalk_idx + len(cross_walk_ids), 1] = edge1_cps_reverse
                cw_cps[crosswalk_idx + len(cross_walk_ids), 2] = (edge1_cps_reverse + edge2_cps_reverse)/2.
            else:
                cw_cps[crosswalk_idx, 0] = edge2_cps
                cw_cps[crosswalk_idx, 1] = edge1_cps
                cw_cps[crosswalk_idx, 2] = (edge1_cps + edge2_cps) / 2.

                cw_cps[crosswalk_idx + len(cross_walk_ids), 0] = edge1_cps_reverse
                cw_cps[crosswalk_idx + len(cross_walk_ids), 1] = edge2_cps_reverse
                cw_cps[crosswalk_idx + len(cross_walk_ids), 2] = (edge1_cps_reverse + edge2_cps_reverse)/2.
            
            cw_boundary_type[crosswalk_idx, 0] = data_types.POINT_TYPES.index('CROSSWALK')
            cw_boundary_type[crosswalk_idx, 1] = data_types.POINT_TYPES.index('CROSSWALK')
            cw_boundary_type[crosswalk_idx + len(cross_walk_ids), 0] = data_types.POINT_TYPES.index('CROSSWALK')
            cw_boundary_type[crosswalk_idx + len(cross_walk_ids), 1] = data_types.POINT_TYPES.index('CROSSWALK')
            
            cw_type[crosswalk_idx] = data_types.POLYGON_TYPES.index('PEDESTRIAN')
            cw_type[crosswalk_idx + len(cross_walk_ids)] = data_types.POLYGON_TYPES.index('PEDESTRIAN')
        
        

        lane_cps = (lane_cps - origin) @ R
        cw_cps = (cw_cps - origin) @ R

        mapel_ids = torch.tensor(np.array(lane_segment_ids+cross_walk_ids+cross_walk_ids), dtype=torch.long)
        mapel_cps = np.concatenate((lane_cps, cw_cps), axis=0)
        mapel_types = torch.tensor(np.concatenate((lane_type, cw_type), axis=0), dtype=torch.long)
        mapel_boundary_types = torch.tensor(np.concatenate((lane_boundary_type, cw_boundary_type), axis=0), dtype=torch.long)
        mapel_in_intersection = torch.tensor(np.concatenate((lane_is_intersection, np.ones(cw_type.shape)*2), axis=0), dtype=torch.long)
        num_mapels = num_lanes + num_cws
        
        map_data = {
            'mapel_ids': mapel_ids,
            'mapel_cps': torch.tensor(mapel_cps, dtype=torch.float32),
            'mapel_type': mapel_types,
            'mapel_boundary_type': mapel_boundary_types,
            'mapel_is_intersection': mapel_in_intersection,
            'num_nodes': num_mapels,
        }

        return map_data
    
################################ centerline ###########################################
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
        l = resample_line(clean_lines(line)[0], num_sample_point)
        
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
    

def transform_cw_id(cw_id, additional_id):
    return int(str(cw_id) + str(cw_id) + str(additional_id))
