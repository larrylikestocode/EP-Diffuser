'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import numpy as np
import tensorflow as tf
from pathlib import Path

from google.protobuf import text_format
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wosac_metrics
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils import trajectory_utils


def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
  """Loads the `SimAgentMetricsConfig` used for the challenge."""
  # pylint: disable=line-too-long
  # pyformat: disable
  config_path = (
                Path(wosac_metrics.__file__).parent / "challenge_2024_config.textproto"
            )
  with open(config_path, "r") as f:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), config)
  return config


# Load the test configuration.
config = load_metrics_config()

def joint_scene_from_states(
    states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.JointScene:
  # States shape: (num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  states = states.numpy()
  simulated_trajectories = []
  for i_object in range(len(object_ids)):
    simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
        center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
        center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
        object_id=object_ids[i_object]
    ))
  return sim_agents_submission_pb2.JointScene(
      simulated_trajectories=simulated_trajectories)


# Now we can replicate this strategy to export all the parallel simulations.
def scenario_rollouts_from_states(
    scenario: scenario_pb2.Scenario,
    states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.ScenarioRollouts:
  # States shape: (num_rollouts, num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  joint_scenes = []
  for i_rollout in range(states.shape[0]):
    joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids))
  return sim_agents_submission_pb2.ScenarioRollouts(
      # Note: remember to include the Scenario ID in the proto message.
      joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)


def get_logged_trajectories(scenario: scenario_pb2.Scenario):
    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
    logged_trajectories = logged_trajectories.gather_objects_by_id(
      tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario)))
    
    logged_trajectories = logged_trajectories.slice_time(
      start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
    
    return logged_trajectories


def get_gt_trajectories(scenario: scenario_pb2.Scenario):
    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
    logged_trajectories = logged_trajectories.gather_objects_by_id(
      tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario)))
    
    gt_trajectories = logged_trajectories.slice_time(
      start_index=submission_specs.CURRENT_TIME_INDEX + 1, end_index=submission_specs.N_FULL_SCENARIO_STEPS)
    
    return gt_trajectories


def get_sim_mask_argo2(scenario):
    sim_agent_ids = submission_specs.get_sim_agent_ids(scenario)
    sim_mask = np.zeros(len(scenario.tracks), dtype=bool)
    for i, track in enumerate(scenario.tracks):
        if track.id in sim_agent_ids:
            sim_mask[i] = True
    
    return sim_mask

def get_sim_mask_waymo(scenario):
    sim_agent_ids = submission_specs.get_sim_agent_ids(scenario)
    sim_mask = np.zeros(len(scenario.tracks), dtype=bool)
    sim_mask_valid = np.zeros(len(scenario.tracks), dtype=bool)
    for i, track in enumerate(scenario.tracks):
        if track.id in sim_agent_ids:
            sim_mask[i] = True
        
        count=0
        for step in range(submission_specs.CURRENT_TIME_INDEX+1):
            if track.states[step].valid:
                count += 1
        
        if count > 0:
            sim_mask_valid[i] = True

    return sim_mask[sim_mask_valid]


def get_result(scenario, simulated_states = None):
    result = {}
    logged_trajectories = get_logged_trajectories(scenario)
    
    scenario_rollouts = scenario_rollouts_from_states(scenario, simulated_states, logged_trajectories.object_id)
    # As before, we can validate the message we just generate.
    submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)
    
    scenario_metrics = wosac_metrics.compute_scenario_metrics_for_bundle(
    config, scenario, scenario_rollouts)
    
    result['scenario_id']=scenario.scenario_id
    result['metametric']=scenario_metrics.metametric

    
    result['average_displacement_error'] = scenario_metrics.average_displacement_error
    result['linear_speed_likelihood'] = scenario_metrics.linear_speed_likelihood
    result['linear_acceleration_likelihood'] = scenario_metrics.linear_acceleration_likelihood
    result['angular_speed_likelihood'] = scenario_metrics.angular_speed_likelihood
    result['angular_acceleration_likelihood'] = scenario_metrics.angular_acceleration_likelihood
    result['distance_to_nearest_object_likelihood'] = scenario_metrics.distance_to_nearest_object_likelihood
    result['collision_indication_likelihood'] = scenario_metrics.collision_indication_likelihood
    result['time_to_collision_likelihood'] = scenario_metrics.time_to_collision_likelihood
    result['distance_to_road_edge_likelihood'] = scenario_metrics.distance_to_road_edge_likelihood
    result['offroad_indication_likelihood'] = scenario_metrics.offroad_indication_likelihood
    result['min_average_displacement_error'] = scenario_metrics.min_average_displacement_error
    result['simulated_collision_rate'] = scenario_metrics.simulated_collision_rate
    result['simulated_offroad_rate'] = scenario_metrics.simulated_offroad_rate    
    
    aggregated_metrics = wosac_metrics.aggregate_metrics_to_buckets(
            config, scenario_metrics
        )
    
    result['kinematic_metrics']=aggregated_metrics.kinematic_metrics
    result['interactive_metrics']=aggregated_metrics.interactive_metrics
    result['map_based_metrics']=aggregated_metrics.map_based_metrics

    return result