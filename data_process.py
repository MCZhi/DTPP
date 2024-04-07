import os
import math
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_utils import *
from trajectory_tree_planner import *
from common_utils import get_filter_parameters, get_scenario_map

from nuplan.planning.utils.multithreading.worker_pool import Task
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping


# define data processor
class DataProcessor(object):
    def __init__(self, scenarios):
        self._scenarios = scenarios

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.max_target_speed = 15 # [m/s]
        self.first_stage_horizon = 3 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 20

        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        self._radius = 80 # [m] query radius scope relative to the current pose.
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
            sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
        agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

        return agent_futures
    
    def get_ego_candidate_trajectories(self):
        planner = SplinePlanner(self.first_stage_horizon, self.future_time_horizon)

        # Gather information about the environment
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        observation = self.scenario.get_tracked_objects_at_iteration(0)
        ego_state = self.scenario.initial_ego_state
        route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            route_roadblocks.append(block)

        candidate_lane_edge_ids = [edge.id for block in route_roadblocks if block for edge in block.interior_edges]

        # Get obstacles
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                        TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                        TrackedObjectType.GENERIC_OBJECT]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
        obstacles = []
        for obj in objects:
            if obj.box.geometry.distance(ego_state.car_footprint.geometry) > 30:
                continue

            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.01:
                    obstacles.append(obj.box)
            else:
                obstacles.append(obj.box)

        # Get starting block
        starting_block = None
        cur_point = (self.scenario.initial_ego_state.rear_axle.x, self.scenario.initial_ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break

        # Get starting edges
        edges = get_candidate_edges(ego_state, starting_block)
        candidate_paths = get_candidate_paths(edges, ego_state, candidate_lane_edge_ids)
        paths = generate_paths(candidate_paths, obstacles, ego_state)
        speed_limit = edges[0].speed_limit_mps or self.max_target_speed

        # Initial tree (root node)
        # traj: x, y, heading, velocity, acceleration, curvature, time
        state = torch.tensor([[0, 0, 0, ego_state.dynamic_car_state.rear_axle_velocity_2d.x, 
                               ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, 0, 0]])
        tree = TrajTree(state, None, 0)

        # 1st stage expand
        tree.expand_children(paths, self.first_stage_horizon, speed_limit, planner)
        leaves = TrajTree.get_children(tree)
        first_trajs = np.stack([leaf.total_traj[1:].numpy() for leaf in leaves]).astype(np.float32)

        # 2nd stage expand
        for leaf in leaves:
            leaf.expand_children(paths, self.future_time_horizon - self.first_stage_horizon, speed_limit, planner)

        # Get all leaves
        leaves = TrajTree.get_children(leaves)
        second_trajs = np.stack([leaf.total_traj[1:].numpy() for leaf in leaves]).astype(np.float32)
        
        return first_trajs, second_trajs

    def plot_scenario(self, data):
        # Create map layers
        create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'])

        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1])
        create_agents_raster(data['neighbor_agents_past'][:, -1])

        # Draw past and future trajectories
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'][:1])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'][:1])

        # Draw candidate trajectories
        draw_plans(data['first_stage_ego_trajectory'], 1)
        draw_plans(data['second_stage_ego_trajectory'], 2)

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def work(self, save_dir, debug=False):
        for scenario in tqdm(self._scenarios):
            map_name = scenario._map_name
            token = scenario.token
            self.scenario = scenario
            self.map_api = scenario.map_api
            print(scenario)

            # get agent past tracks
            ego_agent_past, time_stamps_past = self.get_ego_agent()
            neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
            ego_agent_past, neighbor_agents_past, neighbor_indices = \
                    agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.num_agents)

            # get vector set map
            vector_map = self.get_map()

            # get agent future tracks
            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

            # get candidate trajectories
            try:
                first_stage_trajs, second_stage_trajs = self.get_ego_candidate_trajectories()
            except:
                print(f"Error in {map_name}_{token}")
                continue

            # check if the candidate trajectories are valid
            expert_error_1 = np.linalg.norm(ego_agent_future[None, self.first_stage_horizon*10-1, :2] 
                                            - first_stage_trajs[:, -1, :2], axis=-1)
            expert_error_2 = np.linalg.norm(ego_agent_future[None, self.future_time_horizon*10-1, :2] 
                                            - second_stage_trajs[:, -1, :2], axis=-1)       
            if np.min(expert_error_1) > 1.5 and np.min(expert_error_2) > 4:
                continue
            
            # sort the candidate trajectories
            first_stage_trajs = first_stage_trajs[np.argsort(expert_error_1)]
            second_stage_trajs = second_stage_trajs[np.argsort(expert_error_2)]            

            # gather data
            data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_agent_future": ego_agent_future, 
                    "first_stage_ego_trajectory": first_stage_trajs, "second_stage_ego_trajectory": second_stage_trajs,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future}
            data.update(vector_map)

            # visualization
            if debug:
                self.plot_scenario(data)

            # save to disk
            self.save_to_disk(save_dir, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--debug', action="store_true", help='if visualize the data output', default=False)
    parser.add_argument('--data_path', type=str, help='path to the data')
    parser.add_argument('--map_path', type=str, help='path to the map')    
    parser.add_argument('--save_path', type=str, help='path to save the processed data')
    parser.add_argument('--total_scenarios', type=int, help='total number of scenarios', default=None)

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)
    scenario_filter = ScenarioFilter(*get_filter_parameters(num_scenarios_per_type=30000, 
                                                            limit_total_scenarios=args.total_scenarios))
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of training scenarios: {len(scenarios)}")
    
    del worker, builder, scenario_filter
    processor = DataProcessor(scenarios)
    processor.work(args.save_path, debug=args.debug)
