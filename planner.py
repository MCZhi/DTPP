import math
import time
import matplotlib.pyplot as plt
from shapely import Point, LineString
from planner_utils import *
from obs_adapter import *
from trajectory_tree_planner import TreePlanner
from scenario_tree_prediction import *

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


class Planner(AbstractPlanner):
    def __init__(self, model_path, device):
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._N_points = int(T/DT)
        self._model_path = model_path
        self._device = device

    def name(self) -> str:
        return "DTPP Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TreePlanner(self._device, self._encoder, self._decoder)

    def _initialize_model(self):
        model = torch.load(self._model_path, map_location=self._device)
        self._encoder = Encoder()
        self._encoder.load_state_dict(model['encoder'])
        self._encoder.to(self._device)
        self._encoder.eval()
        self._decoder = Decoder()
        self._decoder.load_state_dict(model['decoder'])
        self._decoder.to(self._device)
        self._decoder.eval()

    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

    def compute_planner_trajectory(self, current_input: PlannerInput):
        # Extract iteration, history, and traffic light
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state

        # Construct input features
        start_time = time.perf_counter()
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # Get starting block
        starting_block = None
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
        
        # Get traffic light lanes
        traffic_light_lanes = []
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                traffic_light_lanes.append(lane_conn)

        # Tree policy planner
        try:
            plan = self._trajectory_planner.plan(iteration, ego_state, features, starting_block, self._route_roadblocks, 
                                             self._candidate_lane_edge_ids, traffic_light_lanes, observation)
        except Exception as e:
            print("Error in planning")
            print(e)
            plan = np.zeros((self._N_points, 3))
            
        # Convert relative poses to absolute states and wrap in a trajectory object
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)
        print(f'Step {iteration+1} Planning time: {time.perf_counter() - start_time:.3f} s')

        return trajectory