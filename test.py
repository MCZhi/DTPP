import yaml
import datetime
import torch
import argparse
import warnings
from tqdm import tqdm
from planner import Planner
from common_utils import *
warnings.filterwarnings("ignore") 

from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.nuboard.base.data_class import NuBoardFile


def build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir):
    """
    Builds the main experiment folder for simulation.
    :return: The main experiment folder path.
    """
    print('Building experiment folders...')

    exp_folder = pathlib.Path(output_dir)
    print(f'\nFolder where all results are stored: {exp_folder}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Build nuboard event file.
    nuboard_filename = exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
    nuboard_file = NuBoardFile(
        simulation_main_path=str(exp_folder),
        simulation_folder=simulation_dir,
        metric_main_path=str(exp_folder),
        metric_folder=metric_dir,
        aggregator_metric_folder=aggregator_metric_dir,
    )

    metric_main_path = exp_folder / metric_dir
    metric_main_path.mkdir(parents=True, exist_ok=True)

    nuboard_file.save_nuboard_file(nuboard_filename)
    print('Building experiment folders...DONE!')

    return exp_folder.name


def build_simulation(experiment, planner, scenarios, output_dir, simulation_dir, metric_dir):
    runner_reports = []
    print(f'Building simulations from {len(scenarios)} scenarios...')

    metric_engine = build_metrics_engine(experiment, output_dir, metric_dir)
    print('Building metric engines...DONE\n')

    # Iterate through scenarios
    for scenario in tqdm(scenarios, desc='Running simulation'):
        # Ego Controller and Perception
        if experiment == 'open_loop_boxes':
            ego_controller = LogPlaybackController(scenario) 
            observations = TracksObservation(scenario)
        elif experiment == 'closed_loop_nonreactive_agents':
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], 
                                 r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, 
                                 jerk_penalty=1e-4, curvature_rate_penalty=1e-2, 
                                 stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controller = TwoStageController(scenario, tracker, motion_model) 
            observations = TracksObservation(scenario)
        elif experiment == 'closed_loop_reactive_agents':      
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], 
                                 r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, 
                                 jerk_penalty=1e-4, curvature_rate_penalty=1e-2, 
                                 stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controller = TwoStageController(scenario, tracker, motion_model) 
            observations = IDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                                     accel_max=1.0, decel_max=2.0, scenario=scenario,
                                     open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT"])
        else:
            raise ValueError(f"Invalid experiment type: {experiment}")
            
        # Simulation Manager
        simulation_time_controller = StepSimulationTimeController(scenario)

        # Stateful callbacks
        metric_callback = MetricCallback(metric_engine=metric_engine)
        sim_log_callback = SimulationLogCallback(output_dir, simulation_dir, "msgpack")

        # Construct simulation and manager
        simulation_setup = SimulationSetup(
            time_controller=simulation_time_controller,
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )

        simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=MultiCallback([metric_callback, sim_log_callback])
        )

        # Begin simulation
        simulation_runner = SimulationRunner(simulation, planner)
        report = simulation_runner.run()
        runner_reports.append(report)
    
    # save reports
    save_runner_reports(runner_reports, output_dir, 'runner_reports')

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0

    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            print("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    number_of_failures = len(scenarios) - number_of_successful
    print(f"Number of successful simulations: {number_of_successful}")
    print(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        print(f"Failed simulations [log, token]:\n{failed_simulations}")
    
    print('Finished running simulations!')

    return runner_reports


def build_nuboard(scenario_builder, simulation_path):
    nuboard = NuBoard(
        nuboard_paths=simulation_path,
        scenario_builder=scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=5006
    )

    nuboard.run()


def main(args):
    # parameters
    experiment_name = args.test_type  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    job_name = 'DTPP_planner'
    experiment_time = datetime.datetime.now()
    experiment = f"{experiment_name}/{job_name}/{experiment_time}"  
    output_dir = f"testing_log/{experiment}"
    simulation_dir = "simulation"
    metric_dir = "metrics"
    aggregator_metric_dir = "aggregator_metric"

    # initialize planner
    torch.set_grad_enabled(False)
    planner = Planner(model_path=args.model_path, device=args.device)

    # initialize main aggregator
    metric_aggregators = build_metrics_aggregators(experiment_name, output_dir, aggregator_metric_dir)
    metric_save_path = f"{output_dir}/{metric_dir}"
    metric_aggregator_callback = MetricAggregatorCallback(metric_save_path, metric_aggregators)
    metric_file_callback = MetricFileCallback(metric_file_output_path=f"{output_dir}/{metric_dir}",
                                              scenario_metric_paths=[f"{output_dir}/{metric_dir}"],
                                              delete_scenario_metric_files=True)
    metric_summary_callback = MetricSummaryCallback(metric_save_path=f"{output_dir}/{metric_dir}",
                                                    metric_aggregator_save_path=f"{output_dir}/{aggregator_metric_dir}",
                                                    summary_output_path=f"{output_dir}/summary",
                                                    num_bins=20, pdf_file_name='summary.pdf')
    main_callbacks = MultiMainCallback([metric_file_callback, metric_aggregator_callback, metric_summary_callback])
    main_callbacks.on_run_simulation_start()

    # build simulation folder
    build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir)

    # build scenarios
    print('Extracting scenarios...')
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)
    if args.load_test_set:
        params = yaml.safe_load(open('test_scenario.yaml', 'r'))
        scenario_filter = ScenarioFilter(**params)
    else:
        scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type))
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    scenarios = builder.get_scenarios(scenario_filter, worker)

    # begin testing
    print('Running simulations...')
    build_simulation(experiment_name, planner, scenarios, output_dir, simulation_dir, metric_dir)
    main_callbacks.on_run_simulation_end()
    simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']

    # show metrics and scenarios
    build_nuboard(builder, simulation_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--map_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_type', type=str, default='closed_loop_nonreactive_agents')
    parser.add_argument('--load_test_set', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scenarios_per_type', type=int, default=20)
    args = parser.parse_args()

    main(args)
