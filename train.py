import os
import csv
import glob
import torch
import argparse
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from scenario_tree_prediction import Encoder, Decoder
from torch.utils.data import DataLoader
from train_utils import *


def train_epoch(data_loader, encoder, decoder, optimizer):
    epoch_loss = []
    epoch_metrics = []
    encoder.train()
    decoder.train()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            # prepare data for predictor
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_gt_future = batch[5].to(args.device)
            neighbors_gt_future = batch[6].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)

            # encode
            optimizer.zero_grad()
            encoder_outputs = encoder(inputs)

            # first stage prediction
            first_stage_trajectory = batch[7].to(args.device)
            neighbors_trajectories, scores, ego, weights = \
                decoder(encoder_outputs, first_stage_trajectory, inputs['neighbor_agents_past'], 30)
            loss = calc_loss(neighbors_trajectories, first_stage_trajectory, ego, scores, weights, \
                             ego_gt_future, neighbors_gt_future, neighbors_future_valid)

            # second stage prediction
            second_stage_trajectory = batch[8].to(args.device)
            neighbors_trajectories, scores, ego, weights = \
                decoder(encoder_outputs, second_stage_trajectory, inputs['neighbor_agents_past'], 80)
            loss += 0.2 * calc_loss(neighbors_trajectories, second_stage_trajectory, ego, scores, weights, \
                              ego_gt_future, neighbors_gt_future, neighbors_future_valid)

            # loss backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()

            # compute metrics
            metrics = calc_metrics(second_stage_trajectory, neighbors_trajectories, scores, \
                                   ego_gt_future, neighbors_gt_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [planningADE, planningFDE, predictionADE, predictionFDE]
    logging.info(f"plannerADE: {planningADE:.4f}, plannerFDE: {planningFDE:.4f}, " +
                 f"predictorADE: {predictionADE:.4f}, predictorFDE: {predictionFDE:.4f}\n")
        
    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, encoder, decoder):
    epoch_loss = []
    epoch_metrics = []
    encoder.eval()
    decoder.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            # prepare data for predictor
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_gt_future = batch[5].to(args.device)
            neighbors_gt_future = batch[6].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)

            # predict
            with torch.no_grad():
                encoder_outputs = encoder(inputs)

                # first stage prediction
                first_stage_trajectory = batch[7].to(args.device)
                neighbors_trajectories, scores, ego, weights = \
                    decoder(encoder_outputs, first_stage_trajectory, inputs['neighbor_agents_past'], 30)
                loss = calc_loss(neighbors_trajectories, first_stage_trajectory, ego, scores, weights, \
                                 ego_gt_future, neighbors_gt_future, neighbors_future_valid)

                # second stage prediction
                second_stage_trajectory = batch[8].to(args.device)
                neighbors_trajectories, scores, ego, weights = \
                    decoder(encoder_outputs, second_stage_trajectory, inputs['neighbor_agents_past'], 80)
                loss += 0.2 * calc_loss(neighbors_trajectories, second_stage_trajectory, ego, scores, weights, \
                                  ego_gt_future, neighbors_gt_future, neighbors_future_valid)
 
            # compute metrics
            metrics = calc_metrics(second_stage_trajectory, neighbors_trajectories, scores, \
                                   ego_gt_future, neighbors_gt_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [planningADE, planningFDE, predictionADE, predictionFDE]
    logging.info(f"val-plannerADE: {planningADE:.4f}, val-plannerFDE: {planningFDE:.4f}, " +
                 f"val-predictorADE: {predictionADE:.4f}, val-predictorFDE: {predictionFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics


def model_training(args):
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up model
    encoder = Encoder().to(args.device)
    logging.info("Encoder Params: {}".format(sum(p.numel() for p in encoder.parameters())))
    decoder = Decoder(neighbors=args.num_neighbors, max_branch=args.num_candidates, \
                      variable_cost=args.variable_weights).to(args.device)
    logging.info("Decoder Params: {}".format(sum(p.numel() for p in decoder.parameters())))

    # set up optimizer
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(glob.glob(os.path.join(args.train_set, '*.npz')), args.num_neighbors, args.num_candidates)
    valid_set = DrivingData(glob.glob(os.path.join(args.valid_set, '*.npz')), args.num_neighbors, args.num_candidates)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    
    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        train_loss, train_metrics = train_epoch(train_loader, encoder, decoder, optimizer)
        val_loss, val_metrics = valid_epoch(valid_loader, encoder, decoder)

        # save to training log
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
               'train-planningADE': train_metrics[0], 'train-planningFDE': train_metrics[1], 
               'train-predictionADE': train_metrics[2], 'train-predictionFDE': train_metrics[3],
               'val-planningADE': val_metrics[0], 'val-planningFDE': val_metrics[1], 
               'val-predictionADE': val_metrics[2], 'val-predictionFDE': val_metrics[3]}

        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        model = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}
        torch.save(model, f'training_log/{args.name}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name', default="DTPP_training")
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--train_set', type=str, help='path to training data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--num_candidates', type=int, help='number of max candidate trajectories', default=30)
    parser.add_argument('--variable_weights', type=bool, help='use variable cost weights', default=False)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=2e-4)
    parser.add_argument('--device', type=str, help='run on which device', default='cuda')
    args = parser.parse_args()

    # Run model training
    model_training(args)