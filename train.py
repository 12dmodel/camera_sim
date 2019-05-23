# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import OnTheFlyDataset
from models.model_utils import get_model, ModelWrapper
from data_generation.pipeline import ImageDegradationPipeline
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint
from utils.training_util import read_config, torch2numpy
from utils.training_util import create_vis, calculate_psnr
from logger import Logger

import numpy as np
import time
import shutil
import setproctitle
import skimage


def train(config, restart_training, num_workers, num_threads):
    torch.set_num_threads(num_threads)
    print("Using {} CPU threads".format(torch.get_num_threads()))

    # TODO: de-hardcode this one.
    N_CHANNEL = 3
    train_config = config["training"]

    batch_size = train_config["batch_size"]
    lr = train_config["learning_rate"]
    w_decay = train_config["weight_decay"]
    step_size = train_config["decay_steps"]
    gamma = train_config["lr_decay"]
    betas = (train_config["beta1"], train_config["beta2"])
    n_epochs = train_config["num_epochs"]

    dataset_configs = train_config["dataset_configs"]
    use_cache = train_config["use_cache"]

    print("Configs:", config)
    # create dir for model
    checkpoint_dir = train_config["checkpoint_dir"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logger = Logger(train_config["logs_dir"])

    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))

    print("Using On the fly TRAIN datasets")
    train_data = OnTheFlyDataset(train_config["dataset_configs"],
                                im_size=(train_config["image_width"],
                                         train_config["image_height"]),
                                use_cache=use_cache)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = get_model(config["architecture"])

    l1_loss = nn.SmoothL1Loss()

    if use_gpu:
        ts = time.time()
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=num_gpu)
        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))


    # for sanity check
    all_parameters = [p for n,p in model.named_parameters() if p.requires_grad]
    if train_config["optimizer"] == "adam":
        print("Using Adam.")
        optimizer = optim.Adam([
                                {'params': all_parameters},
                               ],
                               lr=lr, betas=betas, weight_decay=w_decay, amsgrad=True)
    elif train_config["optimizer"] == "sgd":
        print("Using SGD.")
        optimizer = optim.SGD([
                                {'params': all_parameters},
                              ],
                              lr=lr, momentum=betas[0], weight_decay=w_decay)
    else:
        raise ValueError("Optimizer must be 'sgd' or 'adam', received '{}'".format(train_config["optimizer"]))

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    n_global_iter = 0
    average_loss = MovingAverage(train_config["n_loss_average"])
    best_loss = np.inf
    checkpoint_loaded = False
    if not restart_training:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, 'best')
            start_epoch = checkpoint['epoch']
            n_global_iter = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            checkpoint_loaded = True
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        except:
            start_epoch = 0
            n_global_iter = 0
            best_loss = np.inf
            print("=> load checkpoint failed, training from scratch")
    else:
        start_epoch = 0
        print("=> training from scratch")

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        ts = time.time()
        t4 = None
        t_generate_data = []
        t_train_disc = []
        t_train_gen = []
        t_vis = []
        t_save = []

        for iter, batch in enumerate(train_loader):
            if t4 is not None:
                # collect information and print out average time.
                t0_old = t0
            t0 = time.time()
            if t4 is not None:
                t_generate_data.append(t0 - t4)
                t_train_disc.append(t1 - t0_old)
                t_train_gen.append(t2 - t1)
                t_vis.append(t3 - t2)
                t_save.append(t4 - t3)
                N_report = 100
                N_print = 1000
                if (iter % N_report) == 0:
                    t_generate_data = np.mean(t_generate_data)
                    t_train_disc = np.mean(t_train_disc)
                    t_train_gen = np.mean(t_train_gen)
                    t_vis = np.mean(t_vis)
                    t_save = np.mean(t_save)
                    t_total = t_generate_data + t_train_disc + t_train_gen + t_vis + t_save
                    if (iter % N_print) == 0:
                        print("t_generate_data: {:0.4g} s ({:0.4g}%)".format(t_generate_data, t_generate_data / t_total * 100))
                        print("t_train_disc: {:0.4g} s ({:0.4g}%)".format(t_train_disc, t_train_disc / t_total * 100))
                        print("t_train_gen: {:0.4g} s ({:0.4g}%)".format(t_train_gen, t_train_gen / t_total * 100))
                        print("t_vis: {:0.4g} s ({:0.4g}%)".format(t_vis, t_vis / t_total * 100))
                        print("t_save: {:0.4g} s ({:0.4g}%)".format(t_save, t_save / t_total * 100))
                    logger.scalar_summary('Steps per sec', 1.0 / t_total, n_global_iter)
                    t_generate_data = []
                    t_train_disc = []
                    t_train_gen = []
                    t_vis = []
                    t_save = []

            should_vis = ((n_global_iter + 1) % train_config["vis_freq"]) == 0
            if use_gpu:
                degraded_img = batch['degraded_img'].cuda()
                target_img = batch['original_img'].cuda()
            else:
                degraded_img = batch['degraded_img']
                target_img = batch['original_img']
            t1 = time.time()

            optimizer.zero_grad()
            # Run the input through the model.
            output_img = model(degraded_img)
            loss = l1_loss(output_img, target_img)
            loss.backward()
            optimizer.step()
            logger.scalar_summary('Loss', loss.data[0], n_global_iter)
            psnr = calculate_psnr(output_img, target_img)
            logger.scalar_summary('Train PSNR', psnr, n_global_iter)

            average_loss.update(loss.data[0])
            t2 = time.time()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}" \
                        .format(epoch, iter, loss.data[0]))
            n_global_iter += 1

            if should_vis:
                exp = batch['vis_exposure'] if 'vis_exposure' in batch else None
                img = create_vis(degraded_img[:, :3, ...], target_img, output_img, exp)
                logger.image_summary("Train Images", img, n_global_iter)

            t3 = time.time()
            if (n_global_iter % train_config["save_freq"]) == 0:
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False
                save_dict = {
                    'epoch': epoch,
                    'global_iter': n_global_iter,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }
                save_checkpoint(save_dict,
                    is_best,
                    checkpoint_dir,
                    n_global_iter)
            t4 = time.time()

        print("Finish epoch {}, time elapsed {}" \
                .format(epoch, time.time() - ts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', dest='config_file', required=True, help='path to config file')
    parser.add_argument('--config_spec', dest='config_spec', default='denoiser_specs/configspec.conf', help='path to config spec file')
    parser.add_argument('--restart', action='store_true', help="Whether to remove old files and restart training")
    parser.add_argument('--force_expname', action='store_true', help="Whether to ignore exp name check")
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers in dataloader.')
    parser.add_argument('--num_threads', default=16, type=int, help='number of threads in dataloader.')
    args = parser.parse_args()
    config = read_config(args.config_file, args.config_spec)
    exp_name = config["exp_name"]
    config_name, _ = os.path.splitext(os.path.basename(args.config_file))
    if exp_name != config_name and not args.force_expname:
        raise ValueError("Exp name ({}) and config name ({}) don't match. Use --force_expname to override.".format(exp_name, config_name))

    setproctitle.setproctitle('train_{}'.format(exp_name))
    if args.restart:
        if os.path.exists(config["training"]["checkpoint_dir"]):
            shutil.rmtree(config["training"]["checkpoint_dir"])
        if os.path.exists(config["training"]["logs_dir"]):
            shutil.rmtree(config["training"]["logs_dir"])
    train(config, args.restart, args.num_workers, args.num_threads)

