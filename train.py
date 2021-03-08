import open3d as o3d  # prevent loading error

import os
import sys
import json
import logging
import torch
from easydict import EasyDict as edict

from lib.data_loaders import make_data_loader
from config import get_config

from lib.trainer import ContrastiveLossTrainer, HardestContrastiveLossTrainer, \
    TripletLossTrainer, HardestTripletLossTrainer

import pickle
from datetime import datetime
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def get_trainer(trainer):
  if trainer == 'ContrastiveLossTrainer':
    return ContrastiveLossTrainer
  elif trainer == 'HardestContrastiveLossTrainer':
    return HardestContrastiveLossTrainer
  elif trainer == 'TripletLossTrainer':
    return TripletLossTrainer
  elif trainer == 'HardestTripletLossTrainer':
    return HardestTripletLossTrainer
  else:
    raise ValueError(f'Trainer {trainer} not found')


def main(config, resume=False):
  seed = np.random.randint(1)
  world_size = torch.cuda.device_count()
  print("%d GPUs are available" % world_size)
  mp.spawn(train_parallel, nprocs=world_size, args=(world_size,seed, config))  

def train_parallel(rank, world_size, seed, config):
  # This function is performed in parallel in several processes, one for each available GPU
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '8887'
  dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.cuda.set_device(rank)
  device = 'cuda:%d' % torch.cuda.current_device()
  print("process %d, GPU: %s" % (rank, device))

  train_loader = make_data_loader(
      config,
      config.train_phase,
      config.batch_size,
      num_threads=config.train_num_thread,
      rank=rank, world_size=world_size, seed=seed)

  if config.test_valid:
    val_loader = make_data_loader(
        config,
        config.val_phase,
        config.val_batch_size,
        num_threads=config.val_num_thread,
        rank=rank, world_size=world_size, seed=seed)
  else:
    val_loader = None

  Trainer = get_trainer(config.trainer)
  trainer = Trainer(
      config=config,
      data_loader=train_loader,
      val_data_loader=val_loader,
      rank=rank
  )

  trainer.train()


if __name__ == "__main__":

  with open('train_fcgf_kitti_argv.pickle', 'rb') as fid:
    sys.argv = pickle.load(fid)  
    
  logger = logging.getLogger()
  config = get_config()
  config.out_dir = config.out_dir.replace('2021-03-02_15-29-31',datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
  print("out_dir: %s" % config.out_dir)
  with open ('out_dir.txt', 'w') as fid:
    fid.write(config.out_dir + '\n')
  
  config.batch_size = 6
  
  if False: # AD DEL
    config.batch_size = 1
    config.max_epoch = 3
    config.test_valid = False
    config.train_num_thread = 1
    #config.stat_freq = 20
    # local_file = './outputs/Experiments/KITTINMPairDataset-v0.3/HardestContrastiveLossTrainer/ResUNetBN2C/SGD-lr1e-1-e200-b8i1-modelnout32/2021-03-02_15-29-31/best_val_checkpoint.pth'
    # remote_file = 'remote_checkpoint.pth'
    # config.weights = remote_file

  if not(os.path.isdir(config.out_dir)):
    os.makedirs(config.out_dir)

  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  main(config)
