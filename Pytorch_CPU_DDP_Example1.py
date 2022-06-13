import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import pandas as pd

"""
간단한 모델로의 예시
world_size는 사용가능한 GPU의 갯수를 의미한다.
"""
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '8892'
device = torch.device("cpu")


def example(rank, world_size):
  #default process group 을 만듬
  dist.init_process_group("gloo", rank = rank, world_size = world_size)
  
  #간단한 모델
  model = nn.Linear(10, 10)
  
  #DDP model 만들기
  ddp_model = DDP(model)
  
  loss_fn = nn.MSELoss()
  optimizer = optim.SGD(ddp_model.parameters(), lr = 0.001)
  
  #순전파
  
  for i in range(1500):

    outputs = ddp_model(torch.randn(20, 10).to(device))
    labels = torch.randn(20, 10).to(device)
  
    #역전파
    loss_fn(outputs, labels).backward()
  
    #parameter 업데이트
    optimizer.step()
    print('one step')
def main():
  world_size = 2
  mp.spawn(example, args = (world_size, ), nprocs = world_size, join = True)
  
if __name__ == "__main__":
  main()
