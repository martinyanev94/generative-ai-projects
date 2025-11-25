import torch.distributed as dist

def setup_distributed_environment():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def train_distributed_model(model, data_loader):
    for data in data_loader:
        output = model(data)
        # Proceed with loss calculation and backpropagation
