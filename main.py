import os
from opt import args
from solver import Solver
from dataset import get_train_loader, get_test_loader
from torch.backends import cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = args.nGPU

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.save_check):
        os.makedirs(config.save_check)
    if not os.path.exists(config.save_rec):
        os.makedirs(config.save_rec)

    if config.mode == 'train':
        train_dataset_loader = get_train_loader(config)
        solver = Solver(train_dataset_loader, config)
        solver.train(config)
    elif config.mode == 'rec':
        test_data_loader = get_test_loader(config)
        solver = Solver(test_data_loader, config)
        solver.rec(config)

if __name__ == '__main__':
    main(args)