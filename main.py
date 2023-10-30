import sys
sys.path.append("FedLab")

from omegaconf import OmegaConf
import galois
from feduv.trainer import FedUVSerialClientTrainer
from feduv.partitioned_celeba import PartitionedCelebA
from feduv.pipeline import FedUVPipeline
from feduv.handler import FedUVServerHandler
from feduv.cnn import CNN_CelebA
from tensorboardX import SummaryWriter
import torch
import os
from datetime import datetime
import argparse

def main(args):
    # set up model & dataset
    model = CNN_CelebA(args.code_length)
    bch = galois.BCH(args.code_length, args.message_length, args.d_min)
    celeba_parts = PartitionedCelebA(root=args.dataset_root,
                                     num_clients=args.num_clients,
                                     num_extra=args.num_extra,
                                     seed=42,
                                     normalize=args.normalize,
                                     bch=bch)
    celeba_parts.prepare()

    # set up serial trainer
    trainer = FedUVSerialClientTrainer(
        model, args.num_clients, cuda=args.cuda, device=args.device)

    trainer.setup_dataset(celeba_parts)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr,
                        args.lr_decay, args.lr_decay_step_size)

    # set up global server
    handler = FedUVServerHandler(model=model, global_round=args.com_round, num_clients=args.num_clients,
                                 sample_ratio=args.sample_ratio,
                                 validate_interval=args.validate_interval, cuda=args.cuda, device=args.device)

    # start
    with SummaryWriter(logdir=args.logdir) as writter:
        pipeline = FedUVPipeline(
            handler=handler, trainer=trainer, metric_writter=writter, verbose=args.verbose)
        pipeline.main()

    # save
    torch.save(handler.model.state_dict(),
               os.path.join(args.logdir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone FedUV for celeba")

    # dataset config
    parser.add_argument("--dataset_root", type=str, required=True)
    # not sepecified in paper
    parser.add_argument("--normalize", action="store_true")

    # logging config
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--verbose",
                        action="store_true", default=False)

    # client config
    parser.add_argument("--num_clients", type=int, default=1000)
    parser.add_argument("--num_extra", type=int, default=1000)
    parser.add_argument("--sample_ratio", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    # not sepecified in paper
    parser.add_argument("--batch_size", type=int, default=20)

    # training config
    parser.add_argument("--com_round", type=int, default=20000)
    parser.add_argument("--validate_interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.01)
    parser.add_argument("--lr_decay_step_size", type=float,
                        default=8000)  # not sepecified in paper
    parser.add_argument("--cuda",
                        action="store_true", default=False)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    # bch config
    parser.add_argument('--code_length', type=int, default=127,
                        help='Code length for BCH codeword')
    parser.add_argument('--message_length', type=int,
                        default=64, help='Message length for BCH codeword')
    parser.add_argument('--d_min', type=int, default=21,
                        help='D value for BCH codeword generation')

    args = parser.parse_args()

    args.logdir = args.logdir or os.path.join(
        "runs", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{args.comment}")
    os.makedirs(args.logdir, exist_ok=True)
    OmegaConf.save(OmegaConf.create(vars(args)), os.path.join(
        args.logdir, "params.yaml"))

    main(args)
