# -*- coding: utf-8 -*-
import time
from pathlib import Path
import random

import torch
from supar.utils import Config
from supar.utils.logging import init_logger, logger
from supar.utils.parallel import init_device


def parse(parser):
    parser.add_argument('--conf', '-c', help='path to config file')
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='max num of threads')
    parser.add_argument('--batch-size', default=5000, type=int, help='batch size')
    parser.add_argument("--local_rank", type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--epochs', default=5000, type=int, help='epochs')
    parser.add_argument("--patience", type=int, default=100, help='Patience for early stopping')
    parser.add_argument("-lr", type=float, default=2e-3, help="Learning rate")
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))
    Parser = args.pop('Parser')

    # This code ensure that all experiment related files are present in a 
    # directory of its own
    # Note: The following part is additional and is not present in original supar
    args.path = Path(args.path)

    if args.path.is_file() or args.path.name.endswith('.model'):
        args.exp_dir = args.path.parent
    else:
        args.exp_dir = args.path

    if args.mode == 'train':
        args.exp_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    init_device(args.device, args.local_rank)
    if args.mode == 'train':
        ts = time.localtime()
        ts = time.strftime("%Y-%m-%d--%H:%M:%S", ts)
        if Parser.NAME == 'multi-biaffine-dependency' and args.train_mode == 'finetune':
            init_logger(logger, args.exp_dir / f"finetune-{ts}.log")
        else:
            init_logger(logger, args.exp_dir / f"{args.mode}-{ts}.log")
    else:
        init_logger(logger, args.exp_dir / f"{args.mode}.log")
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(args.path)
        parser.predict(**args)
