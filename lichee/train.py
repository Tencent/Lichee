# -*- coding: utf-8 -*-
import argparse
import logging
import os

from lichee import plugin


def parse_args():
    parser = argparse.ArgumentParser(description="training script arguments.")

    # trainer choose
    parser.add_argument("--trainer", type=str,
                        help="choose trainer, you can run --show_trainer list all supported trainer")
    # path of configuration file
    parser.add_argument("--model_config_file", type=str,
                        help="The path of configuration json file.")
    # list all supported module
    parser.add_argument("--show", type=str,
                        help="list all supported module, [trainer|target|model|loss|optim|lr_schedule]")
    # local process rank
    parser.add_argument("--local_rank", default=0, type=int,
                        help="local process rank")

    parsed_args = parser.parse_args()

    # use for DDP mode
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(parsed_args.local_rank)

    if parsed_args.show:
        return parsed_args

    return parsed_args


def run():
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = parse_args()

    if args.show:
        # TODO
        return

    trainer = plugin.get_plugin(plugin.PluginType.TRAINER, args.trainer)
    trainer = trainer(args.model_config_file)
    trainer.train()
