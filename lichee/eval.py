# -*- coding: utf-8 -*-
import argparse
import logging

from lichee import plugin


def parse_args():
    parser = argparse.ArgumentParser(description="training script arguments.")

    # evaluator choose
    parser.add_argument("--evaluator", type=str,
                        help="choose evaluator, you can run --show_evaluator list all supported evaluator")
    # path of configuration file
    parser.add_argument("--model_config_file", type=str,
                        help="The path of configuration json file.")
    # list all supported module
    parser.add_argument("--show", type=str,
                        help="list all supported module, [trainer|target|model|loss|optim|lr_schedule]")

    parsed_args = parser.parse_args()
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

    evaluator = plugin.get_plugin(plugin.PluginType.EVALUATOR, args.evaluator)
    evaluator = evaluator(args.model_config_file)
    evaluator.eval()
