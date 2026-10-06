"""Evaluate a trained backchannel classifier.

    python -m neural_scheduler.backchannel.eval \
        --checkpoint runs/joint_merged/best_model \
        --val_file test.jsonl --output_dir runs/joint_merged

Works for every architecture; it is read from the checkpoint's ``config.json``.
Prints the metrics table and writes ``<output_dir>/eval_metrics.json`` (or
``--eval_output``). Accepts the arguments of
:mod:`neural_scheduler.backchannel.train`; the model arguments only matter for
a checkpoint without ``config.json``.
"""

from .train import build_parser
from .training import parse_args_with_config, run_eval_only, set_seed, setup_run


def main(argv=None):
    parser = build_parser()
    parser.description = "Evaluate a backchannel classifier checkpoint"
    parser.set_defaults(eval_only=True)
    args = parse_args_with_config(parser, argv)
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    set_seed(args.seed)
    logger = setup_run(args, "eval.log", "eval_joint")
    run_eval_only(args, logger)


if __name__ == "__main__":
    main()
