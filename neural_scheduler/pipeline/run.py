"""Run the inference pipeline from one YAML config.

    python -m neural_scheduler.pipeline.run --config configs/pipeline/tts_with_backchannel.yaml \\
        [--steps 1 2 3] [--set scheduler.pass1.search_window=15 ...] [--dry_run]

The runner turns the config into the command lines of the three steps

    1  neural_scheduler.pipeline.synthesize      segment cache
    2  neural_scheduler.pipeline.tokenize        scheduler input (tokenize_icc in bc_only mode)
    3  neural_scheduler.pipeline.schedule        schedule + audio

prints them, and runs them one after another as subprocesses. Only keys present
in the config are passed on; everything else keeps the default of the step's
own command line.

Config layout (see ``docs/pipeline.md`` for every key)::

    mode: three_pass | bc_only
    steps: [1, 2, 3]
    variants: [classifier]         # optional; each runs into <io.out_dir>/<variant>
    variant_overrides: {...}       # optional; variant -> {dotted.key: value}
    io: {out_dir: ..., input_jsonl: ... | input_dialogue_json: ... | ...}
    classifier: {...}  tts: {...}  tokenizer: {...}  vad: {...}
    scheduler: {checkpoint: ..., pass1: {...}, pass2: {...}, pass3: {...}}
    environment: {step1: {python: ..., env_setup: ..., env: {...}}, ...}

The variant ``bc_never`` sets ``classifier.confidence_threshold`` to 1.0, so the
classifier runs but never backchannels.
"""

import argparse
import copy
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml

MODES = ("three_pass", "bc_only")
STEP_MODULES = {
    1: "neural_scheduler.pipeline.synthesize",
    2: "neural_scheduler.pipeline.tokenize",
    3: "neural_scheduler.pipeline.schedule",
}
ICC_TOKENIZE_MODULE = "neural_scheduler.pipeline.tokenize_icc"

BUILTIN_VARIANT_OVERRIDES = {"bc_never": {"classifier.confidence_threshold": 1.0}}

# How a config key becomes command-line arguments:
#   value          --flag VALUE
#   list           --flag V1 V2 ...
#   flag           --flag            when true
#   negated_flag   --flag            when false
#   bool_optional  --flag / --no-flag
VALUE, LIST, FLAG, NEGATED_FLAG, BOOL_OPTIONAL = "value", "list", "flag", "negated_flag", "bool_optional"

STEP1_ARGS: Dict[str, Tuple[str, str]] = {
    "io.input_dialogue_json": ("--input_dialogue_json", VALUE),
    "io.input_jsonl": ("--input_jsonl", VALUE),
    "io.bc_vocab_jsonl": ("--bc_vocab_jsonl", VALUE),
    "io.input_bc_gt_jsonl": ("--input_bc_gt_jsonl", VALUE),
    "classifier.type": ("--bc_classifier_type", VALUE),
    "classifier.dir": ("--bc_classifier_dir", VALUE),
    "classifier.tokenizer_dir": ("--bc_tokenizer_dir", VALUE),
    "classifier.confidence_threshold": ("--bc_confidence_threshold", VALUE),
    "classifier.context_window_turns": ("--bc_context_window_turns", VALUE),
    "classifier.context_window_future_turns": ("--bc_context_window_future_turns", VALUE),
    "classifier.bc_context": ("--no_bc_context", NEGATED_FLAG),
    "classifier.suppress_consecutive_bcs": ("--no_suppress_consecutive_bcs", NEGATED_FLAG),
    "classifier.max_audio_seconds": ("--bc_max_audio_seconds", VALUE),
    "classifier.vocab_per_class": ("--bc_vocab_per_class", VALUE),
    "classifier.vocab": ("--bc_vocab", LIST),
    "classifier.word_selection": ("--bc_word_selection", VALUE),
    "classifier.ensure_at_least_one_bc": ("--ensure_at_least_one_bc", BOOL_OPTIONAL),
    "classifier.probability": ("--bc_probability", VALUE),
    "classifier.vocab_subset": ("--bc_vocab_subset", LIST),
    "tts.backend": ("--tts_backend", VALUE),
    "tts.ref_s1": ("--ref_s1", VALUE),
    "tts.ref_s2": ("--ref_s2", VALUE),
    "tts.exaggeration": ("--exaggeration", VALUE),
    "tts.cfg_weight": ("--cfg_weight", VALUE),
    "tts.seed": ("--seed", VALUE),
    "tts.device": ("--device", VALUE),
    "tts.inter_turn_silence_ms": ("--inter_turn_silence_ms", VALUE),
    "tts.intra_turn_pause_ms": ("--intra_turn_pause_ms", VALUE),
    "tts.bc_silence_padding_ms": ("--bc_silence_padding_ms", VALUE),
    "tts.voice_line_padding_ms": ("--voice_line_padding_ms", VALUE),
    "tts.run_naive_only": ("--run_naive_only", FLAG),
    "tts.naive_only": ("--naive_only", FLAG),
    "tts.naive_vad_trim": ("--naive_vad_trim", BOOL_OPTIONAL),
    "io.absolute_paths": ("--absolute_paths", FLAG),
    "io.clean_cache": ("--clean_cache", FLAG),
}

STEP2_ARGS: Dict[str, Tuple[str, str]] = {
    "tokenizer.audio_tokenizer_path": ("--audio_tokenizer_path", VALUE),
    "tokenizer.model_id": ("--model_id", VALUE),
    "tokenizer.device": ("--device", VALUE),
    "tokenizer.max_turns": ("--max_turns", VALUE),
    "vad.threshold": ("--vad_threshold", VALUE),
    "vad.min_speech_ms": ("--vad_min_speech_ms", VALUE),
    "vad.min_silence_ms": ("--vad_min_silence_ms", VALUE),
    "io.absolute_paths": ("--absolute_paths", FLAG),
}

STEP3_ARGS: Dict[str, Tuple[str, str]] = {
    "scheduler.checkpoint": ("--checkpoint", VALUE),
    "scheduler.device": ("--device", VALUE),
    "scheduler.output_sr": ("--output_sr", VALUE),
    "scheduler.output_sub_dir": ("--output_sub_dir", VALUE),
    "scheduler.save_video": ("--save_video", BOOL_OPTIONAL),
    "scheduler.save_input_scripts": ("--save_input_scripts", BOOL_OPTIONAL),
    "scheduler.num_examples": ("--num_examples", VALUE),
    "scheduler.randomize": ("--randomize", FLAG),
    "scheduler.pass1.search_window": ("--pass1_search_window", VALUE),
    "scheduler.pass1.search_stride": ("--pass1_search_stride", VALUE),
    "scheduler.pass1.scoring_context": ("--pass1_scoring_context", VALUE),
    "scheduler.pass1.inter_turn_gap": ("--pass1_inter_turn_gap", VALUE),
    "scheduler.pass1.schedule_subturns": ("--pass_1_schedule_subturns", FLAG),
    "scheduler.pass1.speaker_1_subturn_min_pause": ("--pass_1_speaker_1_subturn_min_pause", VALUE),
    "scheduler.pass1.speaker_1_subturn_max_pause": ("--pass_1_speaker_1_subturn_max_pause", VALUE),
    "scheduler.pass1.speaker_2_subturn_min_pause": ("--pass_1_speaker_2_subturn_min_pause", VALUE),
    "scheduler.pass1.speaker_2_subturn_max_pause": ("--pass_1_speaker_2_subturn_max_pause", VALUE),
    "scheduler.pass1.subturn_search_stride": ("--pass_1_subturn_search_stride", VALUE),
    "scheduler.pass1.use_speaker_gap_constraints": ("--pass_1_use_speaker_gap_constraints", FLAG),
    "scheduler.pass1.speaker_1_min_gap": ("--pass_1_speaker_1_min_gap", VALUE),
    "scheduler.pass1.speaker_1_max_gap": ("--pass_1_speaker_1_max_gap", VALUE),
    "scheduler.pass1.speaker_2_min_gap": ("--pass_1_speaker_2_min_gap", VALUE),
    "scheduler.pass1.speaker_2_max_gap": ("--pass_1_speaker_2_max_gap", VALUE),
    "scheduler.pass1.use_strong_overlap_constraints": ("--pass_1_use_strong_overlap_constraints", FLAG),
    "scheduler.pass1.strong_overlap_min": ("--pass_1_strong_overlap_min", VALUE),
    "scheduler.pass1.strong_overlap_max": ("--pass_1_strong_overlap_max", VALUE),
    "scheduler.pass2.search_stride": ("--pass2_search_stride", VALUE),
    "scheduler.pass2.scoring_context": ("--pass2_scoring_context", VALUE),
    "scheduler.pass2.bc_position_limit": ("--pass2_bc_position_limit", VALUE),
    "scheduler.pass2.bc_effect": ("--pass2_bc_effect", VALUE),
    "scheduler.pass2.max_bc_overlap_ratio": ("--pass2_max_bc_overlap_ratio", VALUE),
    "scheduler.pass2.score_only_pauses": ("--pass2_score_only_pauses", FLAG),
    "scheduler.pass3.scoring_context": ("--pass3_scoring_context", VALUE),
    "scheduler.pass3.max_extra_gap": ("--pass3_max_extra_gap", VALUE),
}

# Keys handled by the runner itself rather than passed to a step.
RUNNER_KEYS = {
    "mode",
    "steps",
    "variants",
    "variant_overrides",
    "io.out_dir",
    "io.icc_base_dir",
    "io.tokenized_jsonl",
    "io.output_root",
    "environment",
}
INPUT_KEYS = ("io.input_dialogue_json", "io.input_jsonl", "io.bc_vocab_jsonl", "io.input_bc_gt_jsonl")


class ConfigError(ValueError):
    """The pipeline config is malformed."""


# --------------------------------------------------------------------------- #
# Config handling
# --------------------------------------------------------------------------- #


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: the config must be a mapping")
    return config


def get_key(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_key(config: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(config: Dict[str, Any], assignments: List[str]) -> None:
    """Apply ``key.path=value`` strings; values are parsed as YAML."""
    for assignment in assignments:
        if "=" not in assignment:
            raise ConfigError(f"--set expects key=value, got '{assignment}'")
        key, raw = assignment.split("=", 1)
        set_key(config, key.strip(), yaml.safe_load(raw))


def flatten(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Leaf values keyed by dotted path. ``environment`` and ``variant_overrides`` are kept whole."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and dotted not in ("environment", "variant_overrides"):
            flat.update(flatten(value, dotted + "."))
        else:
            flat[dotted] = value
    return flat


def validate(config: Dict[str, Any]) -> None:
    """Reject unknown keys, a bad mode, missing output directory or ambiguous input."""
    known = set(STEP1_ARGS) | set(STEP2_ARGS) | set(STEP3_ARGS) | RUNNER_KEYS
    unknown = sorted(k for k in flatten(config) if k not in known)
    if unknown:
        raise ConfigError(f"Unknown config key(s): {', '.join(unknown)}")
    mode = config.get("mode", "three_pass")
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got '{mode}'")
    if not get_key(config, "io.out_dir"):
        raise ConfigError("io.out_dir is required")
    steps = config.get("steps", [1, 2, 3])
    if not steps or any(s not in STEP_MODULES for s in steps):
        raise ConfigError(f"steps must be a non-empty subset of {sorted(STEP_MODULES)}, got {steps}")
    if 1 in steps:
        given = [k for k in INPUT_KEYS if get_key(config, k) is not None]
        if len(given) != 1:
            raise ConfigError(f"exactly one of {', '.join(INPUT_KEYS)} is required for step 1, got {given or 'none'}")
    if mode == "bc_only" and 2 in steps and not get_key(config, "io.icc_base_dir"):
        raise ConfigError("io.icc_base_dir is required for step 2 in bc_only mode")
    if 3 in steps and not get_key(config, "scheduler.checkpoint"):
        raise ConfigError("scheduler.checkpoint is required for step 3")


def config_for_variant(config: Dict[str, Any], variant: Optional[str]) -> Dict[str, Any]:
    """The config with the variant's overrides applied."""
    resolved = copy.deepcopy(config)
    if variant is None:
        return resolved
    overrides = dict(BUILTIN_VARIANT_OVERRIDES.get(variant, {}))
    overrides.update((config.get("variant_overrides") or {}).get(variant, {}) or {})
    for key, value in overrides.items():
        set_key(resolved, key, value)
    return resolved


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #


def _emit(config: Dict[str, Any], table: Dict[str, Tuple[str, str]]) -> List[str]:
    args: List[str] = []
    for key, (flag, kind) in table.items():
        value = get_key(config, key)
        if value is None:
            continue
        if kind == VALUE:
            args += [flag, str(value)]
        elif kind == LIST:
            values = value if isinstance(value, (list, tuple)) else [value]
            args += [flag] + [str(v) for v in values]
        elif kind == FLAG:
            if value:
                args.append(flag)
        elif kind == NEGATED_FLAG:
            if not value:
                args.append(flag)
        elif kind == BOOL_OPTIONAL:
            args.append(flag if value else "--no-" + flag[2:])
    return args


def variant_out_dir(config: Dict[str, Any], variant: Optional[str]) -> str:
    out_dir = str(get_key(config, "io.out_dir"))
    return os.path.join(out_dir, variant) if variant else out_dir


def _layout(config: Dict[str, Any], variant: Optional[str]) -> Dict[str, str]:
    """Where one variant's cache, tokenised JSONL and scheduler output live.

    ``io.tokenized_jsonl`` and ``io.output_root`` may use the ``{out_dir}``
    placeholder for the variant's output directory.
    """
    out_dir = variant_out_dir(config, variant)
    cache_dir = os.path.join(out_dir, "segment_cache")
    tokenized = get_key(config, "io.tokenized_jsonl") or os.path.join(cache_dir, "tokenized_result.jsonl")
    output_root = get_key(config, "io.output_root") or out_dir
    return {
        "out_dir": out_dir,
        "cache_dir": cache_dir,
        "tokenized": str(tokenized).replace("{out_dir}", out_dir),
        "output_root": str(output_root).replace("{out_dir}", out_dir),
    }


def step_arguments(config: Dict[str, Any], step: int, variant: Optional[str] = None) -> Tuple[str, List[str]]:
    """Module and arguments of one step for one variant (``config`` already variant-resolved)."""
    paths = _layout(config, variant)
    bc_only = config.get("mode", "three_pass") == "bc_only"

    if step == 1:
        return STEP_MODULES[1], ["--out_dir", paths["out_dir"]] + _emit(config, STEP1_ARGS)
    if step == 2:
        if bc_only:
            args = ["--segment_cache_dir", paths["cache_dir"], "--icc_base_dir", str(get_key(config, "io.icc_base_dir"))]
            return ICC_TOKENIZE_MODULE, args + ["--output_path", paths["tokenized"]] + _emit(config, STEP2_ARGS)
        args = ["--cache_base_dir", paths["cache_dir"], "--output_path", paths["tokenized"]]
        return STEP_MODULES[2], args + _emit(config, STEP2_ARGS)
    if step == 3:
        args = ["--tokenized_jsonl", paths["tokenized"], "--cache_base_dir", paths["cache_dir"]]
        args += ["--output_root", paths["output_root"], "--mode", "bc_only" if bc_only else "three_pass"]
        return STEP_MODULES[3], args + _emit(config, STEP3_ARGS)
    raise ConfigError(f"Unknown step {step}")


def step_command(config: Dict[str, Any], step: int, variant: Optional[str] = None) -> Tuple[List[str], Dict[str, str]]:
    """The command to execute for a step and the extra environment variables it needs.

    With ``environment.step<N>.env_setup`` the command is wrapped in
    ``bash -c "<env_setup> && exec python -m ..."`` so the step runs in the
    environment that snippet activates.
    """
    module, args = step_arguments(config, step, variant)
    environment = get_key(config, f"environment.step{step}", {}) or {}
    python = environment.get("python")
    env_setup = environment.get("env_setup")
    extra_env = {str(k): str(v) for k, v in (environment.get("env") or {}).items()}

    if env_setup:
        inner = [python or "python", "-m", module] + args
        return ["bash", "-c", f"{env_setup} && exec {shlex.join(inner)}"], extra_env
    return [python or sys.executable, "-m", module] + args, extra_env


def plan(config: Dict[str, Any], steps: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Every command of the run, in execution order: all steps of a variant, then the next variant."""
    validate(config)
    steps = sorted(steps or config.get("steps", [1, 2, 3]))
    variants = config.get("variants") or [None]
    commands = []
    for variant in variants:
        resolved = config_for_variant(config, variant)
        for step in steps:
            command, env = step_command(resolved, step, variant)
            commands.append({"variant": variant, "step": step, "command": command, "env": env})
    return commands


def run(config: Dict[str, Any], steps: Optional[List[int]] = None, dry_run: bool = False) -> int:
    """Print and execute the plan. Returns the first non-zero exit status, or 0."""
    for item in plan(config, steps):
        label = f"step {item['step']}" + (f" [{item['variant']}]" if item["variant"] else "")
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in item["env"].items())
        print(f"\n=== {label} ===", flush=True)
        print((prefix + " " if prefix else "") + shlex.join(item["command"]), flush=True)
        if dry_run:
            continue
        env = dict(os.environ)
        env.update(item["env"])
        status = subprocess.run(item["command"], env=env).returncode
        if status != 0:
            print(f"{label} failed with exit status {status}", file=sys.stderr)
            return status
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the inference pipeline from a YAML config.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--steps", type=int, nargs="+", choices=sorted(STEP_MODULES), default=None, help="Steps to run (default: the config's)."
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Override config keys, e.g. --set tts.seed=46 scheduler.pass1.search_window=15.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the commands without running them.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    apply_overrides(config, [item for group in args.overrides for item in group])
    try:
        return run(config, steps=args.steps, dry_run=args.dry_run)
    except ConfigError as err:
        print(f"Config error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
