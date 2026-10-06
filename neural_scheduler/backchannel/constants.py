"""Label schemes, special tokens and audio settings of the backchannel classifier."""

# --- labelling --------------------------------------------------------------
SPEECH_LABEL = "Speech"
BC_LABEL_PREFIX = "backchannel-"
BC_CLASSES = ["Neutral", "Positive", "Emotional", "Anticipation", "Confirmation", "Repetition"]
NO_BACKCHANNEL = "NoBackchannel"

# --- classifier label schemes -----------------------------------------------
# "full": the labeller's six classes plus NoBackchannel.
FULL_LABEL_MAP = {
    "NoBackchannel": 0,
    "Neutral": 1,
    "Positive": 2,
    "Emotional": 3,
    "Anticipation": 4,
    "Confirmation": 5,
    "Repetition": 6,
}
FULL_LABEL_NAMES = list(FULL_LABEL_MAP)
BINARY_LABEL_NAMES = ["NoBackchannel", "Backchannel"]
MERGED_LABEL_NAMES = ["NoBackchannel", "Acknowledgment", "Reactive"]
# "merged": the three classes the pipeline uses. Repetition has no merged class and is dropped.
ORIGINAL_TO_MERGED = {
    "NoBackchannel": 0,
    "Neutral": 1,
    "Positive": 1,
    "Emotional": 2,
    "Anticipation": 2,
    "Confirmation": 2,
    "Repetition": -1,
}
# A merged prediction is written back into the text context with a tag the model saw in training.
MERGED_TO_CANONICAL_BC_TAG = {
    "Acknowledgment": "Neutral",
    "Reactive": "Emotional",
    "NoBackchannel": None,
}

PHASES = ("binary", "full", "merged")


def phase_labels(phase: str):
    """Return ``(num_classes, label_names)`` for a training phase."""
    if phase == "binary":
        return 2, list(BINARY_LABEL_NAMES)
    if phase == "merged":
        return 3, list(MERGED_LABEL_NAMES)
    return 7, list(FULL_LABEL_NAMES)


# --- text format ------------------------------------------------------------
BC_SLOT_TOKEN = "<bc_slot>"
BC_TAG_TOKENS = [f"[BC:{name}]" for name in BC_CLASSES]
LINE_SEP = " </s> "
GPT2_PAD_TOKEN = "<|pad|>"

# --- audio ------------------------------------------------------------------
HUBERT_SR = 16000
MAX_AUDIO_SECONDS_DEFAULT = 15.0
SEAMLESS_SR = 16000
