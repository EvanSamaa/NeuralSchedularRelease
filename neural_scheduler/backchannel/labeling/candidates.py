"""Rule-based pre-filter for backchannel candidates and the prompt shown to the labeller.

A *sentence* is a dict ``{words: [{word, start, end}], start, end, transcript}``
as found under ``metadata:transcript`` in a Seamless Interaction transcript.
"""

from typing import Dict, List, Tuple

# Seconds trimmed from both ends of a conversation before looking for candidates.
EDGE_MARGIN_SEC = 10.0
# A candidate is shorter than this, or has at most MAX_CANDIDATE_WORDS timestamped words.
MAX_CANDIDATE_DURATION_SEC = 0.8
MAX_CANDIDATE_WORDS = 2
# The other speaker must have spoken at most this long before the candidate.
MAX_GAP_TO_OTHER_SPEAKER_SEC = 2.0
RESPONSE_MARKER = "<|response|>"


def get_word_count(sentence: Dict) -> int:
    """Number of words that carry timestamps."""
    return sum(1 for w in sentence["words"] if w["start"] is not None and w["end"] is not None)


def split_sentence_by_gaps(sentence: Dict, gap_threshold: float = 0.5) -> List[Dict]:
    """Split a sentence wherever two consecutive timestamped words are at least
    ``gap_threshold`` seconds apart."""
    words = sentence["words"]
    valid_word_indices = [
        i for i, w in enumerate(words) if w["start"] is not None and w["end"] is not None
    ]
    if len(valid_word_indices) == 0:
        return [sentence]

    split_indices = []
    for i in range(len(valid_word_indices) - 1):
        curr_idx = valid_word_indices[i]
        next_idx = valid_word_indices[i + 1]
        if words[next_idx]["start"] - words[curr_idx]["end"] >= gap_threshold:
            split_indices.append(curr_idx + 1)

    if not split_indices:
        return [sentence]

    new_sentences = []
    prev_idx = 0
    for split_idx in split_indices + [len(words)]:
        segment_words = words[prev_idx:split_idx]
        segment_valid = [
            w for w in segment_words if w["start"] is not None and w["end"] is not None
        ]
        if segment_valid:
            new_sentences.append({
                "words": segment_words,
                "start": segment_valid[0]["start"],
                "end": segment_valid[-1]["end"],
                "transcript": " ".join(w["word"] for w in segment_words),
            })
        prev_idx = split_idx
    return new_sentences


def preprocess_sentences(sentences: List[Dict], gap_threshold: float = 0.5) -> List[Dict]:
    processed = []
    for sent in sentences:
        processed.extend(split_sentence_by_gaps(sent, gap_threshold))
    return processed


def get_conversation_bounds(sentences_0: List[Dict], sentences_1: List[Dict]) -> Tuple[float, float]:
    """Time range in which candidates are considered: the conversation minus
    ``EDGE_MARGIN_SEC`` at each end. Raises ``ValueError`` on an empty transcript."""
    all_times = []
    for sent in sentences_0 + sentences_1:
        if sent["start"] is not None:
            all_times.append(sent["start"])
        if sent["end"] is not None:
            all_times.append(sent["end"])
    return min(all_times) + EDGE_MARGIN_SEC, max(all_times) - EDGE_MARGIN_SEC


def is_temporally_isolated(sentence: Dict, all_sentences_same_speaker: List[Dict],
                           threshold: float = 1.0) -> bool:
    """True if no other sentence of the same speaker lies within ``threshold``
    seconds before or after ``sentence``."""
    current_start = sentence["start"]
    current_end = sentence["end"]
    for other in all_sentences_same_speaker:
        if (other["start"] == current_start and other["end"] == current_end
                and other["transcript"] == sentence["transcript"]):
            continue
        if other["end"] is not None and other["end"] <= current_start:
            if current_start - other["end"] < threshold:
                return False
        if other["start"] is not None and other["start"] >= current_end:
            if other["start"] - current_end < threshold:
                return False
    return True


def has_other_speaker_context(sentence: Dict, speaker_id: int,
                              sentences_0: List[Dict], sentences_1: List[Dict]) -> bool:
    """True if the sentence that starts immediately before this one belongs to
    the other speaker and overlaps it or ended at most
    ``MAX_GAP_TO_OTHER_SPEAKER_SEC`` earlier."""
    all_sentences = (
        [{"speaker_id": 0, **s} for s in sentences_0]
        + [{"speaker_id": 1, **s} for s in sentences_1]
    )
    all_sentences.sort(key=lambda x: x["start"])

    bc_start = sentence["start"]
    bc_end = sentence["end"]
    candidate_idx = None
    for idx, sent in enumerate(all_sentences):
        if sent["start"] == bc_start and sent["end"] == bc_end and sent["speaker_id"] == speaker_id:
            candidate_idx = idx
            break

    if candidate_idx is None or candidate_idx == 0:
        return False

    prev = all_sentences[candidate_idx - 1]
    if prev["speaker_id"] != speaker_id:
        if prev["end"] > bc_start:
            return True
        if bc_start - prev["end"] <= MAX_GAP_TO_OTHER_SPEAKER_SEC:
            return True
    return False


def identify_backchannel_candidates(sentences_0: List[Dict], sentences_1: List[Dict],
                                    min_time: float, max_time: float,
                                    temporal_threshold: float = 1.0) -> List[Dict]:
    """Sentences that are short, isolated from the same speaker's other speech,
    and directly preceded by the other speaker. Returns
    ``[{speaker_id, sentence}]``."""
    candidates = []
    for speaker_id, sentences in [(0, sentences_0), (1, sentences_1)]:
        for sent in sentences:
            if sent["start"] < min_time or sent["end"] > max_time:
                continue
            duration = sent["end"] - sent["start"]
            if duration < MAX_CANDIDATE_DURATION_SEC or get_word_count(sent) <= MAX_CANDIDATE_WORDS:
                if is_temporally_isolated(sent, sentences, threshold=temporal_threshold):
                    if has_other_speaker_context(sent, speaker_id, sentences_0, sentences_1):
                        candidates.append({"speaker_id": speaker_id, "sentence": sent})
    return candidates


def extract_context(backchannel_sent: Dict, sentences_0: List[Dict], sentences_1: List[Dict],
                    half_window_size: float = 10.0) -> List[Dict]:
    """Sentences of both speakers that start within ``half_window_size`` seconds
    before the candidate's start or after its end, sorted by start time."""
    bc_start = backchannel_sent["start"]
    bc_end = backchannel_sent["end"]
    context = []
    all_sentences = (
        [{"speaker_id": 0, **s} for s in sentences_0]
        + [{"speaker_id": 1, **s} for s in sentences_1]
    )
    for sent in all_sentences:
        if sent["start"] == bc_start and sent["end"] == bc_end:
            continue
        if sent["start"] is not None:
            if 0 <= bc_start - sent["start"] <= half_window_size:
                sent["position"] = "before"
                context.append(sent)
            elif 0 <= sent["start"] - bc_end <= half_window_size:
                sent["position"] = "after"
                context.append(sent)
    context.sort(key=lambda x: x["start"])
    return context


def create_backchannel_example(candidate: Dict, sentences_0: List[Dict], sentences_1: List[Dict],
                               half_window_size: float = 10.0) -> Dict:
    sentence = candidate["sentence"]
    return {
        "backchannel": {
            "speaker_id": candidate["speaker_id"],
            "start": sentence["start"],
            "end": sentence["end"],
            "transcript": sentence["transcript"],
            "words": sentence["words"],
        },
        "context": extract_context(sentence, sentences_0, sentences_1, half_window_size),
    }


def generate_prompt(context_bc_pair: Dict, match_by_position: bool = False) -> Tuple[str, str]:
    """Render the context as ``A:`` / ``B:`` lines with the candidate marked.

    The candidate's speaker is always ``B``. By default every line whose
    transcript equals the candidate's is marked, which is how the released
    dataset was labelled; ``match_by_position=True`` marks only the candidate
    itself. Returns ``(prompt_context, candidate_transcript)``.
    """
    bc = context_bc_pair["backchannel"]
    merged = context_bc_pair["context"] + [bc]
    merged.sort(key=lambda x: x["start"])
    id_to_speaker = {0: "B", 1: "A"} if bc["speaker_id"] == 0 else {0: "A", 1: "B"}
    prompt_context = ""
    for sent in merged:
        sentence = " ".join(w["word"] for w in sent["words"])
        if match_by_position:
            is_candidate = sent is bc
        else:
            is_candidate = sent["transcript"] == bc["transcript"]
        if is_candidate:
            prompt_context += f"{id_to_speaker[sent['speaker_id']]}: {sentence} {RESPONSE_MARKER}\n"
        else:
            prompt_context += f"{id_to_speaker[sent['speaker_id']]}: {sentence}\n"
    return prompt_context, bc["transcript"]
