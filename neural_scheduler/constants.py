"""Constants shared by preprocessing, training, the scheduler search and evaluation.

Two id spaces appear throughout the code base and must not be mixed:

* *codec* ids: the 0..16383 codebook indices emitted by the GLM-4-Voice
  WhisperVQ speech tokenizer;
* *vocab* ids: positions in the LLM vocabulary. Codec id ``k`` is the token
  ``<|audio_k|>`` with vocab id ``AUDIO_TOKEN_OFFSET + k``.
"""

# --- Hugging Face model ids -------------------------------------------------
GLM4_VOICE_MODEL_ID = "THUDM/glm-4-voice-9b"
GLM4_VOICE_TOKENIZER_ID = "THUDM/glm-4-voice-tokenizer"

# --- speech tokenizer -------------------------------------------------------
SPEECH_TOKENIZER_SR = 16000
TOKEN_RATE_HZ = 12.5
MS_PER_TOKEN = 80
SAMPLES_PER_TOKEN = 1280  # SPEECH_TOKENIZER_SR / TOKEN_RATE_HZ
NUM_AUDIO_CODES = 16384

# --- vocabulary -------------------------------------------------------------
AUDIO_TOKEN_OFFSET = 152353  # vocab id of <|audio_0|>
SILENCE_CODE = 8520  # codec id the tokenizer emits for silence
SILENCE_TOKEN_ID = AUDIO_TOKEN_OFFSET + SILENCE_CODE  # 160873, vocab id of <|audio_8520|>
BASE_VOCAB_SIZE = 168736  # len(tokenizer) of stock GLM-4-Voice
PADDED_VOCAB_SIZE = 168960  # embedding rows of GLM-4-Voice; new tokens fit without a resize
PAD_TOKEN_ID = 151329
MAX_TURNS = 50  # number of <|start-turn-i|> / <|end-turn-i|> pairs in the vocabulary

# --- training windows -------------------------------------------------------
NUM_CHANNELS = 2
MAX_TOKEN_SIZE = 1024
# One window holds MAX_TOKEN_SIZE / 2 interleaved dialogue tokens: 256 frames per channel, 20.48 s.
WINDOW_FRAMES_PER_CHANNEL = MAX_TOKEN_SIZE // 2 // NUM_CHANNELS
WINDOW_SECONDS = WINDOW_FRAMES_PER_CHANNEL / TOKEN_RATE_HZ

# --- duration head ----------------------------------------------------------
NUM_DURATION_CLASSES = 32  # classes 0..30 are exact, 31 means "31 or more"

# --- channel ids used by the channel embedding ------------------------------
CHANNEL_SCRIPT = 0
CHANNEL_DIALOGUE_A = 1
CHANNEL_DIALOGUE_B = 2
