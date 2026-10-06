"""Data preprocessing for the scheduler: raw two-channel conversations to training JSONL.

Submodules are imported on demand; importing this package pulls in nothing heavy.

* ``speech_tokenizer`` - GLM-4-Voice WhisperVQ loading and batched tokenisation
* ``vad``              - Silero VAD loading and timestamp conversion
* ``turns``            - turn-ID assignment from VAD segments
* ``durations``        - per-channel remaining-duration labels
* ``sequence``         - window -> record (script regions, dialogue, labels)
* ``sources``          - Fisher and Seamless Interaction conversation readers
* ``sharding``         - shard indices, logging, resume bookkeeping
* ``preprocess``, ``merge``, ``inspect`` - command line entry points
"""
