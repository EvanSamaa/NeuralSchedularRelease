"""Inference pipeline: synthesize (step 1), tokenize (step 2), schedule (step 3), reconstruct."""

from .tts import stub_torchvision_meta_registrations_if_requested

# Has to happen before anything imports torch; a no-op unless requested by environment variable.
stub_torchvision_meta_registrations_if_requested()
