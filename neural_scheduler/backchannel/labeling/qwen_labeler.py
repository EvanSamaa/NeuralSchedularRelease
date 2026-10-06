"""Text-only batch labelling with Qwen3-Omni (4-bit)."""

import time
from typing import List, Optional

import torch

USER_PROMPT_TEMPLATE = "Context Phrase: {context}\nClassification target phrase: {target}"
MAX_NEW_TOKENS = 64


class QwenOmniBatchLabeler:
    """Runs the Qwen3-Omni "thinker" on (context, target) text pairs.

    The model is loaded 4-bit quantised (NF4, float16 compute, double
    quantisation) with flash attention and decoded greedily, which needs a
    CUDA GPU and a ``transformers`` version that ships the Qwen3-Omni classes.
    """

    def __init__(self, model_path: str, system_prompt: Optional[str] = None):
        from transformers import (
            BitsAndBytesConfig,
            Qwen3OmniMoeProcessor,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        self.model_path = model_path
        self.system_prompt = system_prompt

        print(f"Initializing 4-bit model on {torch.cuda.get_device_name(0)}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        # Batched generation needs left padding.
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        print("Model loaded. Ready for batch inference.")

    def build_conversation(self, pair) -> list:
        """Chat messages for one ``[{"transcript": context}, {"transcript": target}]`` pair."""
        conversation = []
        if self.system_prompt:
            conversation.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        user_text = USER_PROMPT_TEMPLATE.format(
            context=pair[0]["transcript"], target=pair[1]["transcript"]
        )
        conversation.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        return conversation

    def run_batch_labeling(self, dataset_pairs: list, batch_size: int = 16) -> List[str]:
        """Return the raw generated text for every pair, in order."""
        all_results = []
        total_items = len(dataset_pairs)
        print(f"Starting batch labeling for {total_items} items with Batch Size {batch_size}...")

        for i in range(0, total_items, batch_size):
            batch_start = time.time()
            batch_items = dataset_pairs[i:i + batch_size]
            batch_texts = [
                self.processor.apply_chat_template(
                    self.build_conversation(pair), add_generation_prompt=True, tokenize=False
                )
                for pair in batch_items
            ]
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(
                self.model.device
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    do_sample=False,
                )
            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            decoded_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            all_results.extend(decoded_texts)

            batch_time = time.time() - batch_start
            print(
                f"Batch {i // batch_size + 1}: Processed {len(batch_items)} items in "
                f"{batch_time:.2f}s ({batch_time / len(batch_items):.2f}s per item)"
            )
            if i == 0:
                print(f"Sample Output: {decoded_texts[0]}")

        return all_results
