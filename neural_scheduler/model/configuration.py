"""Configuration of the scheduler LM (GLM-4-Voice plus channel and duration settings)."""

from transformers import PretrainedConfig


class ChatGLMConfig(PretrainedConfig):
    """GLM-4-Voice configuration extended for the scheduler.

    Added fields:

    * ``num_channel``: size of the channel-embedding table. 3 for the scheduler:
      0 = script, 1 = dialogue channel A, 2 = dialogue channel B.
    * ``use_duration_prediction``: whether the model carries a duration head.
    * ``num_duration_classes``: number of duration classes (the last one is
      open-ended).

    A stock GLM-4-Voice ``config.json`` loads into this class unchanged. Its
    ``auto_map`` entry, which points at the stock GLM-4-Voice classes, is
    dropped so that checkpoints saved from the scheduler are not silently
    loaded as stock models by the ``Auto*`` classes.
    """

    model_type = "chatglm"

    def __init__(
            self,
            num_layers=28,
            padded_vocab_size=65024,
            hidden_size=4096,
            ffn_hidden_size=13696,
            kv_channels=128,
            num_attention_heads=32,
            seq_length=2048,
            hidden_dropout=0.0,
            classifier_dropout=None,
            attention_dropout=0.0,
            layernorm_epsilon=1e-5,
            rmsnorm=True,
            apply_residual_connection_post_layernorm=False,
            post_layer_norm=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            bias_dropout_fusion=True,
            multi_query_attention=False,
            multi_query_group_num=1,
            rope_ratio=1,
            original_rope=True,
            apply_query_key_layer_scaling=True,
            attention_softmax_in_fp32=False,
            fp32_residual_connection=False,
            num_channel=3,
            use_duration_prediction: bool = False,
            num_duration_classes: int = 32,
            **kwargs
    ):
        kwargs.pop("auto_map", None)
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.rope_ratio = rope_ratio
        self.original_rope = original_rope
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.num_channel = num_channel
        self.use_duration_prediction = use_duration_prediction
        self.num_duration_classes = num_duration_classes
        super().__init__(**kwargs)
