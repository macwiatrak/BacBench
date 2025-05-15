import json

from transformers import PretrainedConfig


class StripedHyenaConfig(PretrainedConfig):
    model_type = "stripedhyena"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_filters=4096,
        inner_mlp_size=14336,
        attn_layer_idxs=None,
        hyena_layer_idxs=None,
        num_layers=32,
        tie_embeddings=False,
        short_filter_length=3,
        num_attention_heads=32,
        proj_groups=4,
        hyena_filter_groups=1,
        split_k0=True,
        column_split_hyena=True,
        column_split=False,
        model_parallel_size=1,
        pipe_parallel_size=1,
        short_filter_bias=True,
        mha_out_proj_bias=False,
        qkv_proj_bias=False,
        final_norm=True,
        use_cache=True,
        use_flash_attention_2=True,
        use_flash_rmsnorm=True,
        use_flash_depthwise=False,
        use_flashfft=False,
        inference_mode=False,
        prefill_style="fft",
        max_seqlen=32768,
        eps=1e-5,
        state_size=2,
        rotary_emb_base=500000,
        smeared_gqa=False,
        make_vocab_size_divisible_by=8,
        log_intermediate_values=False,
        **kwargs,
    ):
        if hyena_layer_idxs is None:
            hyena_layer_idxs = []
        if attn_layer_idxs is None:
            attn_layer_idxs = []
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.inner_mlp_size = inner_mlp_size
        self.attn_layer_idxs = attn_layer_idxs
        self.hyena_layer_idxs = hyena_layer_idxs
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings
        self.short_filter_length = short_filter_length
        self.num_attention_heads = num_attention_heads
        self.proj_groups = proj_groups
        self.hyena_filter_groups = hyena_filter_groups
        self.split_k0 = split_k0
        self.column_split_hyena = column_split_hyena
        self.column_split = column_split
        self.model_parallel_size = model_parallel_size
        self.pipe_parallel_size = pipe_parallel_size
        self.short_filter_bias = short_filter_bias
        self.mha_out_proj_bias = mha_out_proj_bias
        self.qkv_proj_bias = qkv_proj_bias
        self.final_norm = final_norm
        self.use_cache = use_cache
        self.use_flash_attention_2 = use_flash_attention_2
        self.use_flash_rmsnorm = use_flash_rmsnorm
        self.use_flash_depthwise = use_flash_depthwise
        self.use_flashfft = use_flashfft
        self.inference_mode = inference_mode
        self.prefill_style = prefill_style
        self.max_seqlen = max_seqlen
        self.eps = eps
        self.state_size = state_size
        self.rotary_emb_base = rotary_emb_base
        self.smeared_gqa = smeared_gqa
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.log_intermediate_values = log_intermediate_values
        super().__init__(**kwargs)

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in self.__dict__}

    @classmethod
    def from_original_config(cls, config_path, **kwargs):
        with open(config_path) as f:
            config = json.load(f)

        return cls(**config, **kwargs)
