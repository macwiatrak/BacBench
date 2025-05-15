"""StripedHyena custom code port for the Hugging Face Hub"""

import torch
from configuration_hyena import StripedHyenaConfig
from model import StripedHyena
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput
from transformers.utils import logging
from utils import dotdict

logger = logging.get_logger(__name__)


class StripedHyenaPreTrainedModel(PreTrainedModel):
    config_class = StripedHyenaConfig
    base_model_prefix = "sh"
    supports_gradient_checkpointing = False
    _no_split_modules = ["AttentionBlock", "ParallelGatedConvBlock"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = [r"freq"]
    _keys_to_ignore_on_load_unexpected = [r"fftconv", r"twiddle_factors"]
    _supports_flash_attn_2 = True


class StripedHyenaModelForCausalLM(StripedHyenaPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyena(model_config)
        self.backbone.gradient_checkpointing = False
        self.config = config
        vocab_size = config.vocab_size
        if vocab_size % config.make_vocab_size_divisible_by != 0:
            vocab_size += config.make_vocab_size_divisible_by - (vocab_size % config.make_vocab_size_divisible_by)
        self.vocab_size = vocab_size
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues()

    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        past_key_values=None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            if self.backbone.gradient_checkpointing and self.backbone.training:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            elif labels is not None:
                logger.warning_once(
                    "`use_cache=True` is incompatible with loss calculation. Setting `use_cache=False`..."
                )
                use_cache = False

        inputs = input_ids
        if use_cache:
            if past_key_values is None:
                past_key_values = self.backbone.initialize_inference_params()

                batch_size = input_ids.shape[0]
                past_key_values["mha"].max_batch_size = batch_size
                past_key_values["hyena"].max_batch_size = batch_size
            else:
                seqlen_offset = past_key_values["mha"].seqlen_offset
                if seqlen_offset == 0:
                    # second loop through generate will have prompt_len + 1 as seqlen
                    seqlen_offset = input_ids.shape[-1] - 1
                    past_key_values["hyena"].seqlen_offset = seqlen_offset
                    past_key_values["mha"].seqlen_offset = seqlen_offset
                else:
                    past_key_values["mha"].seqlen_offset += 1
                    past_key_values["hyena"].seqlen_offset += 1

                inputs = input_ids[
                    :,
                    -1:,
                ]

        logits, past_key_values = self.backbone(
            inputs,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

        if return_dict:
            return CausalLMOutputWithPast(
                logits=logits,
                hidden_states=None,
                past_key_values=past_key_values if use_cache else None,
                loss=loss,
            )
        else:
            return logits

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }


class EvoForSeqEmb(StripedHyenaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyena(model_config)
        self.backbone.gradient_checkpointing = False
        self.config = config
        vocab_size = config.vocab_size
        if vocab_size % config.make_vocab_size_divisible_by != 0:
            vocab_size += config.make_vocab_size_divisible_by - (vocab_size % config.make_vocab_size_divisible_by)

        # self.vocab_size = vocab_size
        # self.num_labels = config.num_labels
        # self.hidden = torch.nn.Linear(
        #     config.hidden_size, config.hidden_size * 2, dtype=torch.float32
        # )  # .to(torch.bfloat16)
        # self.classifier = torch.nn.Linear(
        #     config.hidden_size * 2, self.num_labels, dtype=torch.float32
        # )  # .to(torch.bfloat16)#load as bf16
        # self.ln_hidden = torch.nn.LayerNorm(config.hidden_size * 2, dtype=torch.float32)
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues()

    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        past_key_values=None,
        return_dict: bool | None = None,
        eos_index: bool | None = None,
    ) -> tuple | SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # eos_index = (
        #     eos_index
        #     if eos_index is not None
        #     else torch.ones(input_ids.shape[0], 1, dtype=int) * input_ids.shape[1] - 1
        # )

        logits, past_key_values = self.backbone(
            input_ids,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )
        return logits
        # feature=logits[:,-1,:] #use [EOS] Instead [CLS]
        # eos_index=eos_index.to(logits.device)#dynamic-adaption [eos] position for each sequence.
        # logits = logits.to(dtype=self.hidden.weight.dtype).gather(1, eos_index.unsqueeze(-1).expand(-1, -1, logits.size(-1)))
        #
        # # feature.to(self.hidden.weight.dtype)
        # logits = self.classifier(self.ln_hidden(torch.tanh(self.hidden(logits))))
        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()#ignoring label:-100
        #
        #     labels = labels.to(logits.device)
        #     loss = loss_fct(logits.view(-1,self.num_labels), labels)
        #
        # if return_dict:
        #     return SequenceClassifierOutput(
        #         loss = loss,
        #         logits = logits,
        #         hidden_states = None,
        #         attentions = None
        #     )
        # else:
        #     return logits

    @classmethod
    def can_generate(cls) -> bool:
        return False
