from __future__ import annotations

import logging
import re
from typing import Literal

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from bacbench.modeling.utils.utils import average_unpadded

try:
    from faesm.esm import FAEsmForMaskedLM
    from faesm.esmc import ESMC

    faesm_installed = True
except ImportError:
    faesm_installed = False
    logging.warning(
        "faESM (fast ESM) not installed, this will lead to significant slowdown. "
        "Defaulting to use HuggingFace implementation. "
        "Please consider installing faESM: https://github.com/pengzhangzhi/faplm"
    )


# -------------------------------------------------------
# Base class
# -------------------------------------------------------
class SeqEmbedder(nn.Module):
    """
    Parent class for every sequence language‑model embedder. Currently works for a range of pLMs and DNA LMs.

    Sub‑classes must implement:
        * self._load(model_name_or_path)            (create tokenizer & model)
        * self.forward(inputs, pooling)      (returns (B,D) tensor)
    """

    tokenizer: object  # filled by _load()
    model: nn.Module  # filled by _load()
    device: torch.device
    model_type: str | None = None  # filled by _load()

    def __init__(
        self,
        model_name_or_path: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        compile_model: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.model_type = None  # type: str | None
        # this must load the model and tokenizer
        self._load(model_name_or_path)  # implemented by child
        if compile_model:  # optional torch.compile
            self.model = torch.compile(self.model)
        self.model.to(self.device, dtype=self.dtype).eval()

    # ---------- mandatory interface for child classes -------------------
    def _load(self, model_name_or_path: str):  # pragma: no cover
        raise NotImplementedError

    def _forward_batch(
        self,
        inputs: dict[str, torch.Tensor],
        pooling: Literal["cls", "mean"] = None,
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    # ---------- optional: sequence pre‑processing -----------------------
    def _preprocess_seqs(self, seqs: list[str]) -> list[str]:
        """Override if the LM needs special preprocessing (for example ProtBERT)."""
        return seqs

    def _tokenize(self, seqs: list[str], max_seq_len: int) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            seqs,
            add_special_tokens=True,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    # ---------- public method -------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        sequences: list[str],
        max_seq_len: int = 1024,
        pooling: Literal["cls", "mean"] = "mean",
        return_numpy: bool = True,
        gene_mask: list[np.array] = None,  # for EvoEmbedder
    ) -> list[np.ndarray]:
        """
        Return a list of numpy embeddings (one per input sequence).

        :param sequences: list of protein or DNA sequences to embed
        :param max_seq_len: maximum sequence length for the model
        :param pooling: pooling method to use for the embeddings
            * "cls"  – return representation at token 0
            * "mean" – mean of un‑padded token embeddings
        :param return_numpy: if True, return numpy arrays, otherwise return torch tensors
        :param gene_mask: optional mask for EvoEmbedder to take the representations of tokens
            corresponding to genes only
        """
        assert pooling in {"cls", "mean"}

        seqs = self._preprocess_seqs(sequences)

        inputs = self._tokenize(seqs, max_seq_len=max_seq_len)
        if self.model_type in ["evo", "glm2"] and gene_mask is not None:
            rep = self._forward_batch(inputs, pooling, gene_mask=gene_mask)
        else:
            rep = self._forward_batch(inputs, pooling)  # (B,D)
        if not return_numpy:
            return rep
        return list(rep.cpu().type(torch.float32).numpy())


class ESM2Embedder(SeqEmbedder):
    """Embedder for ESM-2 models from Meta."""

    def _load(self, model_name_or_path: str):
        if faesm_installed:
            self.model = FAEsmForMaskedLM.from_pretrained(model_name_or_path)
            self.tokenizer = self.model.tokenizer
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_type = "esm2"

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(**inputs)["last_hidden_state"]  # (B,N,D)
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        # mean over valid tokens
        mask = inputs["attention_mask"].type_as(last_hidden_state)
        return torch.einsum("b n d, b n -> b d", last_hidden_state, mask) / mask.sum(1, keepdim=True)


class ESMCEmbedder(SeqEmbedder):
    """Embedder for ESMC models from EvolutionaryScale."""

    def _load(self, model_name_or_path: str):
        self.model = ESMC.from_pretrained(model_name_or_path, use_flash_attn=True)
        self.tokenizer = self.model.tokenizer
        self.model_type = "esmc"

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(inputs["input_ids"]).embeddings
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        # mean over valid tokens
        protein_representations = average_unpadded(last_hidden_state, inputs["attention_mask"])
        return protein_representations


class ESMPlusPlusEmbedder(SeqEmbedder):
    """Embedder for ESMPlusPlus models from Synthyra."""

    def _load(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = self.model.tokenizer
        self.model_type = "esmplusplus"

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(**inputs).last_hidden_state
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        # mean over valid tokens
        protein_representations = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        return protein_representations


class ProtBERTEmbedder(SeqEmbedder):
    """Embedder for ProtBERT models from HuggingFace."""

    def _load(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
        self.model_type = "protbert"

    def _preprocess_seqs(self, seqs: list[str]) -> list[str]:
        """Override if the LM needs special preprocessing (for example ProtBERT)."""
        seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs]
        return seqs

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        protein_representations = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        return protein_representations


class ProGen2Embedder(SeqEmbedder):
    """Embedder for ProtBERT models from HuggingFace."""

    def _load(self, model_name_or_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        # set padding token
        self.tokenizer.pad_token_id = 0
        self.model_type = "progen2"

    def _preprocess_seqs(self, seqs: list[str]) -> list[str]:
        """Override if the LM needs special preprocessing (for example ProtBERT)."""
        seqs = ["1" + sequence for sequence in seqs]
        return seqs

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(inputs["input_ids"], output_hidden_states=True).hidden_states[-1]
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        protein_representations = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        return protein_representations


class NucleotideTransformerEmbedder(SeqEmbedder):
    """Embedder for Nucleotide Transformer models from HuggingFace."""

    def _load(self, model_name_or_path: str):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model_type = "nucleotide_transformer"

    def _tokenize(self, seqs: list[str], max_seq_len: int) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer.batch_encode_plus(
            seqs, return_tensors="pt", padding="longest", truncation=True, max_length=max_seq_len
        )
        # move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            encoder_attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )["hidden_states"][-1]
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        dna_representations = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        return dna_representations


class DNABERT2Embedder(SeqEmbedder):
    """Embedder for DNABERT-2 models from HuggingFace."""

    def _load(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model_type = "dnabert2"

    def _tokenize(self, seqs: list[str], max_seq_len: int) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer.batch_encode_plus(
            seqs, return_tensors="pt", padding="longest", truncation=True, max_length=max_seq_len
        )
        # move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        dna_representations = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        return dna_representations


class MistralDNAEmbedder(SeqEmbedder):
    """Embedder for Mistral-DNA models from HuggingFace."""

    def _load(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model_type = "mistral_dna"

    def _tokenize(self, seqs: list[str], max_seq_len: int) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer.batch_encode_plus(
            seqs, return_tensors="pt", padding="longest", truncation=True, max_length=max_seq_len
        )
        # move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        dna_representations = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        return dna_representations


class ProkBERTEmbedder(SeqEmbedder):
    """Embedder for ProkBERT models from HuggingFace."""

    def _load(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model_type = "prokbert"

    def _tokenize(self, seqs: list[str], max_seq_len: int) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer.batch_encode_plus(
            seqs, return_tensors="pt", padding="longest", truncation=True, max_length=max_seq_len
        )
        # move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _forward_batch(self, inputs, pooling: Literal["cls", "mean"] = "mean") -> torch.Tensor:
        last_hidden_state = self.model(**inputs).last_hidden_state
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        dna_representations = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        return dna_representations


class gLM2Embedder(SeqEmbedder):
    """Embedder for gLM2 models from HuggingFace."""

    def _load(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model_type = "glm2"

    def _tokenize(self, seqs: list[str], max_seq_len: int) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer.batch_encode_plus(
            seqs, return_tensors="pt", padding="longest", truncation=True, max_length=max_seq_len
        )
        # move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _forward_batch(
        self, inputs, pooling: Literal["cls", "mean"] = "mean", gene_mask: list[np.array] = None
    ) -> torch.Tensor:
        last_hidden_state = self.model(inputs["input_ids"], output_hidden_states=True).last_hidden_state
        if gene_mask is not None:
            # apply gene mask to the last hidden state
            gene_mask = torch.from_numpy(np.stack(gene_mask, axis=0)).to(
                device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )
            # multiply last_hidden_state with gene_mask
            last_hidden_state = last_hidden_state * gene_mask.unsqueeze(-1)
            pooling = "mean"  # force mean pooling if gene_mask is provided
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        seq_representations = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        return seq_representations


class EvoEmbedder(SeqEmbedder):
    """Embedder for Evo models from HuggingFace."""

    def _load(
        self,
        model_name_or_path: str,
        cache_dir: str | None = None,
        revision: str = "1.1_fix",
    ):
        # load the config
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True, revision=revision, cache_dir=cache_dir
        )
        config.use_cache = False
        config.inference_mode = True
        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )

        # support for getting the final layer embedding
        class CustomEmbedding(nn.Module):
            def unembed(self, u):
                return u

        self.model.backbone.unembed = CustomEmbedding()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, cls_token="@", eos_token="&", bos_token="^", pad_token="N"
        )
        self.model_type = "evo"

    def _tokenize(self, seqs: list[str], max_seq_len: int) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer.batch_encode_plus(
            seqs,
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        return inputs

    def _forward_batch(
        self, inputs, pooling: Literal["mean", "eos"] = "mean", gene_mask: list[np.array] = None
    ) -> torch.Tensor:
        # to prevent CUDA OOM, we only move the input_ids to the device and move them to CPU immediately
        # after the forward pass
        last_hidden_state = self.model(inputs["input_ids"].to(self.device)).logits.cpu()  # (batch, length, embed dim)
        if gene_mask is not None:
            # apply gene mask to the last hidden state
            gene_mask = torch.from_numpy(np.stack(gene_mask, axis=0)).to(
                device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )
            # multiply last_hidden_state with gene_mask
            last_hidden_state = last_hidden_state * gene_mask.unsqueeze(-1)
            pooling = "mean"  # force mean pooling if gene_mask is provided
        if pooling == "cls":
            return last_hidden_state[:, 0]  # (B,D)
        if pooling == "eos":
            eos_token_id = self.tokenizer.eos_token_id
            eos_mask = inputs["input_ids"] == eos_token_id
            seq_representations = last_hidden_state[eos_mask].view(-1, last_hidden_state.size(-1))
        elif pooling == "mean":
            seq_representations = torch.einsum(
                "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
            ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        else:
            raise ValueError(f"Use 'mean' or 'eos' pooling for Evo, got {pooling}.")
        return seq_representations


def load_seq_embedder(model_name_or_path: str, device: str = None):
    """Helper function to load a sequence embedder object based on model name or path

    :param model_name_or_path: path to a model on HuggingFace
    :param device: device to load the model on
    :return: SeqEmbedder object for the specific model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # protein LMs
    if "facebook/esm2" in model_name_or_path:
        dtype = torch.float16 if faesm_installed else torch.bfloat16
        return ESM2Embedder(model_name_or_path, dtype=dtype, device=device)

    if "esmc" in model_name_or_path:
        return ESMCEmbedder(model_name_or_path, dtype=torch.float16, device=device)

    if "esmplusplus" in model_name_or_path.lower():
        return ESMPlusPlusEmbedder(model_name_or_path, dtype=torch.bfloat16, device=device)

    if "prot_bert" in model_name_or_path:
        return ProtBERTEmbedder(model_name_or_path, dtype=torch.float16, device=device)

    if "progen2" in model_name_or_path:
        return ProGen2Embedder(model_name_or_path, dtype=torch.float32, device=device)

    # DNA LMs
    if "nucleotide-transformer" in model_name_or_path:
        return NucleotideTransformerEmbedder(model_name_or_path, dtype=torch.float16, device=device)

    if "DNABERT-2" in model_name_or_path:
        return DNABERT2Embedder(model_name_or_path, dtype=torch.float32, device=device)

    if "Mistral-DNA" in model_name_or_path:
        return MistralDNAEmbedder(model_name_or_path, dtype=torch.float32, device=device)

    if "prokbert" in model_name_or_path:
        return ProkBERTEmbedder(model_name_or_path, dtype=torch.float32, device=device)

    # mixed modality LMs
    if "gLM2" in model_name_or_path:
        return gLM2Embedder(model_name_or_path, dtype=torch.bfloat16, device=device)

    if "evo" in model_name_or_path:
        return EvoEmbedder(model_name_or_path, device=device, dtype=torch.bfloat16)

    raise ValueError(
        f"Unknown model name or path: {model_name_or_path},"
        f" supported models are: ESM-2, ESMC, ProtBert, "
        "Nucleotide Transformer, Mistral-DNA, DNABERT-2 "
        f" supported models are: ESM-2, ESMC, ESMPlusPlus, ProtBert, ProGen "
        "Nucleotide Transformer, Mistral-DNA, DNABERT-2, ProkBERT, gLM2 and Evo models "
        f"available at HuggingFace."
    )
