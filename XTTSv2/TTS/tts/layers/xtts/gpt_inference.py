import math

import torch
from torch import nn
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class GPT2InferenceModel(GPT2PreTrainedModel):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear, kv_cache):
        super().__init__(config)
        self.transformer = gpt
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb

        # !CUSTOM!

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # !FORWARD!
        # print("###### Calling this forward function! ######")

        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1] 
        # print("input ids shape:", input_ids.shape)
        if input_ids.shape[1] != 1: # !IMPORTANT! This is called just for the initial generation step.
            gen_inputs = input_ids[:, prefix_len:]

            gen_emb = self.embeddings(gen_inputs)
            gen_emb = gen_emb + self.pos_embedding(gen_emb)
            
            print(gen_emb.shape)

            # if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
            #     # print("Taking the if branch!")
            #     prefix_emb = self.cached_prefix_emb.repeat_interleave(
            #         gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
            #     )
            # else: 
            # !IMPORTANT! Branch we are taking. The emb shape is [1, N, 1024]
            prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            # else end
            # print('prefix_emb shape:', prefix_emb.shape)
            emb = torch.cat([prefix_emb, gen_emb], dim=1)

            # print("PREFIX EMBS:", prefix_emb)
            # print("GEN EMBS:", gen_emb)

        else: # !IMPORTANT! This is called for the subsequent generation steps, always the with latest input_id. The emb shape is [1, 1, 1024]
            emb = self.embeddings(input_ids)
            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - (prefix_len + 1), attention_mask.device
            )

            # print(input_ids, emb)


        # print("emb shape:", emb.shape)
        # print("past_key_values shape:", past_key_values[0][0].shape if past_key_values is not None else None)

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        # print('hidden_states shape:', transformer_outputs.hidden_states[0][0].shape)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
