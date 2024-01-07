# Source: https://github.com/huggingface/trl/blob/6614b8aa6bd752bf19963a5b308f44e079e8e9fe/trl/models/modeling_value_head.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from trl.models.modeling_base import PreTrainedModelWrapper
from trl.models.modeling_value_head import ValueHead


class NNHead(nn.Module):
    r"""
    Replace a single linear layer with an NN head for better expressivity
    """

    def __init__(self, hidden_size, output_size, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.linear_layers = [self.linear1, self.linear2, self.linear3]
        self.layers = [self.linear1, self.relu1, self.linear2, self.relu2,
                       self.linear3]

    def forward(self, hidden_states):
        x = hidden_states
        for layer in self.layers:
            x = layer(x)
        return x




class CustomAutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    r"""
    An autoregressive model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, `push_to_hub` and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the `ValueHead` class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                `ValueHead` class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                `ValueHead` if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                `ValueHead`. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the `ValueHead` with a random distribution. This is the default
                    strategy.
                - **"normal"** -- Initializes the weights of the `ValueHead` with a normal distribution.

    """
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, conditional_twist=False, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        self.conditional_twist = conditional_twist

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)


        for x in self.pretrained_model.named_parameters():
            if 'wte' in x[0]:
                d_model = x[1].shape[-1]
                output_size = x[1].shape[0]
                break

        if conditional_twist:
            hidden_size = d_model * 2
            self.v_head.summary = nn.Linear(hidden_size, 1)
        else:
            hidden_size = d_model
        # Add a custom NN after the model embedding (increases capacity)
        self.nn_head = NNHead(hidden_size, output_size)

        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head (and custom extra layers). The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

        for linear_layer in self.nn_head.linear_layers:
            in_plus_out_for_sd = linear_layer.in_features + linear_layer.out_features
            linear_layer.weight.data.normal_(mean=0.0, std=(2. / (in_plus_out_for_sd)) ** 0.5)
            linear_layer.bias.data.zero_()

    def get_lm_logits(self, base_model_output, last_hidden_state):
        return base_model_output.logits + self.nn_head(last_hidden_state) # Do this so that the text probs are close to the initial model at the beginning


    def remove_requires_grad_base_model(self):
        for x in filter(lambda p: p[1].requires_grad, self.named_parameters()):
            name = x[0]
            if 'nn_head' not in name:
                x[1].requires_grad = False

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        do_value_calc=True,
        condition_twist_on_tokens=None,
        **kwargs,
    ):
        r"""
        MODIFIED NOW TO INCLUDE THE CUSTOM FINAL LAYERS
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        if self.conditional_twist:
            assert condition_twist_on_tokens is not None

        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        if condition_twist_on_tokens is not None:

            if attention_mask is not None:
                print(attention_mask.shape)
                # FORCES attention on all conditioning tokens
                attention_mask = torch.ones_like(condition_twist_on_tokens)

            condition_tokens_base_model_output = self.pretrained_model(
                input_ids=condition_twist_on_tokens,
                attention_mask=attention_mask,
                **kwargs,
            )
            # print(input_ids.shape)
            # print(condition_twist_on_tokens.shape)
            # print(base_model_output.hidden_states[-1].shape)
            # print(condition_tokens_base_model_output.hidden_states[-1].shape)

            last_hidden_state = torch.cat((base_model_output.hidden_states[-1], condition_tokens_base_model_output.hidden_states[-1].expand(base_model_output.hidden_states[-1].shape)), dim=-1)
            # print(condition_tokens_base_model_output.hidden_states[-1].expand(base_model_output.hidden_states[-1].shape).shape)
            # print(last_hidden_state.shape)
            # 1/0
        else:
            last_hidden_state = base_model_output.hidden_states[-1]


        # lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        if do_value_calc:
            value = self.v_head(last_hidden_state).squeeze(-1)
        else:
            value = None

        lm_logits = self.get_lm_logits(base_model_output, last_hidden_state)
        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

    def generate(self, inputs, max_length, *args, attention_mask=None, condition_twist_on_tokens=None, **kwargs):
        r"""
        CUSTOM. Only sample available right now.
        """

        input_ids = inputs

        # cum_logp = None

        while input_ids.shape[-1] < max_length:

            lm_logits, _, _ = self.forward(
                input_ids, return_dict=True,
                output_hidden_states=True, do_value_calc=False, attention_mask=attention_mask,
                condition_twist_on_tokens=condition_twist_on_tokens)

            next_token_logits = lm_logits[:, -1, :]

            # sample
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # print("Probs")
            # probs_selected = torch.gather(probs, 1, next_tokens[:, None])
            # print(probs_selected)
            # # print(probs.max(dim=-1))
            # # print(next_tokens)
            # i = 0
            # for x in next_tokens:
            #     print(probs[i, x])
            #     i += 1
            # if cum_logp is None:
            #     cum_logp = probs_selected.log()
            # else:
            #     cum_logp += probs_selected.log()
            # print("cum_logp")
            # print(cum_logp)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        return input_ids

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        nn_head_state_dict = self.nn_head.state_dict(*args, **kwargs)
        for k, v in nn_head_state_dict.items():
            pretrained_model_state_dict[f"nn_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
            if "nn_head." in k:
                state_dict[k.replace("nn_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        self.nn_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]

            self.v_head = self.v_head.to(first_device)
            self.nn_head = self.nn_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True


class CustomAutoModelForCausalLMWithValueHeadandConditionalTwist(CustomAutoModelForCausalLMWithValueHead):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, conditional_twist=True, **kwargs)

