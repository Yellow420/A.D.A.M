# Author: Chance Brownfield
# Email: ChanceBrownfield@protonmail.com
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.hubert.modeling_hubert import (
    HubertPreTrainedModel,
    HubertModel
)

from ADAM.Emodel.modeling_outputs import SpeechClassifierOutput
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, pipeline
from threading import Thread

class EmoGpt:
    def __init__(self, checkpoint="microsoft/phi-2"):
        # Download and load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)

        # Text generation pipeline
        self.emo_pipeline = pipeline(
            "text-generation",
            tokenizer=self.tokenizer,
            model=self.model,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            device_map="cpu"
        )

    def generate(self, message):
        instruction = "You are a helpful assistant that predicts personal feelings and emotional reactions. You only respond with the proper emotional reaction and predicted personal feelings. You only respond once as 'Assistant'."
        final_prompt = f"Instruction: {instruction}\nUser: {message}\nOutput:"

        # Streamer
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0)
        thread = Thread(target=self.emo_pipeline, kwargs={"text_inputs":final_prompt, "max_new_tokens":50, "streamer":streamer})
        thread.start()

        generated_text = ""
        for word in streamer:
            generated_text += word
            response = generated_text.strip()

            if "User:" in response:
                response = response.split("User:")[0].strip()

            if "Assistant:" in response:
                response = response.split("Assistant:")[1].strip()

            yield response

class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
