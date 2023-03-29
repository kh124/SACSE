import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()
    cls.generator = transformers.DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased') if cls.model_args.generator_name is None else transformers.AutoModelForMaskedLM.from_pretrained(cls.model_args.generator_name)

@torch.no_grad()
def dequeue_and_enqueue(cls, keys):
    # gather keys before updating queue
    # keys = concat_all_gather(keys) #already concatenated before

    batch_size = keys.shape[0]

    ptr = int(cls.queue_ptr)
    assert cls.model_args.bank_size % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    cls.queue[ptr:ptr + batch_size, :] = keys
    # cls.queue[ptr:ptr + batch_size, :] = keys
    ptr = (ptr + batch_size) % cls.model_args.bank_size # move pointer

    cls.queue_ptr[0] = ptr
    # print("enque and dequeue processedddddddddddddddddddddddddddddddddddddddd")

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    cls_token=101,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    # num_sent = input_ids.size(1)

    mlm_outputs = None

    if cls.model_args.do_stronger:

        # set shuff
        cls.bert.set_flag("data_aug_shuffle", True)
        # set cutoff
        cls.bert.set_flag("data_aug_cutoff", True)
        cls.bert.set_flag("data_aug_cutoff.direction", "column")
        cls.bert.set_flag("data_aug_cutoff.rate", cls.model_args.cutoff_rate)
        # set others here

        stronger_input_ids = input_ids[:, 2]
        stronger_attention_mask = attention_mask[:, 2]
        if token_type_ids is not None:
            stronger_token_type_ids = token_type_ids[:, 2]

        stronger_outputs = cls.bert(
            stronger_input_ids,
            attention_mask=stronger_attention_mask,
            token_type_ids=stronger_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        stronger_pooler_output = cls.pooler(attention_mask, stronger_outputs)

        if cls.pooler_type == "cls":
            stronger_pooler_output = cls.mlp(stronger_pooler_output)

        # input_ids = input_ids[:, 0:2]
        # attention_mask = attention_mask[:, 0:2]
        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids[:, 0:2]

        input_ids = input_ids[:, 0]
        attention_mask = attention_mask[:, 0]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, 0]

        input_ids = input_ids.contiguous()
        attention_mask = attention_mask.contiguous()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.contiguous()

    num_sent = input_ids.size(1)
    # 如果使用双编码器，句子数量处理为1
    # input_ids = input_ids[:, 0]
    # attention_mask = attention_mask[:, 0]
    # if token_type_ids is not None:
    #     token_type_ids = token_type_ids[:, 0]
    #
    # input_ids = input_ids.contiguous()
    # attention_mask = attention_mask.contiguous()
    # if token_type_ids is not None:
    #     token_type_ids = token_type_ids.contiguous()

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # with torch.no_grad():
    # outputs_q = cls.encoder_q(
    #     input_ids,
    #     attention_mask=attention_mask,
    #     token_type_ids=token_type_ids,
    #     position_ids=position_ids,
    #     head_mask=head_mask,
    #     inputs_embeds=inputs_embeds,
    #     output_attentions=output_attentions,
    #     output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
    #     return_dict=True,
    # )
    # # Pooling
    # pooler_output_q = cls.pooler(attention_mask, outputs_q)
    # pooler_output_q = pooler_output_q.view((batch_size, num_sent, pooler_output_q.size(-1)))  # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
        # pooler_output_q = cls.mlp(pooler_output_q)

    # z1 = pooler_output
    # z2 = pooler_output_q
    # print("Shape of z1:", z1.shape)
    # print("Shape of z2", z2.shape)
    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # print("Shape of z11:", z1.shape)
    # print("Shape of z21", z2.shape)

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        # mlm_input_ids [64,2,32]
        # mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_input_ids = mlm_input_ids[:, 0]
        attention_mask = attention_mask[0:batch_size]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[0:batch_size]

        # print(mlm_input_ids[0:5])

        with torch.no_grad():
            g_pred = cls.generator(mlm_input_ids, attention_mask)[0].argmax(-1)
        g_pred[:, 0] = cls_token
        # print("The shape of g_pred is :\n", g_pred.shape)
        # print(g_pred[0:5])
        e_inputs = g_pred * attention_mask

        # print("The shape of e_inputs is :\n", e_inputs.shape)
        # print(e_inputs[0:5])
        # with torch.no_grad():
        #     mlm_outputs = encoder(
        #         # mlm_input_ids,
        #         e_inputs,
        #         attention_mask=attention_mask,
        #         token_type_ids=token_type_ids,
        #         position_ids=position_ids,
        #         head_mask=head_mask,
        #         inputs_embeds=inputs_embeds,
        #         output_attentions=output_attentions,
        #         output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        #         return_dict=True,
        #     )
        mlm_outputs = encoder(
            # mlm_input_ids,
            e_inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        pooler_mlm_outputs = cls.pooler(attention_mask, mlm_outputs)
        # pooler_mlm_outputs = pooler_mlm_outputs.view((batch_size, num_sent, pooler_mlm_outputs.size(-1)))
        # z3 = pooler_mlm_outputs[:, 0]
        z3 = pooler_mlm_outputs
        # mlm as stronger augmentation
        dequeue_and_enqueue(cls, z1)

    if cls.model_args.do_stronger:
        z3 = stronger_pooler_output[:, 0]
        dequeue_and_enqueue(cls, z1)

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)) #logits  shape(bs, bs)

    # Hard negative
    # if num_sent >= 3:
    #     z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
    #     cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
    loss_fct = nn.CrossEntropyLoss()

    # if num_sent >= 3:
    # if cls.model_args.do_stronger:
    if cls.model_args.do_stronger or cls.model_args.do_mlm:
        z2_bank_cos = cls.sim(z2.unsqueeze(1), cls.queue.clone().detach().unsqueeze(0))  # tenser shape(bs, K)
        z3_bank_cos = cls.sim(z2.unsqueeze(1), cls.queue.clone().detach().unsqueeze(0))

        softmax = nn.Softmax()
        z2_bank_cos_softmax = softmax(z2_bank_cos)
        z3_bank_cos_softmax = softmax(z3_bank_cos)

        z3_bank_cos_log = torch.log(z3_bank_cos_softmax).t()  # 计算log,转置

        ld_loss_tenser = torch.mm(z2_bank_cos_softmax, z3_bank_cos_log)
        ld_loss = ld_loss_tenser[range(ld_loss_tenser.shape[0]), range(ld_loss_tenser.shape[0])]  # 取出每一个样本标签值处的概率,对角线元素
        ld_loss = abs(sum(ld_loss) / ld_loss_tenser.shape[0])

    #每个原始句子独自作为一类，有多少句子就有多少标签（0，1，2, ..., n-1）
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device) #size(0）returns size of the fist dimension

    # loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    # if num_sent == 3:
    #     # Note that weights are actually logits of weights
    #     z3_weight = cls.model_args.hard_negative_weight
    #     weights = torch.tensor(
    #         [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    #     ).to(cls.device)
    #     cos_sim = cos_sim + weights #logits

    loss = loss_fct(cos_sim, labels)
    # Calculate loss for MLM
    # if mlm_outputs is not None and mlm_labels is not None:
    #     mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
    #     prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
    #     masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
    #     loss = loss + cls.model_args.mlm_weight * masked_lm_loss
    if cls.model_args.do_stronger or cls.model_args.do_mlm:
        loss = loss + cls.model_args.mlm_weight * ld_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)
        self.encoder_q = BertModel(config, add_pooling_layer=False)
        self.register_buffer("queue", torch.randn(self.model_args.bank_size, self.model_args.hidden_len))
        self.queue = nn.functional.normalize(self.queue, dim=1)  # across queue instead of each example
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
