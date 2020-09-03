import logging
import os
import random
import timeit
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Callable, Dict, Tuple, Union, Any, List

import click
import torch
from joblib import Parallel, delayed
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss, DataParallel
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, TensorDataset
from tqdm import trange, tqdm
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, EvalPrediction, set_seed, \
    is_torch_tpu_available, is_apex_available, T5ForConditionalGeneration, SquadV1Processor, T5Tokenizer, SquadExample
from transformers.trainer import get_tpu_sampler
from transformers.trainer_utils import is_wandb_available, PREFIX_CHECKPOINT_DIR

from scripts.utils import write_json, get_output_predictions_file_name
from stresstest.util import batch

logger = logging.getLogger(__name__)

if is_apex_available():
    from apex import amp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


class MyTrainer(Trainer):
    def __init__(self,
                 model: PreTrainedModel,
                 args: TrainingArguments,
                 data_collator: DataCollator,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 prediction_loss_only=False,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
                 ):
        self.model = model.to(args.device)
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch

    def _training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def train(self, model_path=None):
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                    self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        # if self.args.n_gpu > 1:
        #    model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                    self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)

                        if self.args.evaluate_during_training:
                            self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())


def forward_4way(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_value_states=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
):
    # print(fmt_dict(locals()))
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    if getattr(self, 'parallel', False):
        if self.is_decoder:
            device = torch.device('cuda:3')
        else:
            device = torch.device('cuda:0')
    else:
        device = input_ids.device
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        if self.is_decoder:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
        assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
        if self.parallel:
            inputs_embeds = self.embed_tokens(input_ids.to(torch.device('cuda:0'))).to(device)
        else:
            inputs_embeds = self.embed_tokens(input_ids)
    batch_size, seq_length = input_shape

    if past_key_value_states is not None:
        assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
            input_shape, (batch_size, 1)
        )
        # required mask seq length can be calculated via length of past
        # key value states and seq_length = 1 for the last token
        mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
    else:
        mask_seq_length = seq_length

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length).to(device)
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_seq_length = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(
            batch_size, encoder_seq_length, device=device, dtype=torch.long
        )

    # initialize past_key_value_states with `None` if past does not exist
    if past_key_value_states is None:
        past_key_value_states = [None] * len(self.block)

    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask.to(device), input_shape, device)
    extended_attention_mask = extended_attention_mask.to(device)
    if self.is_decoder and encoder_attention_mask is not None:
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask).to(device)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    present_key_value_states = ()
    all_hidden_states = ()
    all_attentions = ()
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)
    # gpu_change = False
    past_device = device
    for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # past_key_value_state = past_key_value_state.to(device) if hasattr(past_key_value_state,
        #                                                                   'to') else past_key_value_state
        this_head_mask = head_mask[i]
        if (getattr(layer_module, 'has_other_device', past_device) != past_device):  # \
            # or self.is_decoder and i == 0:
            past_device = layer_module.has_other_device
            # gpu_change = True
            (
                hidden_states,
                extended_attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                this_head_mask,
                use_cache,
                past_key_value_state,
                output_attentions,
            ) = tuple(
                t.to(past_device) if hasattr(t, 'to') else t for t in (
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    this_head_mask,
                    use_cache,
                    past_key_value_state,
                    output_attentions,
                ))
        # print((
        #     hidden_states,
        #     extended_attention_mask,
        #     position_bias,
        #     encoder_hidden_states,
        #     encoder_extended_attention_mask,
        #     encoder_decoder_position_bias,
        #     this_head_mask,
        #     use_cache,
        #     past_key_value_state,
        #     output_attentions,
        # ))
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=extended_attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            head_mask=this_head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
        hidden_states, present_key_value_state = layer_outputs[:2]

        if i == 0:
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[3 if output_attentions else 2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
        # append next layer key value states
        present_key_value_states = present_key_value_states + (present_key_value_state,)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
    # self.final_layer_norm = self.final_layer_norm.to(device)
    # if self.is_decoder:
    #  device2 = torch.device("cuda:1")
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if use_cache is True:
        assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
        outputs = outputs + (present_key_value_states,)
    if output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if output_attentions:
        outputs = outputs + (all_attentions,)
    return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)


class T5ForConditionalGeneration4WayParallel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.parallel = False

    def set_parallel(self):
        self.parallel = True
        # self.encoder: T5Stack
        # self.decoder: T5Stack
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')
        self.device2 = torch.device('cuda:2')
        self.device3 = torch.device('cuda:3')
        ratio1 = 0.3
        ratio2 = 0.6
        ratio3 = 0.9
        self.encoder.parallel = True
        self.encoder.forward = partial(forward_4way, self.encoder)
        # self.encoder.block.to(self.device1)
        encoder_block = list(self.encoder.block)
        split1 = int(len(encoder_block) * ratio1)
        split2 = int(len(encoder_block) * ratio2)
        split3 = int(len(encoder_block) * ratio3)
        # split_dec = int(len(encoder_block) * ratio3)
        # these will sit on cuda:0
        for i, module in enumerate(encoder_block[:split1]):
            logger.info(f"module {i} stays on cuda:0")

        # these will go to cuda:1
        for i, module in enumerate(encoder_block[split1:split2], split1):
            logger.info(f"moving module {i} to cuda:1!")
            module.to(self.device1)
            module.has_other_device = self.device1

        for i, module in enumerate(encoder_block[split2:split3], split2):
            logger.info(f"moving module {i} to cuda:2!")
            module.to(self.device2)
            module.has_other_device = self.device2

        for i, module in enumerate(encoder_block[split3:], split2):
            logger.info(f"moving module {i} to cuda:2!")
            module.to(self.device3)
            module.has_other_device = self.device3

        self.encoder.final_layer_norm.to(self.device3)

        # self.decoder.to(self.device1)
        # self.encoder.to(self.device0)
        self.decoder.parallel = True
        self.decoder.forward = partial(forward_4way, self.decoder)

        # these will go to device 2 and 3 respectively
        self.decoder.block.to(self.device3)
        for i, module in enumerate(self.decoder.block):
            logger.info(f"moving module of decoder block {i} to cuda:3!")
            # module.to(self.device3)
            module.has_other_device = self.device3

        # for i, module in enumerate(list(self.decoder.block)[split_dec:], split_dec):
        #     print(f"moving module {i} to cuda:2!")
        #     module.to(self.device2)
        #     module.has_other_device = self.device2

        self.decoder.final_layer_norm.to(self.device3)

        # self.lm_head = self.lm_head.to(self.device1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_past_key_value_states=None,
            use_cache=None,
            labels=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("lm_labels")

        # print(fmt_dict(locals()))
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        assert encoder_outputs is None
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # if self.parallel:
            #     self.encoder.embed_tokens.to(self.device0)
            #     (input_ids,
            #      attention_mask,
            #      inputs_embeds,
            #      head_mask,
            #      output_attentions,
            #      output_hidden_states,
            #      ) = tuple(t.to(self.device0) if t is not None else t for t in (input_ids,
            #                                                                     attention_mask,
            #                                                                     inputs_embeds,
            #                                                                     head_mask,
            #                                                                     output_attentions,
            #                                                                     output_hidden_states,
            #                                                                     ))

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        # if self.parallel:
        #     (decoder_input_ids,
        #      decoder_attention_mask,
        #      decoder_inputs_embeds,
        #      decoder_past_key_value_states,
        #      hidden_states,
        #      attention_mask,
        #      head_mask,
        #      use_cache,
        #      output_attentions,
        #      output_hidden_states,
        #      ) = tuple(t.to(self.device1) if hasattr(t, 'to') else t for t in (decoder_input_ids,
        #                                                                        decoder_attention_mask,
        #                                                                        decoder_inputs_embeds,
        #                                                                        decoder_past_key_value_states,
        #                                                                        hidden_states,
        #                                                                        attention_mask,
        #                                                                        head_mask,
        #                                                                        use_cache,
        #                                                                        output_attentions,
        #                                                                        output_hidden_states,
        #                                                                        ))
        #     self.decoder.embed_tokens.to(self.device1)
        # print(f"DECODER IS ON DEVICE: {self.decoder.device}")
        # print(f"ARE THEY THE SAME? {self.decoder.embed_tokens == self.encoder.embed_tokens}")
        # print(f"ARE THE BLOCKS THE SAME TOO? {self.decoder.block == self.encoder.block}")

        # print(f"DECODER EMBED TOKENS IS ON DEVICE: {}")
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # print("SEQUENCE OUTPUT")
        # print(sequence_output)
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        # print("SEQUENCE OUTPUT")
        # print(sequence_output)
        # if self.parallel:
        #     self.lm_head = self.lm_head.to('cuda:1')
        if self.parallel:
            sequence_output = sequence_output.to(self.device0)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:
            # if self.parallel:
            #    labels = labels.to(self.device1)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))  # .to('cuda:0')
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs

        return decoder_outputs + encoder_outputs


def foward_2way(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_value_states=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    if self.is_decoder:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cuda:0')
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        if self.is_decoder:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
        assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
        inputs_embeds = self.embed_tokens(input_ids.to(torch.device('cuda:0'))).to(device)

    batch_size, seq_length = input_shape

    if past_key_value_states is not None:
        assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
            input_shape, (batch_size, 1)
        )
        # required mask seq length can be calculated via length of past
        # key value states and seq_length = 1 for the last token
        mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
    else:
        mask_seq_length = seq_length

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length).to(device)
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_seq_length = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(
            batch_size, encoder_seq_length, device=device, dtype=torch.long
        )

    # initialize past_key_value_states with `None` if past does not exist
    if past_key_value_states is None:
        past_key_value_states = [None] * len(self.block)

    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask.to(device), input_shape, device)
    extended_attention_mask = extended_attention_mask.to(device)
    if self.is_decoder and encoder_attention_mask is not None:
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask).to(device)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    present_key_value_states = ()
    all_hidden_states = ()
    all_attentions = ()
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)
    gpu_change = False
    for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # past_key_value_state = past_key_value_state.to(device) if hasattr(past_key_value_state,
        #                                                                   'to') else past_key_value_state
        this_head_mask = head_mask[i]
        if not self.is_decoder and not gpu_change and getattr(layer_module, 'has_other_device', False):
            gpu_change = True
            device1 = torch.device('cuda:1')
            (
                hidden_states,
                extended_attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                this_head_mask,
                use_cache,
                past_key_value_state,
                output_attentions,
            ) = tuple(
                t.to(device1) if hasattr(t, 'to') else t for t in (
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    this_head_mask,
                    use_cache,
                    past_key_value_state,
                    output_attentions,
                ))

        layer_outputs = layer_module(
            hidden_states,
            attention_mask=extended_attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            head_mask=this_head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
        hidden_states, present_key_value_state = layer_outputs[:2]

        if i == 0:
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[3 if output_attentions else 2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
        # append next layer key value states
        present_key_value_states = present_key_value_states + (present_key_value_state,)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
    # self.final_layer_norm = self.final_layer_norm.to(device)
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if use_cache is True:
        assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
        outputs = outputs + (present_key_value_states,)
    if output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if output_attentions:
        outputs = outputs + (all_attentions,)
    return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)


class T5ForConditionalGeneration2WayParallel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.parallel = False

    def set_parallel(self):
        self.parallel = True
        # self.encoder: T5Stack
        # self.decoder: T5Stack
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')
        ratio = 0.7
        self.encoder.forward = partial(foward_2way, self.encoder)
        # self.encoder.block.to(self.device1)
        encoder_block = list(self.encoder.block)
        for i, module in enumerate(encoder_block[int(len(encoder_block) * ratio):], int(len(encoder_block) * ratio)):
            logger.info(f"moving module {i} to cuda:1!")
            module.to(self.device1)
            module.has_other_device = True
        self.encoder.final_layer_norm.to(self.device1)

        # self.decoder.to(self.device1)
        # self.encoder.to(self.device0)

        self.decoder.forward = partial(foward_2way, self.decoder)
        self.decoder.block.to(self.device1)
        self.decoder.final_layer_norm.to(self.device1)

        # self.lm_head = self.lm_head.to(self.device1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_past_key_value_states=None,
            use_cache=None,
            labels=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("lm_labels")

        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        assert encoder_outputs is None
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        if self.parallel:
            sequence_output = sequence_output.to(self.device0)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:
            # if self.parallel:
            #    labels = labels.to(self.device1)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))  # .to('cuda:0')
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs

        return decoder_outputs + encoder_outputs


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # predictions_folder: str = field(
    #     default=None,
    #     metadata={"help": "Path to the folder to save the predictions."},
    # )
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    eval_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of workers to pre-process the dataset."},
    )
    debug: bool = field(
        default=...
    )
    eval_all_checkpoints: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate all checkpoints."}
    )
    model_parallel: int = field(
        default=None,
        metadata={"help": "Accepts strictly 0 2 and 4 as values and will employ no, double or 4 way model parallelism."}
    )


def get_dataset(in_file, tokenizer, args: DataTrainingArguments, evaluate=False):
    processor = SquadV1Processor()
    data_dir = os.path.dirname(in_file)
    file_name = os.path.basename(in_file)
    if evaluate:
        examples = processor.get_dev_examples(data_dir, filename=file_name)
    else:
        examples = processor.get_train_examples(data_dir, filename=file_name)
    processed_examples = [add_eos_to_example(e) for e in examples]
    # TODO: something parallel?
    if args.num_workers > 1:
        features = Parallel(args.num_workers)(
            delayed(convert_to_features)(e, tokenizer, args.target_max_len, args.max_len) for e in
            batch(tqdm(processed_examples), batch_size=10)
        )
    else:
        features = [convert_to_features(e, tokenizer, args.target_max_len, args.max_len) for e in
                    batch(tqdm(processed_examples), batch_size=10)]
    # flatten batched list
    features = [f for fs in features for f in fs]
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_target_attention_mask = torch.tensor([f[3] for f in features], dtype=torch.long)
    n = random.randint(0, len(features))
    logger.info("Random Question")
    logger.info(examples[n].question_text)
    logger.info(" ".join(f"[{tokenizer.decode(e.item())}]" for e in all_input_ids[n] if e))
    logger.info("Its Answer")
    logger.info(examples[n].answer_text)
    logger.info(" ".join(f"[{tokenizer.decode(e.item())}]" for e in all_target_ids[n] if e))
    dataset = TensorDataset(
        all_input_ids,
        all_attention_masks,
        all_target_ids,
        all_target_attention_mask,
    )
    # if debug_features:
    #    debug_features_examples_dataset(dataset, examples, features, tokenizer)
    # return dataset, examples, features
    return dataset, examples


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    do_lower_case: Optional[str] = field(
        default=True, metadata={"help": "Whether to lowercase. (Not sure if it does anything for t5)."}
    )


def collate_eval(batch):
    input_ids = torch.stack([example[0] for example in batch])

    attention_mask = torch.stack([example[1] for example in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def collate_training(batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    # input_ids = torch.stack([example['input_ids'] for example in batch])
    # lm_labels = torch.stack([example['target_ids'] for example in batch])
    # lm_labels[lm_labels[:, :] == 0] = -100
    # attention_mask = torch.stack([example['attention_mask'] for example in batch])
    # decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
    input_ids = torch.stack([example[0] for example in batch])
    lm_labels = torch.stack([example[2] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100

    attention_mask = torch.stack([example[1] for example in batch])
    decoder_attention_mask = torch.stack([example[3] for example in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lm_labels': lm_labels,
        'decoder_attention_mask': decoder_attention_mask
    }


def convert_to_features(example_batch, tokenizer: T5Tokenizer, max_ans_length, max_context_length):
    inputs = [e['input_text'] for e in example_batch]
    targets = [e['target_text'] for e in example_batch]
    input_encodings = tokenizer.batch_encode_plus(inputs, truncation=True, pad_to_max_length=True,
                                                  max_length=max_context_length)
    target_encodings = tokenizer.batch_encode_plus(targets, truncation=True, pad_to_max_length=True,
                                                   max_length=max_ans_length)

    return [*zip(input_encodings['input_ids'], input_encodings['attention_mask'], target_encodings['input_ids'],
                 target_encodings['attention_mask'])]


def add_eos_to_example(example: SquadExample):
    output = dict()
    output['input_text'] = 'question: %s  context: %s </s>' % (example.question_text, example.context_text)
    output['target_text'] = '%s </s>' % example.answer_text
    return output


def generate_predictions(eval_dataset, examples, model, tokenizer, training_args):
    # n_gpu = 0 if training_args.no_cuda else torch.cuda.device_count()
    # eval_batch_size = training_args.per_gpu_eval_batch_size * max(1, n_gpu)
    answers = []
    eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, sampler=eval_sampler,
                                                  batch_size=training_args.per_device_eval_batch_size,
                                                  collate_fn=collate_eval)
    # multi-gpu evaluate
    # if training_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    #    model = torch.nn.DataParallel(model)
    # Eval!

    start_time = timeit.default_timer()
    results = []
    for batch in tqdm(eval_dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(training_args.device)
        with torch.no_grad():
            outs = model.generate(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask'],
                                  max_length=16,
                                  early_stopping=True)
        outs = [tokenizer.decode(ids) for ids in outs]
        answers.extend(outs)
        logger.debug(outs)
        results.extend(outs)
    eval_time = timeit.default_timer() - start_time
    logger.info(f"Evaluation done in total {eval_time} secs ({eval_time / len(eval_dataset)} sec per example)")
    predictions = dict()
    for result, example in zip(results, examples):
        if predictions.get(example.qas_id):
            logger.warning("Duplicate entry detected...")
        predictions[example.qas_id] = result
    return predictions


@click.command()
@click.argument("in-files", nargs=-1)
@click.option('--out-folder', type=str, required=True)
@click.option("model_paths", '--model-path', type=str, multiple=True)
@click.option('--no-cuda', type=bool, is_flag=True, default=None)
@click.option('--per-gpu-eval-batch-size', type=int, default=8)
@click.option('--max-answer-length', type=int, default=16)
@click.option('--max-seq-length', type=int, default=512)
@click.option('--num-workers', type=int, default=4)
@click.option('--debug-features', type=bool, is_flag=True, default=False)
def t5_predictions(in_files, out_folder, model_paths, no_cuda, per_gpu_eval_batch_size, num_workers,
                   debug_features, max_seq_length, max_answer_length):
    data_args = DataTrainingArguments(
        max_len=max_seq_length,
        target_max_len=max_answer_length,
        debug=debug_features,
        num_workers=num_workers
    )
    training_args = TrainingArguments(
        no_cuda=no_cuda,
        output_dir=out_folder,
        per_gpu_eval_batch_size=per_gpu_eval_batch_size,

    )
    logger.info(data_args)
    logger.info(training_args)
    for model_path in model_paths:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        for in_file in in_files:
            eval_dataset, examples = get_dataset(in_file, tokenizer, data_args, evaluate=True)
            predictions = generate_predictions(eval_dataset, examples, model, tokenizer, training_args)
            model.to(training_args.device)
            if training_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = DataParallel(model)
            write_json(
                predictions,
                get_output_predictions_file_name(in_file, out_folder, os.path.basename(os.path.normpath(model_path)))
            )
