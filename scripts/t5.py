# adapted from
# https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb#scrollTo=v7TUzb-T-YtF
import glob
import json
import logging
import os
import shutil
import string
import timeit
from dataclasses import dataclass, field
import random
from itertools import chain, islice
from typing import Dict, List, Optional, Tuple

import click
import wandb
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset

from transformers import T5ForConditionalGeneration, T5Tokenizer, SquadV1Processor, SquadExample, HfArgumentParser, \
    TrainingArguments, WEIGHTS_NAME
from transformers import Trainer, set_seed
from transformers.trainer_utils import is_wandb_available

from scripts.t5utils import T5ForConditionalGeneration4WayParallel, T5ForConditionalGeneration2WayParallel, MyTrainer

logger = logging.getLogger(__name__)

# process the examples in input and target text format and the eos token at the end
from transformers.data.metrics.squad_metrics import squad_evaluate

from scripts.utils import write_json, get_output_predictions_file_name
from scripts.utils_transformers import get_tokenizer
from stresstest.classes import YouIdiotException
from stresstest.util import batch


def add_eos_to_example(example: SquadExample):
    output = dict()
    output['input_text'] = 'question: %s  context: %s </s>' % (example.question_text, example.context_text)
    output['target_text'] = '%s </s>' % example.answer_text
    return output


# tokenize the examples
def convert_to_features(example_batch, tokenizer: T5Tokenizer, max_ans_length, max_context_length):
    inputs = [e['input_text'] for e in example_batch]
    targets = [e['target_text'] for e in example_batch]
    input_encodings = tokenizer.batch_encode_plus(inputs, truncation=True, pad_to_max_length=True,
                                                  max_length=max_context_length)
    target_encodings = tokenizer.batch_encode_plus(targets, truncation=True, pad_to_max_length=True,
                                                   max_length=max_ans_length)

    return [*zip(input_encodings['input_ids'], input_encodings['attention_mask'], target_encodings['input_ids'],
                 target_encodings['attention_mask'])]


# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
# @dataclass
# class T2TDataCollator:
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


def collate_eval(batch):
    input_ids = torch.stack([example[0] for example in batch])

    attention_mask = torch.stack([example[1] for example in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


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


# args = {
#     "num_cores": 8,
#     'training_script': 'train_t5_squad.py',
#     "model_name_or_path": 't5-base',
#     "max_len": 512,
#     'train_file_path': 'testsmall123/baseline-test-rb.json',
#     'valid_file_path': 'testsmall123/control-test-rb.json',
#     "target_max_len": 16,
#     "output_dir": './test-t5',
#     "overwrite_output_dir": True,
#     "per_gpu_train_batch_size": 2,
#     "per_gpu_eval_batch_size": 8,
#     "gradient_accumulation_steps": 2,
#     "learning_rate": 1e-4,
#     "num_train_epochs": 4,
#     "do_train": False,
#     "do_eval": True,
#     'no_cuda': True
# }

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    if training_args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    # if training_args.do_eval and not training_args.do_train and not data_args.predictions_folder:
    #     raise ValueError("Supply predictions folder destination to save the predictions!")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.debug(model_args)
    logger.debug(training_args)
    logger.debug(data_args)
    # raise NotImplementedError
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Set seed
    set_seed(training_args.seed)
    if training_args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    tokenizer = get_tokenizer(model_args.model_name_or_path, do_lower_case=False)
    if data_args.model_parallel == 4:
        model = T5ForConditionalGeneration4WayParallel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    elif data_args.model_parallel == 2:
        model = T5ForConditionalGeneration2WayParallel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    elif data_args.model_parallel is None:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError(f"Can only have no, 2way or 4way model parallelism! (expected: {data_args.model_parallel})")
    if training_args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    # Get datasets
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        eval_dataset, examples = get_dataset(data_args.eval_file_path, tokenizer, data_args, evaluate=True)
    else:
        eval_dataset, examples = None, None
    # Training
    if training_args.do_train:
        if training_args.local_rank in [-1, 0]:
            train_dataset, _ = get_dataset(data_args.train_file_path, tokenizer, data_args)
            torch.save(train_dataset, 'features.bin')
        else:
            torch.distributed.barrier()
            train_dataset = None

        if training_args.local_rank == 0:
            torch.distributed.barrier()

        else:
            train_dataset = torch.load('features.bin')
        # Initialize our Trainer
        if data_args.model_parallel:
            trainer = MyTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collate_training,
                prediction_loss_only=True
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collate_training,
                prediction_loss_only=True
            )
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        if training_args.do_train:
            model_path = os.path.basename(training_args.output_dir)
        else:
            model_path = os.path.basename(model_args.model_name_or_path)
        checkpoints = [training_args.output_dir]
        if data_args.eval_all_checkpoints and training_args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(training_args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            # logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info(f"Evaluate the following checkpoints: {checkpoints}")
        results = {}

        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1]
            if not all(s in string.digits for s in global_step):
                global_step = ''
            # no model parallelism here (didnt check model.generate)
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)
            device = torch.device("cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu")
            model.to(device)
            model_str = f'{model_path}-{global_step}' if global_step else model_path
            # Note that DistributedSampler samples randomly
            click.echo(f"Generating predictions for model {click.style(model_str, fg='blue')}, "
                       f"running on {click.style(str(training_args.device), fg='green')}")
            predictions = generate_predictions(eval_dataset, examples, model, tokenizer, training_args)
            final_metric = squad_evaluate(examples, predictions)

            if is_wandb_available():
                if training_args.do_train:
                    step = int(global_step) if global_step else trainer.global_step
                else:
                    step = 0
                # for now WANDB cannot 'log back in time'
                wandb.log(final_metric, step=step)
            print(f"GLOBAL STEP: {global_step}")
            result = dict(
                (k + ("_{}".format(global_step) if global_step else '_final'), v) for k, v in final_metric.items())

            logger.info(f"Result for {model_str}: {result}")
            results.update(result)

        # sort results by best
        checkpoint_scores = {
            c.split('_')[-1]: v for c, v in
            results.items() if any(c.endswith(digit) for digit in string.digits) and c.startswith('exact')
        }
        sorted_checkpoint_scores = {k: v for k, v in
                                    sorted(checkpoint_scores.items(), key=lambda k_v: k_v[1], reverse=True)}
        best_cp = next((c for c, v in sorted_checkpoint_scores.items() if v > results['exact_final']), None)

        if best_cp:
            click.echo(f"Best checkpoint is: {best_cp}")
            # copy over best results
            best_cp_folder = f'checkpoint-{best_cp}'

            click.echo(f"Copying over files: from {os.path.join(training_args.output_dir, best_cp_folder)} "
                       f"to {training_args.output_dir}")
            files_to_copy = glob.glob(os.path.join(training_args.output_dir, best_cp_folder, '*'))
            for file in files_to_copy:
                shutil.copy(file, training_args.output_dir)
        else:
            click.echo("best checkpoint is the last step...")
        # remove 'kek'points
        folders_to_remove = [p for p in glob.glob(os.path.join(training_args.output_dir, '*')) if os.path.isdir(p)]
        click.echo('Folders to remove: ')
        for folder in folders_to_remove:
            click.echo(f"Removing {folder}")
            shutil.rmtree(folder)
        if training_args.do_train:
            logger.info(results)
            write_json(results, os.path.join(training_args.output_dir, 'dev-results.json'))
        else:
            write_json(predictions, get_output_predictions_file_name(
                data_args.eval_file_path,
                training_args.output_dir,
                os.path.basename(os.path.normpath(model_args.model_name_or_path))
            ))


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


if __name__ == "__main__":
    main()
