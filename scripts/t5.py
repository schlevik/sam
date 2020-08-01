import json
import os
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset

from transformers import T5ForConditionalGeneration, T5Tokenizer, SquadV1Processor, SquadExample
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)
from loguru import logger

# process the examples in input and target text format and the eos token at the end
from transformers.data.metrics.squad_metrics import squad_evaluate

from scripts.utils import write_json
from scripts.utils_transformers import get_tokenizer
from stresstest.util import batch, sample_iter, load_json


def add_eos_to_example(example: SquadExample):
    output = dict()
    output['input_text'] = 'question: %s  context: %s </s>' % (example.question_text, example.context_text)
    output['target_text'] = '%s </s>' % example.answer_text
    return output


# tokenize the examples
def convert_to_features(example_batch, tokenizer: T5Tokenizer, max_ans_length, max_context_length):
    inputs = [e['input_text'] for e in example_batch]
    targets = [e['target_text'] for e in example_batch]
    input_encodings = tokenizer.batch_encode_plus(inputs, pad_to_max_length=True, max_length=max_context_length)
    target_encodings = tokenizer.batch_encode_plus(targets, pad_to_max_length=True, max_length=max_ans_length)

    return [*zip(input_encodings['input_ids'], input_encodings['attention_mask'], target_encodings['input_ids'],
                 target_encodings['attention_mask'])]


def get_dataset(in_file, tokenizer, evaluate=False):
    processor = SquadV1Processor()
    data_dir = os.path.dirname(in_file)
    file_name = os.path.basename(in_file)
    if evaluate:
        examples = processor.get_dev_examples(data_dir, filename=file_name)
    else:
        examples = processor.get_train_examples(data_dir, filename=file_name)
    processed_examples = [add_eos_to_example(e) for e in examples]
    features = [convert_to_features(e, tokenizer, 10, 384) for e in batch(tqdm(processed_examples), batch_size=10)]
    # flatten batched list
    features = [f for fs in features for f in fs]
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_target_attention_mask = torch.tensor([f[3] for f in features], dtype=torch.long)
    n = random.randint(0, len(features))
    logger.info("Random Question")
    logger.debug(examples[n].question_text)
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


# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T2TDataCollator:
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    args = {
        "num_cores": 8,
        'training_script': 'train_t5_squad.py',
        "model_name_or_path": 't5-base',
        "max_len": 512,
        'train_file_path': 'testsmall123/baseline-test-rb.json',
        'valid_file_path': 'testsmall123/control-test-rb.json',
        "target_max_len": 16,
        "output_dir": './test-t5',
        "overwrite_output_dir": True,
        "per_gpu_train_batch_size": 2,
        "per_gpu_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "num_train_epochs": 4,
        "do_train": False,
        "do_eval": True,
        'no_cuda': True
    }
    path = 'args.json'
    with open(path, 'w+') as f:
        json.dump(args, f)
    # we will load the arguments from a json file,
    # make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_json_file(path)
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

    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
    #     training_args.local_rank,
    #     training_args.device,
    #     training_args.n_gpu,
    #     bool(training_args.local_rank != -1),
    #     training_args.fp16,
    # )
    # logger.info("Training/evaluation parameters %s" % training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print('loading data')
    train_dataset, _ = get_dataset(data_args.train_file_path, tokenizer)
    valid_dataset, examples = get_dataset(data_args.valid_file_path, tokenizer, evaluate=True)
    logger.debug(examples[0])
    print('loading done')
    logger.debug(model_args)
    logger.debug(training_args)
    logger.debug(data_args)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
        prediction_loss_only=True
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        # logger.info("*** Evaluate ***")
        #
        # eval_output = trainer.evaluate()
        #
        # output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     logger.info("***** Eval results *****")
        #     for key in sorted(eval_output.keys()):
        #         logger.info("  %s = %s", key, str(eval_output[key]))
        #         writer.write("%s = %s\n" % (key, str(eval_output[key])))
        # results.update(eval_output)
        model = T5ForConditionalGeneration.from_pretrained(training_args.output_dir)
        tokenizer = T5Tokenizer.from_pretrained(training_args.output_dir)
        answers = []

        def collate_f(batch):
            input_ids = torch.stack([example[0] for example in batch])

            attention_mask = torch.stack([example[1] for example in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

        dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, collate_fn=collate_f)
        results = []
        for batch in tqdm(dataloader):
            outs = model.generate(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask'],
                                  max_length=16,
                                  early_stopping=True)
            outs = [tokenizer.decode(ids) for ids in outs]
            answers.extend(outs)
            logger.debug(outs)
            results.extend(outs)
        predictions = dict()
        for result, entry in zip(results, sample_iter(load_json(data_args.valid_file_path))):
            predictions[entry.qa_id] = result
        final_metric = squad_evaluate(examples, predictions)
        logger.debug(final_metric)
        write_json(final_metric, os.path.join(training_args.output_dir, 'eval-predictions.json'))

    return results


if __name__ == "__main__":
    main()
