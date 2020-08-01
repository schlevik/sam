import os

import click

from scripts.utils import get_output_predictions_file_name


@click.command()
@click.argument('command', type=str)
@click.option('--sam', default='RB')
@click.option('--dataset-name', default='squad1')
@click.option('--model-name', default='bert-base-uncased')
@click.option('--model-type', default='bert')
@click.option("--train-file", default="train-v1.1.json")
@click.option("--eval-file", default="dev-v1.1.json")
@click.option("--batch-size", default=None)
@click.option("--extra-args", default="--fp16 --save-steps 0 --debug-features")
@click.option("--notify", type=str, default='viktor.schlegel@manchester.ac.uk')
def generate_dvc(command, sam, dataset_name, model_name, model_type, train_file, eval_file, batch_size, extra_args,
                 notify):
    sam_path = sam.lower()
    sam_dataset_path = f"data/football/{sam_path}"
    baseline_file = f"{sam_dataset_path}/baseline-{dataset_name}-{sam_path}.json"
    intervention_file = f"{sam_dataset_path}/intervention-{dataset_name}-{sam_path}.json"
    control_file = f"{sam_dataset_path}/control-{dataset_name}-{sam_path}.json"
    predictions_folder = f"data/football/{sam_path}/predictions"
    model_folder = f"{model_name}-{dataset_name}"
    model_path = f"models/{model_folder}"
    train_path = f"data/datasets/{dataset_name}/{train_file}"
    eval_path = f"data/datasets/{dataset_name}/{eval_file}"
    baseline_predictions = get_output_predictions_file_name(baseline_file, predictions_folder, model_folder)
    intervention_predictions = get_output_predictions_file_name(intervention_file, predictions_folder, model_folder)
    control_predictions = get_output_predictions_file_name(control_file, predictions_folder, model_folder)
    if not batch_size:
        batch_size = 8 if "large" in model_name else 24

    if command == 'train-transformers':

        cmd = (
            f"python main.py --debug --notify {notify} train {train_path} --model-path {model_name} "
            f"--model-type {model_name} --eval-file {eval_path} --save-model-folder {model_path} "
            f"--do-eval-after-training --num-workers 8 --per-gpu-train-batch-size {batch_size} "
            f"--max-answer-length 30 {extra_args}"
        )
        stage_name = f"train-{model_name}-on-{dataset_name}"
        dvc_cmd = (
            f"dvc run -n {stage_name} -d {train_path} -d {eval_path} -o {model_path} "
            f"{cmd}"
        )
    elif command == 'predict-transformers':

        cmd = (f"python main.py --debug --notify {notify} "
               f"predictions {baseline_file} {intervention_file} {control_file} "
               f"--model-path {model_path} --out-folder {predictions_folder} '--max-answer-length 10 {extra_args}")

        stage_name = f"eval-{model_folder}-on-{sam_path}"

        dvc_cmd = (f"dvc run -n {stage_name} -d scripts/predict_transformers.py -d {model_path} -d {baseline_file} "
                   f"-d {intervention_file} -d {control_file} -o {baseline_predictions} -o {control_predictions} "
                   f"-o {intervention_predictions}  {cmd}")
    elif command == 'evaluate':
        eoi_metric_file = f"{dataset_name}-{sam_path}.json"
        eoi_metric_path = f"metrics/{eoi_metric_file}"
        cmd = (f"python main.py evaluate-intervention --baseline-file {baseline_file} "
               f"--predictions-folder {predictions_folder} --control --output {eoi_metric_path}")

        stage_name = f"evaluate-intervention-{dataset_name}-{sam_path}"

        dvc_cmd = (f"dvc run -n {stage_name} -d {baseline_file} -d {intervention_file} -d {control_file} "
                   f"-d scripts/evaluate_intervention.py -d {predictions_folder} -o {eoi_metric_path} {cmd}")
    elif command == "generate":
        conf_name = f'conf/{dataset_name}.json'
        cmd = (f"python main.py generate-balanced --config {conf_name} --seed 56 "
               f"--num-workers 8 --do-save --out-path {sam_dataset_path} --modifier-type {sam}")
        stage_name = f"generate-{sam_path}-for-{dataset_name}"
        dvc_cmd = (f"dvc run -n {stage_name} -d scripts/generate_balanced.py "
                   f"-d {conf_name} -o {baseline_file} -o {intervention_file} -o {control_file} {cmd}")
    elif command == 'filter':
        # todo: evaluate all trained on dataset X on confs/test.json
        ...
        raise NotImplementedError()
    elif command == 'predict-allennlp':
        # todo: predict allennlp
        stage_name = f"eval-{model_name}-on-{dataset_name}"
        model_path = os.path.join(model_path, "model.tar.gz")
        cmd = (
            f"mkdir -p {predictions_folder} &&"
            f"allennlp predict {model_path} {baseline_file} --output-file {baseline_predictions} "
            f"--use-dataset-reader --silent && python main.py convert-allennlp {baseline_file} {baseline_predictions} "
            f"&& allennlp predict {model_path} {intervention_file} --output-file {intervention_predictions} "
            f"--use-dataset-reader --silent &&"
            f"python main.py convert-allennlp {intervention_file} {intervention_predictions} &&"
            f"allennlp predict {model_path} {control_file} --output-file {control_predictions} "
            f"--use-dataset-reader --silent && python main.py convert-allennlp {control_file} {control_predictions}"
        )

        dvc_cmd = (f"dvc run -n {stage_name} -d {model_path} -d {baseline_file} "
                   f"-d {intervention_file} -d {control_file} -o {baseline_predictions} -o {control_predictions} "
                   f"-o {intervention_predictions}  {cmd}")
    elif command == 'predict-baselines':
        # todo: simple baselines
        raise NotImplementedError()
    elif command == 'train-allennlp':
        cmd = f"TRAIN_SET={train_path} EVAL_SET={eval_path} CUDA=0 " \
              f"allennlp train conf/{model_name}.jsonnet -s {model_folder}"
        stage_name = f"train-{model_name}-on-{dataset_name}"
        dvc_cmd = f"dvc run -n {stage_name} -d {train_path} -d {eval_path} -o {model_path} {cmd}"
    else:
        raise NotImplementedError()
    click.secho("Python command:", fg='green', bold=True)
    click.echo(cmd)
    click.secho("DVC command:", fg='green', bold=True)
    click.echo(dvc_cmd)
