import glob
import json
import os

import click
from loguru import logger

from stresstest import implemented_domains
from stresstest.eval_utils import EvalMetric
from stresstest.textmetrics import Distance
from stresstest.util import do_import

BASELINE = 'baseline'
INTERVENTION = 'intervention'


class Domain(click.ParamType):
    def convert(self, value, param, ctx):
        try:
            return do_import(f'{value}.bundle', relative_import='stresstest')
        except AttributeError:
            self.fail(f"Domain '{value}' unknown, please choose from {implemented_domains}")


def get_all_subclasses(cls):
    """
    Returns all (currently imported) subclasses of a given class.

    :param cls: Class to get subclasses of.

    :return: all currently imported subclasses.
    """
    return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in get_all_subclasses(c))


class FormatParam(click.ParamType):
    def convert(self, value, param, ctx):
        try:
            return do_import(value, relative_import='stresstest.ds_utils')
        except AttributeError:
            import inspect
            from stresstest import ds_utils
            all_functions = inspect.getmembers(ds_utils, inspect.isfunction)
            all_function_names = [name for name, func in all_functions]
            self.fail(f"Format '{value}' unknown, please choose from {all_function_names}.")


class MetricParam(click.ParamType):
    def convert(self, value, param, ctx):
        vals = value.split(',')
        metric_classes = []
        for val in vals:
            try:
                metric_classes.append(do_import(val, relative_import='stresstest.textmetrics'))
            except AttributeError:
                self.fail(f"Metric '{value}' unknown, please choose from {get_all_subclasses(Distance)}")
        return metric_classes


class EvalMetricParam(click.ParamType):
    def convert(self, value, param, ctx):
        vals = value.split(',')
        metric_classes = []
        for val in vals:
            try:
                metric_classes.append(do_import(val, relative_import='stresstest.eval_utils'))
            except AttributeError:
                self.fail(f"Metric '{value}' unknown, please choose from {get_all_subclasses(EvalMetric)}")
        return metric_classes


def write_json(result, path, pretty=True):
    if os.path.dirname(path).replace(".", ""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w+") as f:
        if pretty:
            json.dump(result, f, indent=4, separators=(',', ': '))
        else:
            json.dump(result, f)


def get_templates(templates, action: str = None, n: int = None, command: str = "Executing command"):
    sentences = templates['sentences']
    actions = list(sentences.keys())
    if action is not None:
        actions = [action]

    if n is not None:
        for action in actions:
            sentences[action] = [sentences[action][n]]

    actions_str = click.style(', '.join(actions), fg='blue')
    n_str = click.style(text=str(n) if n else "all", fg='green', bold=True)
    click.echo(f"{click.style(command, fg='red')} for actions: '{actions_str}'; sentences: {n_str} !")
    return actions, sentences


def match_prediction_to_gold(gold_file, prediction_folder):
    gold_descriptor = os.path.splitext(os.path.basename(gold_file))[0]
    logger.debug(f"Files in prediction folder: {glob.glob(os.path.join(prediction_folder, '*.json'))}")

    prediction_files = [p for p in glob.glob(os.path.join(prediction_folder, '*')) if
                        os.path.basename(p).startswith(gold_descriptor)]
    logger.debug(f"Files matching prefix '{gold_descriptor}': {prediction_files}")
    return gold_descriptor, prediction_files


def extract_model_name(gold_descriptor, prediction_file):
    model_name = os.path.splitext(prediction_file)[0].split(gold_descriptor)[1][1:]
    return model_name


def get_output_predictions_file_name(in_file, output_folder, weights_path):
    output_base = os.path.splitext(os.path.basename(in_file))[0]
    weights_addon = os.path.basename(weights_path)
    weights_addon = weights_addon.replace(".tar", "").replace(".gz", "").replace(".zip", "").replace(".tgz", "")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_file_name = f"{output_base}-{weights_addon}.json"
    output = os.path.join(output_folder, output_file_name)
    return output