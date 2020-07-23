import os
import string
from typing import Dict

import click

from scripts.utils import write_json, match_prediction_to_gold, BASELINE, INTERVENTION, extract_model_name, CONTROL
from stresstest.eval_utils import get_mean_var_ci_bernoulli, align
from stresstest.eval_utils import evaluate_intervention as eval_intervention

from stresstest.util import load_json, highlight


def color_map(baseline, intervention, other=None):
    other = other or ['almost', 'nearly']
    if any(d in baseline for d in string.digits):
        baseline = [b for b in baseline.split(" ") if all(c in string.digits for c in b)]
    else:
        baseline = baseline.split(" ")
    if any(d in intervention for d in string.digits):
        baseline = [b for b in intervention.split(" ") if all(c in string.digits for c in b)]
    else:
        intervention = intervention.split(" ")
    result = dict()
    for b in baseline:
        result[b] = 'green'
    for i in intervention:
        result[i] = 'red'
    for o in other:
        result[o] = 'yellow'
    return result


@click.command()
@click.option("--baseline-file", type=str, default='data/baseline.json')
@click.option("--predictions-folder", type=str, default='data/predictions')
@click.option("--output", type=str, default="metrics/result-intervention.json")
@click.option("--do-print", is_flag=True, default=False)
@click.option("--do-save", is_flag=True, default=False)
@click.option("--control", is_flag=True, default=False)
def evaluate_intervention(predictions_folder, baseline_file, output, do_print, do_save, control):
    gold = load_json(baseline_file)
    intervention_basename = os.path.basename(baseline_file).replace(BASELINE, INTERVENTION)
    intervention_file = baseline_file.replace(os.path.basename(baseline_file), intervention_basename)
    gold_intervention = load_json(intervention_file)
    gold_descriptor, prediction_files = match_prediction_to_gold(baseline_file, predictions_folder)
    gold_intervention_descriptor, prediction_intervention_files = match_prediction_to_gold(intervention_file,
                                                                                           predictions_folder)

    click.echo(f"Evaluation by intervention with baseline gold: {click.style(baseline_file, fg='blue')}")
    click.echo(f"And intervention gold: {click.style(intervention_file, fg='blue')}")

    # gold_flat = list(sample_iter(gold))
    # gold_intervention_flat = list(sample_iter(gold_intervention))

    # q_m_by_id = {m.qa_id: m for m in gold_intervention_flat}
    #
    # aligned = [(b, q_m_by_id[b.qa_id]) for b in gold_flat if b.qa_id in q_m_by_id]
    # aligned = [(b, m) for b, m in aligned if b.answer != m.answer]
    #
    # for b, intervention in aligned:
    #     assert b.qa_id == intervention.qa_id
    # aligned_baseline, aligned_intervention = zip(*aligned)

    if control:
        control_basename = os.path.basename(baseline_file).replace(BASELINE, CONTROL)
        control_file = baseline_file.replace(os.path.basename(baseline_file), control_basename)
        gold_control = load_json(control_file)
        # gold_control_flat = list(sample_iter(gold_control))
        # q_c_by_id = {c.qa_id: c for c in gold_control_flat}
        # aligned = [(b, q_m_by_id[b.qa_id], q_c_by_id[b.qa_id]) for b in gold_flat if b.qa_id in q_c_by_id]
        # aligned = [(b, m, c) for b, m, c in aligned if b.answer != m.answer]
        _, control_prediction_files = match_prediction_to_gold(control_file, predictions_folder)

        click.echo(f"And control gold: {click.style(control_file, fg='blue')}")
        # assert c_aligned_baseline == aligned_baseline, c_aligned_intervention == aligned_intervention
    else:
        control_prediction_files = [""] * len(prediction_files)
        gold_control = None

    aligned_baseline, aligned_intervention, aligned_control = align(gold, gold_intervention,
                                                                    gold_control)

    result = dict()

    for predictions_file, prediction_intervention_file, control_prediction_file in \
            zip(sorted(prediction_files), sorted(prediction_intervention_files), sorted(control_prediction_files)):
        predictions: Dict[str, str] = load_json(predictions_file)
        predictions_intervention: Dict[str, str] = load_json(prediction_intervention_file)
        model_name = extract_model_name(gold_descriptor, predictions_file)
        click.echo(f"Evaluating predictions of model {click.style(model_name, fg='green')}")
        click.echo(f"Evaluating {click.style(str(len(aligned_baseline)), fg='green', bold=True)} sample(s).")
        predictions_control: Dict[str, str] = load_json(control_prediction_file) if control_prediction_file else None
        (
            overall_result, results_baseline, results_intervention, results_control,
            correct_before_intervention, correct_change_correct, correct_keep_wrong, correct_change_wrong,
            wrong_change_right, wrong_keep_right, correct_baseline_control, correct_baseline_control_intervention
        ) = eval_intervention(aligned_baseline, aligned_intervention, aligned_control, predictions, predictions_intervention,
                                  predictions_control)
        click.echo(f"Got {sum(results_baseline)} correct for baseline.")
        click.echo(f"Got {sum(results_intervention)} correct for intervention.")
        click.echo(f"Out of {sum(results_baseline)} correct baseline results, got {len(correct_change_correct)} "
                   f"correct after intervention.")
        click.echo(f"Interventions that the model 'ignored': {len(correct_keep_wrong)}")
        click.echo(f"Interventions that left the model 'confused': {len(correct_change_wrong)}")
        click.echo(f"Wrong predictions that the model changed to correct: {len(wrong_change_right)}")
        click.echo(f"Wrong predictions that the model didn't change but that became correct: {len(wrong_keep_right)}")

        if do_print:
            print_examples(correct_baseline_control, correct_baseline_control_intervention, correct_change_correct,
                           correct_keep_wrong, correct_change_wrong, wrong_change_right, wrong_keep_right)

        mean, var, ci = get_mean_var_ci_bernoulli(overall_result)
        printable_result = f'{mean:.4f} +/- {ci:.4f}'

        # click.echo(f"Mean under the {click.style('EM', fg='green')} metric on "
        #            f"{click.style(printable_result, fg='green', bold=True)}")

        result[model_name] = {
            'evaluation_on_intervention': {
                'human_readable': printable_result,
                'mean': mean,
                '95ci': ci,
                'control': control
            },
            'n': len(aligned_baseline),
            'behaviour': {
                'correct_baseline': sum(results_baseline),
                'correct_intervention': sum(results_intervention),
                'right->change->right': len(correct_change_correct),
                'right->keep->wrong': len(correct_keep_wrong),
                'right->change->wrong': len(correct_change_wrong),
                'wrong->change->right': len(wrong_change_right),
                'wrong->keep->right': len(wrong_keep_right)
            }
        }
        if control:
            click.echo(f"Got {sum(results_control)} correct for control.")
            result[model_name]['behaviour'].update({
                'correct_control': sum(results_control),
                'correct_baseline_control': len(correct_baseline_control),
                'right+control->change->right': len(correct_baseline_control_intervention)
            })
        click.echo(f"Overall result: {printable_result}.")
        click.echo()
    if do_save:
        write_json(result, output)


def print_examples(correct_baseline_control, correct_baseline_control_intervention, correct_change_correct,
                   correct_keep_wrong, correct_change_wrong, wrong_change_right,
                   wrong_keep_right, max=10):
    click.echo(f"Examples for 'Correct baseline and Control'")
    click.echo()

    for baseline, ctrl, prediction, prediction_control in correct_baseline_control[:max]:
        click.echo(highlight(baseline.passage, colors=color_map(baseline.answer, ctrl.answer)))
        click.echo(highlight(ctrl.passage, colors=color_map(baseline.answer, ctrl.answer)))
        assert baseline.question == ctrl.question
        click.secho(baseline.question, fg='blue')
        click.secho(f"{baseline.answer} vs {prediction}", fg='green')
        click.secho(f"{ctrl.answer} vs {prediction_control}", fg='red')
        click.echo(20 * "===")
    for b, i, c, p_b, p_i, p_c in correct_baseline_control_intervention[:max]:
        click.echo(highlight(i.passage, colors=color_map(b.answer, i.answer)))
        click.echo(highlight(c.passage, colors=color_map(b.answer, c.answer)))
        assert b.question == c.question == i.question
        assert i.answer == c.answer
        click.secho(b.question, fg='blue')
        click.secho(f"{b.answer} vs {p_b}", fg='green')
        click.secho(f"{c.answer} vs {p_c}", fg='red')
        click.secho(f"{i.answer} vs {p_i}", fg='magenta')
        click.echo(20 * "===")
    labels = [
        click.style(t, fg='green', bold=True) for t in (
            "Correct after intervention", "Model ignored modifier",
            "Model was confused", "Model was wrong, changed the prediction and became right",
            "Model didn't change prediction, which became correct")
    ]
    subsets = [correct_change_correct, correct_keep_wrong, correct_change_wrong, wrong_change_right,
               wrong_keep_right]
    for label, subset in zip(labels, subsets):
        click.echo(f"Examples for '{label}'")
        click.echo()
        for baseline, intervention, prediction, prediction_on_intervention in subset[:max]:
            click.echo(highlight(intervention.passage, colors=color_map(baseline.answer, intervention.answer)))
            assert baseline.question == intervention.question
            click.secho(baseline.question, fg='blue')
            click.secho(f"{baseline.answer} vs {prediction}", fg='green')
            click.secho(f"{intervention.answer} vs {prediction_on_intervention}", fg='red')
            click.echo(20 * "===")
