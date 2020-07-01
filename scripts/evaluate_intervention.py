import string

import click

from scripts.utils import write_json
from stresstest.eval_utils import get_mean_var_ci_bernoulli, EM

from stresstest.util import load_json, sample_iter, highlight


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
@click.option("--predictions", type=str, default='data/predictions.json')
@click.option("--predictions-intervention", type=str, default='data/predicitons_modifier.json')
@click.option("--gold", type=str, default='data/stresstest.json')
@click.option("--gold-intervention", type=str, default='data/stresstest_modifier.json')
@click.option("--output", type=str, default="metrics/score_intervention.json")
@click.option("--do-print", is_flag=True, default=False)
@click.option("--do-save", is_flag=True, default=False)
def evaluate_intervention(predictions, predictions_intervention, gold, gold_intervention, output, do_print, do_save):
    gold = load_json(gold)
    gold_intervention = load_json(gold_intervention)
    predictions = load_json(predictions)
    predictions_intervention = load_json(predictions_intervention)

    gold_flat = list(sample_iter(gold))
    gold_intervention_flat = list(sample_iter(gold_intervention))
    longest_answer = max(len(b.answer.split()) for b in gold_flat + gold_intervention_flat)
    q_m_by_id = {m.qa_id: m for m in gold_intervention_flat}
    aligned = [(b, q_m_by_id[b.qa_id]) for b in gold_flat if b.qa_id in q_m_by_id]
    aligned = [(b, m) for b, m in aligned if b.answer != m.answer]
    for b, intervention in aligned:
        assert b.qa_id == intervention.qa_id
    aligned_baseline, aligned_intervention = zip(*aligned)

    click.echo(f"Evaluating {click.style(str(len(aligned)), fg='green', bold=True)} sample(s).")
    em = EM(relaxed=True, max_length=3)
    results_baseline = em(aligned_baseline, predictions)
    click.echo(f"Got {sum(results_baseline)} correct for baseline.")

    # got it correct in first place
    correct_before_intervention = [
        (d, aligned_intervention[i], predictions[d.id][d.qa_id], predictions_intervention[d.id][d.qa_id]) for i, d in
        enumerate(aligned_baseline) if results_baseline[i] == 1
    ]

    results_intervention = em(aligned_intervention, predictions_intervention)
    click.echo(f"Got {sum(results_intervention)} correct for intervention.")

    # changed prediction correctly in line with altered semantics
    # model right -model changes prediction- model right
    correct_change_correct = [
        (d, aligned_intervention[i], predictions[d.id][d.qa_id], predictions_intervention[d.id][d.qa_id]) for i, d in
        enumerate(aligned_baseline) if results_baseline[i] == 1 and results_intervention[i] == 1
    ]
    click.echo(f"Out of {sum(results_baseline)} correct baseline results, got {len(correct_change_correct)} "
               f"correct after intervention.")

    # unimpressed: didn't change prediction although semantics dictated change
    # model right -model keeps prediction- model wrong
    correct_keep_wrong = [
        (d, aligned_intervention[i], predictions[d.id][d.qa_id], predictions_intervention[d.id][d.qa_id]) for i, d in
        enumerate(aligned_baseline) if
        predictions[d.id][d.qa_id] ==
        predictions_intervention[d.id][d.qa_id].replace("almost ", "").replace("nearly ", "")
        and results_baseline[i]
    ]
    click.echo(f"Interventions that the model 'ignored': {len(correct_keep_wrong)}")

    # confused: changed prediction when semantics dictated change but changed to wrong
    # model right -model changes prediction- model wrong
    correct_change_wrong = [
        (d, aligned_intervention[i], predictions[d.id][d.qa_id], predictions_intervention[d.id][d.qa_id]) for i, d in
        enumerate(aligned_baseline) if
        predictions[d.id][d.qa_id] !=
        predictions_intervention[d.id][d.qa_id].replace("almost ", "").replace("nearly ", "")
        and results_baseline[i] and not results_intervention[i]
    ]
    click.echo(f"Interventions that left the model 'confused': {len(correct_change_wrong)}")
    # model wrong -model changes prediction- model right
    wrong_change_right = [
        (d, aligned_intervention[i], predictions[d.id][d.qa_id], predictions_intervention[d.id][d.qa_id]) for i, d in
        enumerate(aligned_baseline) if
        predictions[d.id][d.qa_id] !=
        predictions_intervention[d.id][d.qa_id].replace("almost ", "").replace("nearly ", "")
        and not results_baseline[i] and results_intervention[i]
    ]
    # model wrong -model keeps prediction- model right
    wrong_keep_right = [
        (d, aligned_intervention[i], predictions[d.id][d.qa_id], predictions_intervention[d.id][d.qa_id]) for i, d in
        enumerate(aligned_baseline) if
        results_intervention[i]
    ]
    click.echo(f"Wrong predictions that the model changed to correct: {len(wrong_change_right)}")
    wrong_keep_right = [d for d in wrong_keep_right if d not in wrong_change_right and d not in correct_change_correct]
    click.echo(f"Wrong predictions that the model didn't change but that became correct: {len(wrong_keep_right)}")
    # sanity checks
    assert len(correct_before_intervention) > len(correct_change_correct)
    assert len(correct_before_intervention) == len(correct_keep_wrong) + len(correct_change_wrong) + len(
        correct_change_correct)
    assert sum(results_intervention) == len(correct_change_correct) + len(wrong_change_right) + len(
        wrong_keep_right), sum(
        results_intervention)
    subsets = [correct_change_correct, correct_keep_wrong, correct_change_wrong, wrong_change_right, wrong_keep_right]
    if do_print:
        click.secho(f"Those correct before intervention", fg='blue')
        labels = [
            click.style(t, fg='green', bold=True) for t in (
                "Correct after intervention", "Model ignored modifier",
                "Model was confused", "Model was wrong, changed the prediction and became right",
                "Model didn't change prediction, which became correct")
        ]
        for label, subset in zip(labels, subsets):
            click.echo(f"Examples for '{label}'")
            click.echo()
            for baseline, intervention, prediction, prediction_on_intervention in subset:
                click.echo(highlight(intervention.passage, colors=color_map(baseline.answer, intervention.answer)))
                assert baseline.question == intervention.question
                click.secho(baseline.question, fg='blue')
                click.secho(f"{baseline.answer} vs {prediction}", fg='green')
                click.secho(f"{intervention.answer} vs {prediction_on_intervention}", fg='red')
                click.echo(20 * "===")

    # result = {'num_samples': len(predictions), 'full': _get_score(gold, predictions)}
    overall_results = [1 if correct + results_intervention[i] == 2 else 0 for i, correct in enumerate(results_baseline)]
    mean, var, ci = get_mean_var_ci_bernoulli(overall_results)
    printable_result = f'{mean:.4f} +/- {ci:.4f}'

    # click.echo(f"Mean under the {click.style('EM', fg='green')} metric on "
    #            f"{click.style(printable_result, fg='green', bold=True)}")
    click.echo()
    result = {
        'evaluation_on_intervention': {
            'human_readable': printable_result,
            'mean': mean,
            '95ci': ci
        },
        'n': len(aligned),
        'correct_baseline': sum(results_baseline),
        'correct_intervention': sum(results_intervention),
        'right->change->right': len(correct_change_correct),
        'right->keep->wrong': len(correct_keep_wrong),
        'right->change->wrong': len(correct_change_wrong),
        'wrong->change->right': len(wrong_change_right),
        'wrong->keep->right': len(wrong_keep_right)
    }
    if do_save:
        write_json(result, output)
