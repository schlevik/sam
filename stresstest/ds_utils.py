from dataclasses import asdict
from typing import List

from stresstest.classes import EventPlan


def squad(dataset):
    data = []
    for d in dataset:
        qas = []
        paragraph = d['passage']
        for qa in d['qas']:
            answer = qa['answer']
            idx_answer_in_paragraph = match_answer_in_paragraph(qa, d)
            qas.append({
                'question': qa['question'],
                'id': qa['id'],
                'answers': [{'answer_start': idx_answer_in_paragraph, 'text': answer}]
            })
        article = {
            'title': d['id'],
            'paragraphs': [{'context': paragraph, 'qas': qas}]
        }
        data.append(article)
    result = {
        'version': '0.1',
        'data': data,
    }
    return result


def match_answer_in_paragraph(answer_text: str, evidence: List[int], passages):
    passage = ' '.join(passages)
    for evidence_idx in evidence:
        evidence_sent = passages[evidence_idx]
        idx_evidence_in_paragraph = passage.index(evidence_sent)
        if answer_text in evidence_sent:
            idx_answer_in_evidence = evidence_sent.index(answer_text)
            idx_answer_in_paragraph = idx_evidence_in_paragraph + idx_answer_in_evidence
            assert passage[idx_answer_in_paragraph:idx_answer_in_paragraph + len(answer_text)] == answer_text
            yield idx_answer_in_paragraph


def from_squad(dataset):
    result = []
    for d in dataset['data']:
        datum = d['paragraphs'][0]
        datum['passage'] = datum['context']
        datum['id'] = d['title']
        for qa in datum['qas']:
            qa['answer'] = qa['answers'][0]['text']
        result.append(datum)
    return result


def to_squad(uuid, event_plans, all_events, template_choices, baseline_stories, mqs, qs, modified_stories,
             control_stories):
    baseline = []
    modified = []
    control = []
    for event_plan, events, template_choices, baseline_story, mqs, qs, story, control_story in \
            zip(event_plans, all_events, template_choices, baseline_stories, mqs, qs, modified_stories,
                control_stories):
        story_id = uuid()
        modified_paragraph = {
            "id": story_id,
            'qas': format_qas(story, event_plan, events, qs, story_id, template_choices),
            "context": ' '.join(story),
            'passage_sents': story
        }
        modified_doc = {'title': modified_paragraph['id'], 'paragraphs': [modified_paragraph]}

        baseline_paragraph = {
            "id": story_id,
            "context": ' '.join(baseline_story),
            'passage_sents': baseline_story,
            'qas': format_qas(baseline_story, event_plan, events, qs, story_id, template_choices)
        }

        baseline_doc = {'title': baseline_paragraph['id'], 'paragraphs': [baseline_paragraph]}

        control_paragraph = {
            "id": story_id,
            "qas": modified_doc['paragraphs'][0]['qas'],
            "context": ' '.join(control_story),
            'passage_sents': control_story
        }

        control_doc = {'title': control_paragraph['id'], 'paragraphs': [control_paragraph]}
        baseline.append(baseline_doc)
        modified.append(modified_doc)
        control.append(control_doc)
    return baseline, modified, control


def format_qas(passages: List[str], event_plan: EventPlan, events, qs, story_id, template_choices):
    qas = []
    for i, q in enumerate(qs):
        qa = {
            "id": f"{story_id}/{i}",
            "question": q.realized,
            "answer": q.answer,
            "reasoning": event_plan.reasoning_type.name,
            'type': q.type,
            'target': q.target,
            'evidence': q.evidence,
            'event_type': q.event_type,
            # 'question_data': q.question_data,
            'ep': event_plan.event_types,
            'events': [asdict(e) for e in events],
            'template_choices': template_choices,
        }
        qa['answers'] = [{
            'answer_start': answer_start_idx,
            'text': qa['answer']
        } for answer_start_idx in match_answer_in_paragraph(q.answer, q.evidence, passages)]
        assert qa['answers']
        qas.append(qa)
    return qas
