from dataclasses import asdict
from typing import List, Tuple

from loguru import logger

from stresstest.baseline_utils import mask_passage, mask_question
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


def get_token_offsets(answer_text: str, evidence: List[int], passages: List[str]):
    token_offsets = []
    answer_tokens = answer_text.split(" ")
    for i in evidence:
        passage_tokens = passages[i].split(" ")
        for j in range(len(passage_tokens)):
            window = passage_tokens[j:j + len(answer_tokens)]
            if window == answer_tokens:
                token_offsets.append((i, j))
    return token_offsets


def match_answer_in_paragraph(passages: List[str], token_offsets: List[Tuple[int, int]]):
    sent_start_offsets = [0]
    for p in passages[:-1]:
        sent_start_offsets.append(sent_start_offsets[-1] + len(p) + 1)
    logger.debug(f"Sent start offsets: {sent_start_offsets}")
    for sent_idx, token_start_idx in token_offsets:
        cum = 0
        sent_tokens = passages[sent_idx].split(" ")
        res = ...
        for i, t in enumerate(sent_tokens):
            logger.debug(f"sent tokens: {sent_tokens}, token: {sent_tokens[i]}, cum: {cum}")
            if i == token_start_idx:
                res = sent_start_offsets[sent_idx] + cum
                break
            cum += len(t) + 1  # whitespace
        logger.debug(f"Yielding {res}")
        yield res


# for evidence_idx in evidence:
#     evidence_sent = passages[evidence_idx]
#     idx_evidence_in_paragraph = passage.index(evidence_sent)
#     if answer_text in evidence_sent:
#         idx_answer_in_evidence = evidence_sent.index(answer_text)
#         idx_answer_in_paragraph = idx_evidence_in_paragraph + idx_answer_in_evidence
#         assert passage[idx_answer_in_paragraph:idx_answer_in_paragraph + len(answer_text)] == answer_text
#         yield idx_answer_in_paragraph


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
             control_stories, mask_p, mask_q, keep_answer_candidates):
    baseline = []
    modified = []
    control = []
    for event_plan, events, template_choices, baseline_story, mqs, qs, story, control_story in \
            zip(event_plans, all_events, template_choices, baseline_stories, mqs, qs, modified_stories,
                control_stories):
        story_id = uuid()

        mq = mqs[0]
        answer_token_offsets = get_token_offsets(mq.answer, mq.evidence, story)
        assert answer_token_offsets, f"{mq}, {story}"
        if mask_p:
            story = mask_passage(mq, story, events, keep_answer_types=keep_answer_candidates)
            mq.answer = mask_passage(mq, [mq.answer], events, keep_answer_types=keep_answer_candidates)[0]

        modified_paragraph = {
            "id": story_id,
            'qas': format_qas(story, event_plan, events, [mq], story_id, template_choices, mask_q,
                              answer_token_offsets),
            "context": ' '.join(story),
            'passage_sents': story
        }
        modified_doc = {'title': modified_paragraph['id'], 'paragraphs': [modified_paragraph]}

        q = qs[0]
        answer_token_offsets_baseline = get_token_offsets(q.answer, q.evidence, baseline_story)
        assert answer_token_offsets_baseline, f"{q}, {baseline_story}"
        if mask_p:
            baseline_story = mask_passage(q, baseline_story, events, keep_answer_types=keep_answer_candidates)
            q.answer = mask_passage(q, [q.answer], events, keep_answer_types=keep_answer_candidates)[0]
        baseline_paragraph = {
            "id": story_id,
            "context": ' '.join(baseline_story),
            'passage_sents': baseline_story,
            'qas': format_qas(baseline_story, event_plan, events, [q], story_id, template_choices, mask_q,
                              answer_token_offsets_baseline)
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


def format_qas(passages: List[str], event_plan: EventPlan, events, qs, story_id, template_choices, mask_q,
               answer_token_offsets):
    qas = []
    for i, q in enumerate(qs):
        qa = {
            "id": f"{story_id}/{i}",
            "question": q.realized if not mask_q else mask_question(q),
            "answer": q.answer,
            "reasoning": event_plan.reasoning_type.name,
            "num_modifications": event_plan.num_modifications,
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
        } for answer_start_idx in match_answer_in_paragraph(passages, answer_token_offsets)]
        assert qa['answers']
        qas.append(qa)
    return qas
