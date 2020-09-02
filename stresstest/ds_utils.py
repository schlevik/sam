import re
import string
from dataclasses import asdict
from itertools import count
from typing import List, Tuple

from loguru import logger
from tqdm import tqdm

from stresstest.baseline_utils import mask_passage, mask_question
from stresstest.classes import EventPlan, Event


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


def format_qas(passages: List[str], event_plan: EventPlan, events: List[Event], qs, story_id, template_choices, mask_q,
               answer_token_offsets):
    modifier_type = [e.features[0] for e in events if e.features]
    assert all(modifier_type[0] == m for m in modifier_type)
    modifier_type = modifier_type[0]
    qas = []
    for i, q in enumerate(qs):
        qa = {
            "id": f"{story_id}/{i}",
            "question": q.realized if not mask_q else mask_question(q),
            "answer": q.answer,
            "reasoning": event_plan.reasoning_type.name,
            "num_modifications": event_plan.num_modifications,
            "modifier_type": modifier_type,
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


def find_all(haystack, needle):
    """Yields all the positions of the pattern needle in the string haystack."""
    i = haystack.find(needle)
    while i != -1:
        yield i
        i = haystack.find(needle, i + 1)


def find_all_relaxed(haystack, needle):
    """Yields all the positions of the pattern p in the string s."""
    needle = needle.lower()
    if '-' not in needle:
        haystack = haystack.replace('-', ' ')
    haystack = haystack.lower()
    for i in range(len(haystack)):
        if haystack[i:i + len(needle)] == needle:
            yield i, i + len(needle)
    # while i != -1:
    #     yield i
    #     i = haystack.find(needle, i + 1)


def filter_hotpotqa(d):
    paragraphs = d['data'][0]['paragraphs']
    paragraphs_filtered = []
    for p in paragraphs:
        answers = p['qas'][0]['answers']
        if answers:
            # normalize context whitespace
            context = " ".join(p['context'].split())
            p['context'] = context
            answer_text = answers[0]['text']
            # answer should be in context...
            if answer_text in context:
                for answer in answers:
                    # should be the same as HQA only comes with 1 answer
                    assert answer['text'] == answer_text
                p['qas'][0]['answers'] = [{"text": answer_text, 'answer_start': i} for i in
                                          find_all(context, answer_text)]
                assert p['qas'][0]['answers']
                paragraphs_filtered.append(p)
    d['data'][0]['paragraphs'] = paragraphs_filtered
    logger.info(f"Discarded {len(paragraphs) - len(paragraphs_filtered)} examples...")
    return d


def filter_wikihop(d):
    paragraphs = d['data'][0]['paragraphs']
    paragraphs_filtered = []
    for p in paragraphs:
        answers = p['qas'][0]['answers']
        if answers:
            # normalize context whitespace
            # context = " ".join(p['context'].split())
            context = p['context']

            p['context'] = context
            # context = context.lower()
            # answer_text = answers[0]['text']
            p['qas'][0]['question'] = " ".join(p['qas'][0]['question'].split("_"))
            # answer should be in context...
            new_answers = []
            for answer in answers:
                start = answer['answer_start']
                text = answer['text']
                answer_in_context = context[start:start + len(text)]

                if answer_in_context.lower() == text:
                    new_answers.append({"text": answer_in_context, 'answer_start': start})

            if new_answers:
                p['qas'][0]['answers'] = new_answers
                paragraphs_filtered.append(p)
            # assert answer_text in context
            # if answer_text in context:
            #     for answer in answers:
            #         # should be the same as Wikihop only comes with 1 answer
            #         assert answer['text'] == answer_text
            #
            #     assert p['qas'][0]['answers']
            #     paragraphs_filtered.append(p)
    d['data'][0]['paragraphs'] = paragraphs_filtered
    logger.info(f"Discarded {len(paragraphs) - len(paragraphs_filtered)} examples...")
    return d


def filter_drop(d):
    paragraphs = d['data'][0]['paragraphs']
    num_qas_before_filtering = sum(len(d['qas']) for d in paragraphs)
    paragraphs_filtered = []
    for i, p in enumerate(paragraphs):
        context = " ".join(p['context'].split())
        new_p = {'context': context, 'qas': []}
        for qa in p['qas']:
            answers = qa['answers']
            if answers:
                new_answers = []
                for answer in answers:
                    text = " ".join(answer['text'].split())
                    new_answers.extend(
                        [{"text": context[s:e], 'answer_start': s} for s, e in find_all_relaxed(context, text)])
                new_answers = [dict(y) for y in set(tuple(ans.items()) for ans in new_answers)]
                for answer in new_answers:
                    text = answer['text']
                    start = answer['answer_start']
                    assert text == context[start:start + len(text)]
                if new_answers:
                    qa['answers'] = new_answers
                    new_p['qas'].append(qa)
        if new_p['qas']:
            paragraphs_filtered.append(new_p)

    d['data'][0]['paragraphs'] = paragraphs_filtered

    logger.info(f"Discarded {num_qas_before_filtering - sum(len(d['qas']) for d in d['data'][0]['paragraphs'])} "
                f"of {num_qas_before_filtering} examples...")
    return d


regex = re.compile(r"^\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s, do_remove_punc=True):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.replace('\\', '')

    def remove_articles(text):
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())
        # return text

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
        # return text

    def lower(text):
        return text.lower()

    if do_remove_punc:
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    else:
        return white_space_fix(remove_articles(lower(s)))


def filter_searchqa(d, max_c=0):
    delim = '[title]'
    paragraphs = d['data'][0]['paragraphs']
    paragraphs_filtered = []
    no_ans = 0
    for i, p in enumerate(tqdm(paragraphs)):
        answers = p['qas'][0]['answers']
        if answers:
            context = p['context']
            assert delim in context, p
            if max_c:
                context = context[:list(find_all(context, delim))[:max_c + 1][-1]]
            p['context'] = context
            assert len(p['qas']) == 1
            assert all(p['qas'][0]['answers'][0]['text'] == ans['text'] for ans in p['qas'][0]['answers']), '\n'.join(
                str(a) for a in p['qas'][0]['answers'])
            answer_text = normalize_answer(answers[0]['text'])
            if answer_text not in p['context'].lower():
                answer_text = normalize_answer(answers[0]['text'], do_remove_punc=False)
            # assert answer_text in p['context'].lower(), i
            # assert answer_text in context, f"{i}"
            # if answer_text in context:
            p['qas'][0]['answers'] = [{"text": context[i:i + len(answer_text)], 'answer_start': i} for i in
                                      find_all(context.lower(), answer_text)]
            if p['qas'][0]['answers']:
                for answer in p['qas'][0]['answers']:
                    assert p['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])] == answer[
                        'text']
                paragraphs_filtered.append(p)
        else:
            no_ans += 1

    d['data'][0]['paragraphs'] = paragraphs_filtered
    logger.info(f"Discarded {len(paragraphs) - len(paragraphs_filtered)} of {len(paragraphs)} examples... "
                f"({no_ans} had no answer)")
    return d


def filter_newsqa(d, skip_empty=False):
    paragraphs = d['data'][0]['paragraphs']
    num_qas_before_filtering = sum(len(d['qas']) for d in paragraphs)
    paragraphs_filtered = []
    for i, p in enumerate(paragraphs):
        context = p['context']
        new_p = {'context': context, 'qas': []}
        for qa in p['qas']:
            answers = qa['answers']
            if answers:
                assert len(answers) == 1
                answer = answers[0]
                answer_text = answer['text']
                if skip_empty and not answer_text:
                    # skip empty answer
                    continue
                assert answer_text in context
                # if answer_text in context:
                # for answer in answers:
                #     # should be the same as HQA only comes with 1 answer
                #     assert answer['text'] == answer_text
                qa['answers'] = [{"text": answer_text, 'answer_start': i} for i in find_all(context, answer_text)]
                assert qa['answers']
                text = qa['answers'][0]['text']
                start = qa['answers'][0]['answer_start']
                assert text == new_p['context'][start:start + len(text)]
                # paragraphs_filtered.append(p)
                new_p['qas'].append(qa)
        if new_p['qas']:
            paragraphs_filtered.append(new_p)
    d['data'][0]['paragraphs'] = paragraphs_filtered

    logger.info(f"Discarded {num_qas_before_filtering - sum(len(d['qas']) for d in d['data'][0]['paragraphs'])} "
                f"of {num_qas_before_filtering} examples...")
    return d


def filter_generic(d, predicate):
    documents = d['data']
    try:
        num_qas_before_filtering = sum(len(p['qas']) for doc in documents for p in doc['paragraphs'])
    except:
        num_qas_before_filtering = sum(len(p['qas']) for doc in documents for _, p in doc['paragraphs'])
    for document in documents:
        paragraphs = document['paragraphs']
        paragraphs_filtered = []
        for p in enumerate(paragraphs):
            if isinstance(p, tuple):
                p = p[1]
            if predicate(p):
                paragraphs_filtered.append(p)
        document['paragraphs'] = paragraphs_filtered
    try:
        logger.info(
            f"Discarded {num_qas_before_filtering - sum(len(p['qas']) for doc in documents for p in doc['paragraphs'])} "
            f"of {num_qas_before_filtering} examples...")
    except:
        logger.info(
            f"Discarded {num_qas_before_filtering - sum(len(p['qas']) for doc in documents for _, p in doc['paragraphs'])} "
            f"of {num_qas_before_filtering} examples...")
    d['data'] = documents
    return d


def export_brat_format(d, highlights=None) -> List[Tuple[str, str]]:
    texts = []
    annotations = []
    for document in d['data']:
        paragraphs = document['paragraphs']
        for p in enumerate(paragraphs):
            if isinstance(p, tuple):
                p = p[1]
            text = ["Paragraph:"]
            text.append("")
            text.append(p['context'])
            text.append("")
            all_answer_texts = []
            for qa in p['qas']:
                text.append("Question")
                text.append(qa['question'])
                text.append("Answer(s):")
                answer_texts = set(a['text'] for a in qa['answers'])
                all_answer_texts.extend(answer_texts)
                text.append(", ".join(answer_texts))
                text.append("")
            final_text = "\n".join(text)
            texts.append(final_text)
            annotation = []
            i = count(1)
            for answer_text in set(all_answer_texts):
                for start in find_all(final_text, answer_text):
                    annotation.append(f"T{next(i)}\tAnswer {start} {start + len(answer_text)}\t{answer_text}")
            for category, keywords in highlights.items():
                for keyword in keywords:
                    for start in find_all(final_text, keyword):
                        annotation.append(f"T{next(i)}\t{category} {start} {start + len(keyword)}\t{keyword}")
            annotations.append('\n'.join(annotation))
    return list(zip(texts, annotations))
