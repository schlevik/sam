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


def match_answer_in_paragraph(qa, datum):
    # TODO: rather return all mathches in the whole paragraph
    if len(qa['evidence']) > 1:
        raise NotImplementedError("For now works only with SSQ retrieval type questions!")
    answer = qa['answer']
    evidence_idx = qa['evidence'][0]
    passage = datum['context']
    evidence_sent = datum['passage_sents'][evidence_idx]
    idx_evidence_in_paragraph = passage.index(evidence_sent)
    idx_answer_in_evidence = evidence_sent.index(answer)
    idx_answer_in_paragraph = idx_evidence_in_paragraph + idx_answer_in_evidence
    assert passage[idx_answer_in_paragraph:idx_answer_in_paragraph + len(answer)] == answer
    return idx_answer_in_evidence + idx_evidence_in_paragraph


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
