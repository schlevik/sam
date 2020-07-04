def squad(dataset):
    data = []
    for d in dataset:
        qas = []
        paragraph = d['passage']
        for qa in d['qas']:
            answer = qa['answer']
            assert len(qa['evidence']) == 1
            evidence_idx = qa['evidence'][0]
            evidence_sent = d['passage_sents'][evidence_idx]
            idx_evidence_in_paragraph = paragraph.index(evidence_sent)
            idx_answer_in_evidence = evidence_sent.index(answer)
            idx_answer_in_paragraph = idx_evidence_in_paragraph + idx_answer_in_evidence
            assert paragraph[idx_answer_in_paragraph:idx_answer_in_paragraph + len(answer)] == answer
            qas.append({
                'question': qa['question'],
                'id': qa['id'],
                'answers': [{'answer_start': idx_answer_in_evidence + idx_evidence_in_paragraph, 'text': answer}]
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
