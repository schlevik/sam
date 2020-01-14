from stresstest.classes import Templates
from stresstest.passage import conditions
from stresstest.passage.graph import generate_path
from stresstest.passage.strategies import ReasonableStrategy
from stresstest.question.independent_conditions import bare_minimum, \
    is_not_modified
from stresstest.question.question import generate_question
from stresstest.realization.template import TemplateStringifier
from stresstest.util import load_graph

tpl = """
Passage: {}
Question: {}
Answer: {}
"""


def generate_passage_question_and_answer_reasonable(graph_path, clauses_path,
                                                    p=None):
    g = load_graph(graph_path)
    strategy = ReasonableStrategy(conditions.__all__)
    if not p:
        p = generate_path(g, 'start', 'end', strategy)
    templates = Templates(clauses_path, [])
    q = generate_question(p,
                          templates['question.target'].keys(),
                          templates['question.action'].keys(),
                          [bare_minimum, is_not_modified], [])
    s = TemplateStringifier(templates, p, q)
    if q:
        print(tpl.format(s.to_string_path(), s.to_string_question(),
                         s.to_string_answer()))
    else:
        print(tpl.format(
            s.to_string_path(),
            "I can't possibly think of a question to ask!",
            ""))
    return p
