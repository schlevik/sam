from stresstest.graph import generate_path, random_strategy
from stresstest.question import generate_question, generate_answer
from stresstest.stringify import TemplateStringifier
from stresstest.util import load_graph

tpl = """
Passage: {}
Question: {}
Answer: {}
"""


def generate_path_question_and_answer(graph_path, clauses_path):
    g = load_graph(graph_path)
    p = generate_path(g, 'sos', 'eos', random_strategy)
    s = TemplateStringifier(clauses_path)
    q = generate_question(p, s.cfg['question'])
    print(tpl.format(p.stringify(s), q.stringify(s), generate_answer(p, q)))
