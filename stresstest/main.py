from typing import List

import networkx as nx

from stresstest.classes import Config
from stresstest.passage.rules import NoFoulTeam, \
    NPlayersMention, UniqueAttribute, GoalWithDistractor, PassageRule
from stresstest.passage.graph import generate_path
from stresstest.passage.strategies import ReasonableStrategy
from stresstest.question.independent_rules import BareMinimum, \
    IsNotModified, QuestionRule
from stresstest.question.question import generate_question
from stresstest.realization.rules import SingularPlural, \
    Modifier
from stresstest.realization.template import TemplateStringifier
from stresstest.util import load_graph


def generate_random(graph: nx.Graph, strategy_rules: List[PassageRule],
                    templates: Config, question_rules: List[QuestionRule],
                    realise=True):
    strategy = ReasonableStrategy(strategy_rules)
    q = None
    p = None
    while not q:
        # generate until we get something reasonable
        p = generate_path(graph, 'start', 'end', strategy)
        q = generate_question(p,
                              templates['question.target'].keys(),
                              templates['question.action'].keys(),
                              question_rules)
    return TemplateStringifier(templates, p, q, realise)


def generate(graph_path='stresstest/resources/unnamed0.graphml',
             clauses_path='stresstest/resources/clauses.conf',
             models=None,
             strategy_rules=None,
             realise=True):
    graph = load_graph(graph_path)
    strategy_rules = [NoFoulTeam(),
                      UniqueAttribute(3)
                      ] + strategy_rules if strategy_rules else []
    templates = Config(clauses_path, [SingularPlural(), Modifier()])
    question_rules = [BareMinimum(), IsNotModified()]
    r = generate_random(graph, strategy_rules, templates, question_rules,
                        realise)
    p = r.path
    if realise:
        print(r)

        if models:
            for model in models:
                print(
                    f"{model.name} prediction: "
                    f"{model.predict(r.question_string, r.passage_string)}"
                )
    return p


def generate_distractor():
    generate(strategy_rules=[GoalWithDistractor()])
