from stresstest.classes import Templates
from stresstest.passage.rules import AtLeastOneSentence, NoFoulTeam, \
    NPlayersMention, UniqueElaborations, GoalWithDistractor
from stresstest.passage.graph import generate_path
from stresstest.passage.strategies import ReasonableStrategy
from stresstest.question.independent_rules import BareMinimum, \
    IsNotModified
from stresstest.question.question import generate_question
from stresstest.realization.rules import SingularPlural, \
    Modifier
from stresstest.realization.template import TemplateStringifier
from stresstest.util import load_graph

tpl = """
Passage: {}
Question: {}
Answer: {}"""


def generate_passage_question_and_answer_reasonable(graph_path, clauses_path,
                                                    p=None, models=None):
    g = load_graph(graph_path)
    strategy = ReasonableStrategy(
        [AtLeastOneSentence(), NoFoulTeam(), NPlayersMention(),
         UniqueElaborations(3)])
    if not p:
        p = generate_path(g, 'start', 'end', strategy)
    templates = Templates(clauses_path,
                          [SingularPlural(), Modifier()])
    q = generate_question(p,
                          templates['question.target'].keys(),
                          templates['question.action'].keys(),
                          [BareMinimum(), IsNotModified()]
                          )
    s = TemplateStringifier(templates, p, q)
    if q:
        passage = s.to_string_path()
        question = s.to_string_question()
        answer = s.to_string_answer()
        print(tpl.format(passage, question, answer))
        if models:
            for name, model in models:
                print(
                    f"{name} prediction: {model.predict(question, passage)['best_span_str']}")

    else:
        print(tpl.format(
            s.to_string_path(),
            "I can't possibly think of a question to ask!",
            ""))
    return p


def generate_passage_question_and_answer_reasonable_with_distractor(graph_path,
                                                                    clauses_path,
                                                                    p=None,
                                                                    models=None):
    g = load_graph(graph_path)
    strategy = ReasonableStrategy(
        [AtLeastOneSentence(), NoFoulTeam(), GoalWithDistractor(),
         UniqueElaborations(3)])
    if not p:
        p = generate_path(g, 'start', 'end', strategy)
    templates = Templates(clauses_path,
                          [SingularPlural(), Modifier()])
    q = generate_question(p,
                          templates['question.target'].keys(),
                          templates['question.action'].keys(),
                          [BareMinimum(), IsNotModified()]
                          )
    s = TemplateStringifier(templates, p, q)
    if q:
        passage = s.to_string_path()
        question = s.to_string_question()
        answer = s.to_string_answer()
        print(tpl.format(passage, question, answer))
        if models:
            for name, model in models:
                print(
                    f"{name} prediction: {model.predict(question, passage)['best_span_str']}")

    else:
        print(tpl.format(
            s.to_string_path(),
            "I can't possibly think of a question to ask!",
            ""))
    return p
