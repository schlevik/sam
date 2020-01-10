import random
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from aiconf import ConfigReader
from ailog import Loggable
import names

from stresstest.util import alphnum

JsonDict = Dict[str, Any]


class Stringifier(Loggable, ABC):
    @abstractmethod
    def to_string(self, path: List[str]):
        ...

    @abstractmethod
    def to_string_question(self, question):
        ...


def consolidate(path: List[str]) -> Tuple[List[str], Dict[str, int]]:
    new_path: List[str] = []
    index_map = dict()
    for node in path:
        if node.startswith("."):
            new_path[-1] = "".join([new_path[-1], node])
        else:
            new_path.append(node)
        index_map[node] = len(new_path) - 1

    return new_path, index_map


class TemplateStringifier(Stringifier):
    def to_string_question(self, question):
        return " ".join(
            [self.choice("question.answer-type", question.answer_type),
             self.choice("question.question-type", question.question_type)])

    def __init__(self, templates_path):
        self.cfg: JsonDict = ConfigReader(templates_path).read_config()

    def choice(self, target: str, key: str):
        return random.choice(self.cfg[target][key])

    def resolve_variable(self, string, var_table):
        var_names = re.findall(r"#(\w+)", string)
        for var_name in var_names:
            var_value = var_table.get(var_name, None)
            if not var_value:
                if 'player' in var_name:
                    var_value = names.get_full_name(gender='female')
                elif 'team' in var_name:
                    var_value = " ".join(
                        [random.choice(self.cfg['team-name.first']),
                         random.choice(self.cfg['team-name.last'])]
                    )
                elif 'min' == var_name:
                    var_value = str(random.choice(list(range(1, 45))))
                elif "m" == var_name:
                    var_value = str(random.choice(list(range(5, 30))))
                else:
                    var_value = self.choice("variables", var_name)
            string = re.sub(f"#{var_name}", var_value, string)
            var_table[var_name] = var_value
        return string, var_table

    def to_string(self, path: List[str]):
        path, index_map = consolidate(path)
        variables_table: Dict[str, str] = dict()
        realised_path = [self.choice("path", c) for c in path]
        resolved_path = []
        for s in realised_path:
            if '#' in s:
                s, variables_table = self.resolve_variable(s, variables_table)
            resolved_path.append(s.strip())

        return (" ".join(resolved_path),
                {alphnum(n): resolved_path[v] for n, v in index_map.items()})
