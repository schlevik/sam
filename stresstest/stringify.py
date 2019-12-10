import random
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from aiconf import ConfigReader
from ailog import Loggable
import names

JsonDict = Dict[str, Any]


class Stringifier(Loggable, ABC):
    @abstractmethod
    def to_string(self, path: List[str]):
        ...


def consolidate(path: List[str]) -> List[str]:
    new_path = []
    for node in path:
        if node.startswith("."):
            new_path[-1] = "".join([new_path[-1], node])
        else:
            new_path.append(node)
    return new_path


class TemplateStringifier(Stringifier):
    def __init__(self, templates_path):
        self.cfg: JsonDict = ConfigReader(templates_path).read_config()

    def choice(self, key: str, target: str):
        return random.choice(self.cfg[target][key])

    def resolve_variable(self, string, var_table):
        var_names = re.findall(r"#(\w+)", string)
        for var_name in var_names:
            var_value = var_table.get(var_name, None)
            if not var_value:
                if 'player' in var_name:
                    var_value = names.get_full_name(gender='female')
                elif 'team' in var_name:
                    var_value = random.choice()
                elif 'min' == var_name:
                    var_value = str(random.choice(list(range(1, 90))))
                elif "m" == var_name:
                    var_value = str(random.choice(list(range(5, 30))))
                else:
                    var_value = self.choice(var_name, "variables")
            string = re.sub(f"#{var_name}", var_value, string)
            var_table[var_name] = var_value
        return string, var_table

    def to_string(self, path: List[str]):
        path = consolidate(path)
        variables_table = dict()
        realised_path = [self.choice(c, "path") for c in path]
        resolved_path = []
        for s in realised_path:
            if '#' in s:
                s, variables_table = self.resolve_variable(s, variables_table)
            resolved_path.append(s)

        return resolved_path
