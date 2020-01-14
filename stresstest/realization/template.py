import random
import re
from typing import List, Dict, Tuple

import names

from stresstest.classes import Stringifier, Templates, Path
from stresstest.question.question import Question
from stresstest.util import alphnum


class TemplateStringifier(Stringifier):
    def to_string_question(self):
        target = self.templates \
            .random_with_conditions(self.path,
                                    keys=["question.target",
                                          self.question.target])
        action = self.templates \
            .random_with_conditions(self.path,
                                    keys=["question.action",
                                          self.question.action])
        return " ".join((target, action))

    def to_string_answer(self):
        return self.atomic_to_surface[self.path[self.question.answer_position]]

    def __init__(self, templates: Templates,
                 path: Path, question: Question):
        self.variables_table: Dict[str, str] = dict()
        self.templates = templates
        self.path = path
        self.consolidated_path = None
        self.realised_path = None
        self.resolved_path = None
        self.question = question
        self.consolidated_to_surface: Dict[str, str] = dict()
        self.atomic_to_surface: Dict[str, str] = dict()
        self.index_map: Dict[int, int] = dict()

    def resolve_variable(self, string):
        var_names = re.findall(r"#(\w+)", string)
        for var_name in var_names:
            var_value = self.variables_table.get(var_name, None)
            if not var_value:
                if 'player' in var_name:
                    var_value = names.get_full_name(gender='female')
                elif 'team' in var_name:
                    var_value = " ".join([
                        self.templates.random_with_conditions(
                            self.path, keys=['team-name', 'first']),
                        self.templates.random_with_conditions(
                            self.path, keys=['team-name', 'last'])
                    ])
                elif 'min' == var_name:
                    # TODO: will need conditions here as well
                    var_value = str(random.choice(list(range(1, 45))))
                elif "m" == var_name:
                    # TODO: will need conditions here as well
                    var_value = str(random.choice(list(range(5, 30))))
                else:
                    var_value = self.templates. \
                        random_with_conditions(self.path,
                                               keys=["variables", var_name])
            string = re.sub(f"#{var_name}", var_value, string)
            self.variables_table[var_name] = var_value
        return string

    def consolidate(self, path: Path) -> Path:
        new_path: List[str] = []
        for i, node in enumerate(path):
            if node.startswith("."):
                new_path[-1] = "".join([new_path[-1], node])
            else:
                new_path.append(node)
            self.index_map[i] = len(new_path) - 1
        p = Path()
        p.steps = new_path
        return p

    def to_string_path(self):
        # TODO: Implement conditions here as well
        self.consolidated_path = self.consolidate(self.path)
        self.realised_path = []
        self.logger.info(self.path)
        self.logger.info(self.consolidated_path)

        self.realised_path.extend(self.templates.random_with_conditions(
            self.consolidated_path,
            keys=['path', c]) for c in self.consolidated_path)

        self.resolved_path = []
        self.logger.info("Realised path:")
        self.logger.info(self.realised_path)
        # recursive resolution of templates
        while "$" in " ".join(self.realised_path):
            for i, s in enumerate(self.realised_path):
                if "$" in s:
                    var_names = re.findall(r"\$(\S+)", s)
                    for var_name in var_names:
                        start = s.index(var_name) - 1
                        end = start + len(var_name) + 1
                        resolved = self.templates. \
                            random_with_conditions(self.realised_path,
                                                   keys=['path', var_name])
                        self.logger.info(resolved)
                        self.realised_path[i] = s[:start] + resolved + s[end:]
            self.logger.info(self.realised_path)
        for i, s in enumerate(self.realised_path):
            # if '#' in s:
            #    s = self.resolve_variable(s)

            self.consolidated_to_surface[self.consolidated_path[i]] = s
            for j in (k for k, v in self.index_map.items() if v == i):
                self.atomic_to_surface[self.path[j]] = s

            self.resolved_path.append(s.strip())
        # TODO: normalise space
        return " ".join(self.resolved_path)
