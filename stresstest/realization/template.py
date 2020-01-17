import random
import re
from typing import List, Dict

import names

from stresstest.classes import Stringifier, Templates, Path
from stresstest.question.question import Question


class TemplateStringifier(Stringifier):

    def to_string_question(self):
        """
        Transfers the question into its string form.

        Returns:

        """
        target = self.templates \
            .random_with_conditions(path=self.path,
                                    keys=["question.target",
                                          self.question.target])
        action = self.templates \
            .random_with_conditions(path=self.path,
                                    keys=["question.action",
                                          self.question.action])
        return " ".join((target, action))

    def to_string_answer(self):
        return self.atomic_to_surface(self.question.answer_position)

    def __init__(self, templates: Templates,
                 path: Path, question: Question):
        """

        Args:
            templates:
            path:
            question:

        """
        self.variables_table: Dict[str, str] = dict()
        self.templates = templates
        self._consolidated_path = None
        self.path = path
        self.realised_path = None
        self.resolved_path = None
        self.question = question
        self.index_map: Dict[int, int] = dict()

    def consolidated_to_surface(self, i) -> str:
        return self.resolved_path[i]

    def atomic_to_surface(self, i):
        return self.resolved_path[self.index_map[i]]

    @property
    def consolidated_path(self):
        if not self._consolidated_path:
            self._consolidated_path = self.consolidate(self.path)
        return self._consolidated_path

    def resolve_variable(self, string, position):
        var_names = re.findall(r"#(\w+)", string)
        for var_name in var_names:
            var_value = self.variables_table.get(var_name, None)
            if not var_value:
                if 'player' in var_name:
                    var_value = names.get_full_name(gender='female')
                elif 'team' in var_name:
                    var_value = " ".join([
                        self.templates.random_with_conditions(
                            path=self.consolidated_path,
                            realised_path=self.realised_path,
                            keys=['team-name', 'first'],
                            position=position),
                        self.templates.random_with_conditions(
                            path=self.consolidated_path,
                            realised_path=self.realised_path,
                            keys=['team-name', 'last'],
                            position=position)
                    ])
                elif 'min' == var_name:
                    # TODO: will need conditions here as well
                    var_value = str(random.choice(list(range(1, 45))))
                elif "m" == var_name:
                    # TODO: will need conditions here as well
                    var_value = str(random.choice(list(range(5, 30))))
                else:
                    var_value = self.templates. \
                        random_with_conditions(path=self.consolidated_path,
                                               realised_path=self.realised_path,
                                               keys=['variables', var_name],
                                               position=position)
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
        self.logger.info(self.index_map)
        return p

    def to_string_path(self):
        self.logger.debug("Path:")
        self.logger.debug(self.path)
        self.logger.debug("Consolidated path:")
        self.logger.debug(self.consolidated_path)

        self.realised_path = []
        self.realised_path.extend(
            self.templates.random_with_conditions(
                path=self.consolidated_path,
                keys=['path', c], position=i)
            for i, c in enumerate(self.consolidated_path)
        )

        self.resolved_path = []
        self.logger.debug("Realised path:")
        self.logger.debug(self.realised_path)
        # recursive resolution of templates
        while "$" in " ".join(self.realised_path):
            for i, s in enumerate(self.realised_path):
                if "$" in s:
                    var_names = re.findall(r"\$(\S+)", s)
                    for var_name in var_names:
                        start = s.index(var_name) - 1
                        end = start + len(var_name) + 1
                        # Resolve if there is a a fitting template,
                        # otherwise just put an empty string
                        resolved = \
                            self.templates.random_with_conditions(
                                path=self.consolidated_path,
                                realised_path=self.realised_path,
                                keys=['path', var_name],
                                position=i
                            ) or ""
                        self.realised_path[i] = s[:start] + resolved + s[end:]
        for i, s in enumerate(self.realised_path):
            if '#' in s:
                s = self.resolve_variable(s, i)
            self.logger.debug(f"Handling {self.consolidated_path[i]}:'{s}'")
            all_of_them = [k for k, v in self.index_map.items() if v == i]
            self.logger.debug(all_of_them)
            self.resolved_path.append(s.strip())
        self.logger.debug("Resolved Path")
        self.logger.debug(self.resolved_path)
        self.logger.debug('variable table:')
        self.logger.debug(self.variables_table)
        # TODO: normalise space
        return " ".join(self.resolved_path)
