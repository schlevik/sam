import random
import re
from typing import List, Dict

import names
from quicklog import Loggable

from stresstest.classes import Templates, Path
from stresstest.question.question import Question


class TemplateStringifier(Loggable):
    """
    Template based realisation algorithm implementation.

    Transforms logical form based path and question into their final
    string forms.

    Realisation can happen subject to
    :class:`stresstest.classes.rule` s to maintain finer-grained
    control over the random selection of templates.

    Attributes:
        variables_table (Dict[str, str]):
            Map of variables from the path and their corresponding
            values. Populated during path realisation.

        templates (Templates):
            Templates to use for realisation of logical forms.

        path (Path):
            The content of the passage represented as a path in the
            content graph.

        consolidated_path (Path):
            Like ``path`` but with nodes starting with point (``.``)
            (recursively) collapsed with their predecessors. I.e.::

                path: ['sos' 'action', '.goal' '.spectacular', ...]
                consolidated_path: ['sos', 'action.goal.spectacular', ...]

        realised_path (Path):
            The corresponding realisations of ``consolidated_path``
            according to templates and conforming to rules.
            Populated during  the ``to_string_path`` method call.

        resolved_path:
            ``realised_path`` with variable names substituted for their
            values.

        question (Question):
            Question corresponding to the passage.

        index_map: (Dict[int,int]):
            The map of indices from ``path`` to ``consolidated_path``.

    """
    _no_whitespace = ['eos']

    def __init__(self, templates: Templates,
                 path: Path, question: Question):
        """


        Args:
            templates:
            path:
            question:

        """
        self.variables_table: Dict[str, str] = dict()
        self.templates: Templates = templates
        self.path: Path = path
        self.realised_path = []
        self.resolved_path = []
        self.question = question
        # initialise consolidated path and the index map
        self.index_map: Dict[int, int] = dict()
        self.consolidated_path = self._consolidate(self.path)

    def to_string_question(self) -> str:
        """
        Transforms the question into its string form according to
        existing question templates and rules.

        Returns:
            String version of the question.

        """
        target = self.templates \
            .random_with_rules(path=self.path,
                               keys=["question.target",
                                     self.question.target])
        action = self.templates \
            .random_with_rules(path=self.path,
                               keys=["question.action",
                                     self.question.action])
        return " ".join((target, action))

    def to_string_answer(self) -> str:
        """
        Returns the string form of the answer expected by the question.

        Should be called after ``to_string_path``.
        Returns:
            String version of the answer.

        """
        return self.resolved_path[self.index_map[self.question.answer_position]]

    def _resolve_hash_variable(self, string: str, position: int):
        var_names = re.findall(r"#(\w+)", string)
        for var_name in var_names:
            var_value = self.variables_table.get(var_name, None)
            if not var_value:
                if 'player' in var_name:
                    var_value = names.get_full_name(gender='female')
                elif 'team' in var_name:
                    var_value = " ".join([
                        self.templates.random_with_rules(
                            path=self.consolidated_path,
                            realised_path=self.realised_path,
                            keys=['team-name', 'first'],
                            position=position),
                        self.templates.random_with_rules(
                            path=self.consolidated_path,
                            realised_path=self.realised_path,
                            keys=['team-name', 'last'],
                            position=position)
                    ])
                elif 'min' == var_name:
                    # TODO: will need rules here as well
                    var_value = str(random.choice(list(range(1, 45))))
                elif "m" == var_name:
                    # TODO: will need rules here as well
                    var_value = str(random.choice(list(range(5, 30))))
                else:
                    var_value = self.templates. \
                        random_with_rules(path=self.consolidated_path,
                                          realised_path=self.realised_path,
                                          keys=['variables', var_name],
                                          position=position)
            string = re.sub(f"#{var_name}", var_value, string)
            self.variables_table[var_name] = var_value
        return string

    def _consolidate(self, path: Path) -> Path:
        new_path: List[str] = []
        for i, node in enumerate(path):
            if node.startswith("."):
                new_path[-1] = "".join([new_path[-1], node])
            else:
                new_path.append(node)
            self.index_map[i] = len(new_path) - 1
        p = Path(new_path)
        return p

    def _resolve_dollar_template(self, template, position):
        #
        var_names = re.findall(r"\$(\S+)", template)

        for var_name in var_names:
            start = template.index(var_name) - 1
            end = start + len(var_name) + 1
            # Resolve if there is a a fitting template,
            # otherwise just put an empty string
            resolved = \
                self.templates.random_with_rules(
                    path=self.consolidated_path,
                    realised_path=self.realised_path,
                    keys=['path', var_name],
                    position=position
                ) or ""
            self.realised_path[position] = template[
                                           :start] + resolved + template[end:]

    def to_string_path(self) -> str:
        """
        Produces a string realisation (passage) of a given path in the
        content graph.

        Returns:
            String rendition of the selected content (path).

        """
        if self.realised_path:
            raise RuntimeError(f"realised_path not empty! "
                               f"Don't run {self.__class__.__name__} twice!")

        self.logger.debug("Path:")
        self.logger.debug(self.path)
        self.logger.debug("Consolidated path:")
        self.logger.debug(self.consolidated_path)

        # Resolve using path -> string templates
        self.realised_path.extend(
            self.templates.random_with_rules(
                path=self.consolidated_path,
                keys=['path', c], position=i) or ""
            for i, c in enumerate(self.consolidated_path)
        )

        self.logger.debug("Realised path:")
        self.logger.debug(self.realised_path)

        # hierarchical resolution of $-templates
        while "$" in " ".join(self.realised_path):
            for i, s in enumerate(self.realised_path):
                if "$" in s:
                    self._resolve_dollar_template(s, i)

        # Resolution of #-variables
        for i, s in enumerate(self.realised_path):
            if '#' in s:
                s = self._resolve_hash_variable(s, i)
            self.logger.debug(f"Handling {self.consolidated_path[i]}:'{s}'")
            self.resolved_path.append(s.strip())

        self.logger.debug("Resolved Path")
        self.logger.debug(self.resolved_path)
        self.logger.debug('variable table:')
        self.logger.debug(self.variables_table)
        # TODO: normalise space
        result = " ".join(r for r in self.resolved_path if r)
        return result.replace(" . ", ". ")
