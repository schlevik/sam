from abc import abstractmethod

from stresstest.classes import Path, Rule, Choices
from stresstest.util import get_sentence_of_word


class QuestionRule(Rule):
    def __call__(self, *, path: Path, choices: Choices, **kwargs) -> Choices:
        target = kwargs.get('target', None)
        action = kwargs.get('action', None)
        missing = None
        if not target:
            missing = 'target'
        if not action:
            missing = 'action'
        if not target and not action:
            missing = 'target and action'
        if missing:
            raise ValueError("QuestionRule signature is (path: Path, "
                             "choices: Choices, target:str, action:str) "
                             f"(keyword only). {missing} is missing.")
        return self.evaluate_rule(path=path, choices=choices,
                                  target=target, action=action)

    @abstractmethod
    def evaluate_rule(self, *, path: Path, target: str, action: str,
                      choices: Choices) -> Choices:
        ...


class BareMinimum(QuestionRule):
    def evaluate_rule(self, target: str, action: str, path: Path,
                      choices: Choices) -> Choices:
        alphnum = path.alph_num()
        result = []
        for i in choices:
            if alphnum[i] == target:
                if (action in alphnum) and \
                        action in alphnum[get_sentence_of_word(i, alphnum)]:
                    result.append(i)

        return Choices(result)


class IsNotModified(QuestionRule):
    def evaluate_rule(self, target: str, action: str, path: Path,
                      choices: Choices) -> Choices:

        alphnum = path.alph_num()

        result = []
        for i in choices:
            sentence = alphnum[get_sentence_of_word(i, alphnum)]
            if "altering" not in sentence:
                result.append(i)

        return Choices(result)
