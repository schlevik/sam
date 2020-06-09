from stresstest.generate import StoryGenerator
from stresstest.classes import Config


class OneGoalStoryGenerator(StoryGenerator):
    def __init__(self, config: Config):

        super().__init__(config)
        self.goal_scored = lambda: any(
            s.event_type == 'goal' for s in self.sentences)

    def set_action(self):
        actions = self.EVENT_TYPES
        # if last sentence, must be goal
        if self.goal_scored():
            actions = actions - 'goal'
        elif self.world['num_sentences'] - 1 == self.current_event.sentence_nr:
            actions = actions - 'foul'

        self.current_event.event_type = actions.random()
