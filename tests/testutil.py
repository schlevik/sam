from stresstest.passage.generate import StoryGenerator
from stresstest.realization.templates import Templates


def interactive_env(path='stresstest/resources/config.conf'):
    from stresstest.classes import Config
    c = Config(path, [])
    g = StoryGenerator(c)
    t = Templates()
    return g, c, t
