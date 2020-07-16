import re

from loguru import logger


# percent looks like if/then/else thing


# percent = {
#     # preamble decider for a VBD-led VP
#     "PREAMBLE-VBD": {
#         # at the beginning of the sentence
#         "begin": {
#             "condition": lambda ctx: ctx['sent_nr'] == 0,
#             # put some starting comments
#             True: ["$BEGIN.VBD.matchstart"],
#             # whether to use a contrast or not
#             False: ["%CONTRAST-VBD"]
#         }
#     },
#
#     ,
#
#     "CONTRAST-VBD": {
#         "condition": _is_contrastive,
#         "contrastive": ["$BEGIN.VBD.contrastive"],
#         "supportive": ["$BEGIN.VBD.supportive"],
#         "neutral": ["$BEGIN.VBD.neutral"]
#     }
#
# }


# assuming we only have one action
def _vbx(template, event_type):
    """
    Gives the mode of the action verb (e.g. VBD, VBG, etc)
    """
    logger.debug(event_type)
    logger.debug(template)
    # select action verb. if no action verb, then select any first appearing verb
    vbx = next((x for x in template if x.startswith("$V") and f".{event_type}" in x), None)
    if not vbx:
        logger.debug("Template has no action verb!")
        try:
            vbx = next(x for x in template if ("VB") in x and f".{event_type}" in x)
        except StopIteration:
            first_vbg = next((i for i, x in enumerate(template) if x.endswith('ing')), -1)
            first_vbd = next((i for i, x in enumerate(template) if x.endswith('ed')), -1)
            if first_vbg > 0 and first_vbg > first_vbd:
                return "VBG"
            elif first_vbd > 0 and first_vbd > first_vbg:
                return "VBD"
            else:
                # TODO: hmmm
                return "VBD"
    logger.debug(f"VBX:{vbx}")
    vbx = re.findall(r'(VB.).*\.', vbx)[0]
    logger.debug(f"VBX:{vbx}")
    assert vbx.startswith('VB')
    return vbx


_possible_verb_forms = ("VBG", "VBD")
possible_contrastive = ["contrastive", 'supportive', 'neutral']


def _is_first_event_of_its_type(ctx):
    event_type = ctx['sent'].event_type
    return next(sent.sentence_nr for sent in ctx['sentences'] if sent.event_type == event_type) == ctx['sent_nr']
