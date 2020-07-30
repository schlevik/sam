from stresstest.baseline_utils import mask
from stresstest.ds_utils import get_token_offsets, match_answer_in_paragraph


def test_get_token_offsets():
    answer_text: str = 'cake'
    evidence = [0, 1]
    passages = ["I like cake .", "The cake tastes cake . ", "Irrelevant sentence ."]
    expectation = [(0, 2), (1, 1), (1, 3)]
    reality = get_token_offsets(answer_text, evidence, passages)
    assert reality == expectation


def test_match_answer_in_paragraph():
    passages = ["I like cake .", "The cake tastes cake .", "Irrelevant sentence ."]
    token_offsets = [(0, 2), (1, 1), (1, 3)]
    expectation = [7, 18, 30]
    reality = list(match_answer_in_paragraph(passages, token_offsets))
    assert reality == expectation


def test_answer_still_matches_after_mask():
    passages = ["I like cake .", "The cake tastes cake .", "Irrelevant sentence ."]
    answer_text: str = 'cake'
    evidence = [0, 1]
    offsets = get_token_offsets(answer_text, evidence, passages)
    expectation = [(0, 2), (1, 1), (1, 3)]
    assert offsets == expectation
    masked_passages = [mask(p, keep='cake') for p in passages]
    for si, ti in offsets:
        assert masked_passages[si].split(" ")[ti] == 'cake'
    reality = list(match_answer_in_paragraph(masked_passages, offsets))
    for offset in reality:
        assert " ".join(masked_passages)[offset:offset + len("cake")] == "cake"


def test_get_token_offsets_bridge():
    passages = [
        'In the 16 th minute a pin-point cross went to Barbara Ferreira who was just waiting in the centre and she'
        ' swept high to the edge of the area for Jacqueline Arnold to poke past the last line of defence for '
        'a wonderful 16 metres goal .',

        "Pamela Alvarez 's 17 metres goal , following a wonderful juggle , all but arrived in "
        "minute 41 after Mildred Garcia 's pass .",

        "Then Gigantic Duckburg 's Melissa Reinhardt was felled by Mary Franklin .",

        "Gigantic Duckburg advanced the action with a 20 metres goal as Evonne Marana drilled in Jack "
        " 's soft clearance .",

        'Further Van Wolhok , on the end of it , curled the ball in the middle of the goal '
        'drawing attention from everyone around .',

        'On the 71 th minute an amazing 32 metres shot from Denise Crumpton homing into the corner '
        'past a helpess last line of defence constituted a counter strike .'
    ]
    evidence = [0, 3]
    #answer_text =
    #get_token_offsets
