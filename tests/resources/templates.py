import random

sentences = {
    "test": [
        "$a $a $a",
        '$b.c $b.c $b.c',
        '$c'
    ],
    "unique_sentence_test": [
        "0",
        "1",
        "2",
        "3",
    ]
}
dollar = {
    "a": "1 2 3".split(),  # flat
    'b': {  # nested
        "c": "1 2 3".split()
    },
    'c': ['$d b $d'],
    'd': ['1', '2']
}

at = {

}

# percent looks like if/then/else thing
percent = {

}

bang = {

}
templates = {

}
