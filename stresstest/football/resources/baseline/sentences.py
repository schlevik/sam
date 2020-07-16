sentences = {
    "goal": [
        "%CONNECTIVE.VBD> $ACTOR $ACTORTEAM.name-pos-post @RB.goal "
        "$MODIFIER.VBD $VBD.goal a ($JJ.positive) goal ($REASON.PP.goal).",

        "%CONNECTIVE.VBD> $ACTOR @RB.goal $MODIFIER.VBD $VBD.goal a ($JJ.positive) goal for $ACTORTEAM.name .",

        "$ACTORTEAM.name-pos-pre player $ACTOR $VP.attention %CONNECTIVE.ADVP , @RB.goal $MODIFIER.VBG "
        "$VBG.goal a ($JJ.positive) goal $DISTANCE.PP .",

        # TODO: modifier goal arrived
        "$ACTOR 's goal ( , [$RDM.VBG.goal|$RDM.NOVB] , ) @RB.goal $MODIFIER.nonactor.VBD "
        "arrived $TIME after !PRPS teammate $COACTOR 's $PASS-TYPE and [$RDM.CC-V.goal|$RDM.S.goal] .",

        # TODO: MODIFIER VBI or post-processing
        "$TIME a $PASS-TYPE [went to|arrived at] ($ACTORTEAM.name-pos-pre) $COACTOR $POSITION.VERTICAL.PP and "
        "$COREF-PLAYER swept $POSITION.HEIGHT.NN to the $POSITION.BOX for $ACTOR to @RB.goal $MODIFIER.VBD "
        "poke past the $GOALKEEPER .",

        #TODO: modifier goal arrived
        "A $JJ.positive $DISTANCE.JJ strike from $ACTOR @RB.goal $MODIFIER.VBG [flying|homing] into $POSITION.GOAL past "
        "[the $GOALKEEPER|a helpess $GOALKEEPER] ($RDM.PP.goal) %CONNECTIVE.VP .",

        "$ACTOR , one of $ACTORTEAM.name-pos-pre better performers today, %CONNECTIVE.VP "
        "as !PRP @RB.goal $MODIFIER.VBD scored $TIME [$REASON.PP.goal| and $RDM.S.goal] .",

        "$ACTOR @RB.goal $MODIFIER.VBD scored $TIME to %CONNECTIVE.IVP when !PRP $REASON.CC-V.goal (and $REASON.CC-V.goal) "
        "before @RB.goal $MODIFIER.VBG $VBG.goal the ball $POSITION.PP.GOAL .",

        "%CONNECTIVE.VBD> $ACTOR @RB.goal $MODIFIER.VBD scored $TIME , @RB.goal $MODIFIER.VBG $VBG.goal the ball "
        "$POSITION.PP.GOAL after !PRP $REASON.CC-V.goal (and $REASON.CC-V.goal) .",

        "%CONNECTIVE.VBD> the ball arrived [on|at] the $POSITION.BOX (at pace) and [$RDM.VBG.goal] , $ACTOR "
        "@RB.goal $MODIFIER.VBD $VBDO.goal (just) $POSITION.PP.GOAL (to leave the $GOALKEEPER with no chance) .",

        # TODO: modifier goal arrived
        "%CONNECTIVE.VBD> $ACTOR was free on the $POSITION.BOX , and with the defence slow to react, "
        "the $ACTORTEAM.name-pos-pre player 's drive @RB.goal $MODIFIER.VBD squirmed  beyond the $GOALKEEPER .",

        "%CONNECTIVE.VBD> $ACTOR , on the end of it , @RB.goal $MODIFIER.VBD $VBDO.goal $POSITION.GOAL $RDM.VBG.goal .",

        "$ACTOR @RB.goal $MODIFIER.VBD $VBD.goal "
        "[$ACTORTEAM.name-pos-pre !NEXT $NN.goal|the !NEXT $NN.goal for $ACTORTEAM.name] "
        "to %CONNECTIVE.IVP after $REASON.S.goal .",

        "$ACTORTEAM.name %CONNECTIVE.IVP when $ACTOR @RB.goal $MODIFIER.VBD $VBD.goal $COACTOR $PASS-TYPE ."

    ],
    "foul": [
        "%CONNECTIVE.VBD> $COACTOR ($COACTORTEAM.name-pos-post) had gone down with $INJURY "
        "after a ($JJ.negative) foul by ($ACTORTEAM.name-pos-pre) $ACTOR .",

        "%CONNECTIVE.VBD> ($COACTORTEAM.name-pos-pre) $COACTOR $VBD-PASSIVE.foul ($ADVJ.neg) by "
        "[$ACTOR ($ACTORTEAM.name-pos-post)|($COACTORTEAM.name-pos-pre) $ACTOR] ($TIME) .",

        "%CONNECTIVE.VBD> $ACTOR $VBD.foul ($COACTORTEAM.name-pos-pre) $COACTOR ($TIME) .",

        "%CONNECTIVE.VBD> $ACTOR $VBD.foul $COACTOR ($COACTORTEAM.name-pos-post) [$RDM.PP.foul|and $RDM.S.any] .",

        "$RDM.S.any as $COACTOR was withdrawn $TIME with !PRPS $BODYPART in a brace following a ($JJ.negative) "
        "challenge from $ACTOR .",

        "%CONNECTIVE.VBG> $ACTOR $VBG.foul $COACTOR $RDM.PP.foul .",

        "%CONNECTIVE.VBG> $ACTOR $RDM.VBG.foul , "
        "$VBG.foul $COACTOR near the $POSITION.BOX .",

        "%CONNECTIVE.VBD> $ACTOR $RDM.VBD.foul with a ($JJ.negative) $NN.foul ($RDM.PP.foul) .",

        "%CONNECTIVE.VBG> $COACTOR winning the ball [in $POSITION.HORIZONTAL| $POSITION.VERTICAL.PP] "
        "(for $ACTORTEAM.name) and drawing a $NN.foul from $ACTOR ."
        # "If (only) #sent.actor $VBD.goal the penalty, "
        # "the score would be @CONDITION.then, otherwise it would "
        # "stay @CONDITION.else, @CONDITION.value"
    ]
}