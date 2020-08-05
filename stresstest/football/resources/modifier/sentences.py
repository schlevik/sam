sentences = {
    "goal": [
        # 0
        "%CONNECTIVE.VBD> $PP.time $NP.actor , $NP.team.actor-possessive-post , { @RB.goal "
        "$VP.VBD.modifier } [$VP.VBD.shoot in a ($JJ.positive) $NP.goal | $VP.VBD.score] $PP.distance [$PP.goal-cause-coactor|after $S.goal-cause-coactor] .",
        # TODO add more stuff & test
        "$PP.goal-cause-coactor $NP.actor %CONNECTIVE.VP (and $VP.VBD.attention) as !PRP { @RB.goal $VP.VBD.modifier } [$VP.VBD.shoot in a ($JJ.positive) $NP.goal | $VP.VBD.score] $PP.distance "
        "for $NP.team.actor $PP.time .",
        #2
        "$NP.team.actor-possessive player $NP.actor $VP.VBD.attention %CONNECTIVE.ADVP , { @RB.goal $VP.VBG.modifier } "
        "[$VP.VBG.shoot in a ($JJ.positive) $NP.goal | $VP.VBG.score a ($JJ.positive) goal]  $PP.distance [$PP.goal-cause-coactor|(just) after $S.goal-cause-coactor] (following $NP.goal-cause) .",

        # TODO: maybe rephrase the arrived, it's stupid
        "$NP.actor 's $JJ.distance $NP.goal ( , $VP.VBG.goal-effect , ) { @RB.goal $VP.VBD.modifier-nonactor } "
        "arrived $PP.time $PP.goal-cause-coactor (and [$S.attention-crowd|$VP.VBD.attention-crowd]) .",
        #4
        "$PP.time a $NP.pass-type [went to|arrived at] ($NP.team.actor-possessive) $NP.coactor "
        "[$PP.position.vertical | who was (just) waiting $PP.position.vertical] and "
        "$NP.coref-player swept $NP.position.height to the $NP.position.box for $NP.actor to { @RB.goal $VP.VBI.modifier } "
        "poke past the $NP.goalkeeper for a $JJ.positive $JJ.distance $NP.goal .",

        "$PP.time a $JJ.positive $JJ.distance $NP.shot from $NP.actor { @RB.goal $VP.VBG.modifier-nonactor } "
        "[flying $PP.position.goal|homing into $NP.position.goal]  past "
        #                                                THIS IS WEIRD
        "[the $NP.goalkeeper|a helpess $NP.goalkeeper] ($PP.goal-effect) %CONNECTIVE.VP .",
        #6
        "$NP.actor , one of $NP.team.actor-possessive better performers [that day|of the match] , %CONNECTIVE.VP "
        "as !PRP { @RB.goal $VP.VBD.modifier } $VP.VBD.score $PP.time $PP.goal-cause-coactor (and $S.attention-crowd) .",

        "$NP.actor { @RB.goal $VP.VBD.modifier } $VP.VBD.score $PP.time to %CONNECTIVE.IVP when !PRP $VP.VBD.goal-cause "
        "(and $VP.VBD.goal-cause) "
        "before { @RB.goal $VP.VBG.modifier } $VP.VBG.shoot in the ball $PP.position.goal .",
        #8
        "%CONNECTIVE.VBD> $NP.actor { @RB.goal $VP.VBD.modifier } $VP.VBD.score $PP.time , { @RB.goal $VP.VBG.modifier } "
        "[$VP.VBG.shoot in the ball|hit in the ball|slot in] "
        "$PP.distance $PP.position.goal  after !PRP $VP.VBD.goal-cause (and $VP.VBD.goal-cause) .",

        "%CONNECTIVE.VBD> the ball from $NP.coactor arrived [on|at] the $NP.position.box (at pace) and $VP.VBG.goal-effect , $NP.actor "
        "{ @RB.goal $VP.VBD.modifier } $VP.VBD.shoot the ball (just) $PP.position.goal "
        "(to leave the $NP.goalkeeper with no chance) .",
        #10
        "$NP.actor was free on the $NP.position.box %CONNECTIVE.ADVP , and with the defence slow to react, "
        "the $NP.team.actor-possessive player 's $NP.shot { @RB.goal "
        "$VP.VBD.modifier-nonactor } squirmed beyond the $NP.goalkeeper (and $PP.position.goal) for a highlight $PP.time .",

        "%CONNECTIVE.VBD> $PP.goal-cause-coactor , $NP.actor (, on the end of it ,) { @RB.goal $VP.VBD.modifier } $VP.VBD.shoot the ball "
        "$PP.position.goal $VP.VBG.goal-effect .",
        #12
        "$NP.actor { @RB.goal $VP.VBD.modifier } [$VP.VBD.shoot in|$VP.VBD.score] "
        "[$NP.team.actor-possessive !NEXT $NP.goal|the !NEXT $NP.goal for $NP.team.actor] $PP.distance "
        "to %CONNECTIVE.IVP [$PP.goal-cause-coactor|after $S.goal-cause-coactor] .",

        "$NP.team.actor { @RB.goal $VP.VBD.modifier } %CONNECTIVE.VP with a $JJ.distance $NP.goal as $NP.actor { @RB.goal $VP.VBD.modifier } $VP.VBD.shoot in "
        "$NP.coactor 's $NP.pass-type $PP.time . ", 
        #14
        "$NP.actor { @RB.goal $VP.VBD.modifier } %CONNECTIVE.VP $PP.time when !PRP { @RB.goal $VP.VBD.modifier } $VP.VBD.score a $JJ.distance $NP.goal from "
        "$NP.coactor 's $NP.pass-type .",
        
        "$NP.actor { @RB.goal $VP.VBD.modifier } opened the scoring when he { @RB.goal $VP.VBD.modifier } converted a "
        "$NP.kick-type $PP.time with a $NP.shot that glanced off the underside of the crossbar and $PP.position.goal ."
    ],
    "foul": [
        # 0 
        "%CONNECTIVE.VBD> $NP.coactor ($NP.team.nonactor-possessive-post) had gone down with $NP.injury "
        "after a ($JJ.negative) foul by ($NP.team.actor-possessive) $NP.actor .",

        "%CONNECTIVE.VBD> ($NP.team.nonactor-possessive) $NP.coactor $VP.VBD.foul-passive ($RB.neg) by "
        "[$NP.actor (, $NP.team.actor-possessive-post ,)|($NP.team.actor-possessive) $NP.actor] ($PP.time) .",
        # 2
        "%CONNECTIVE.VBD> $NP.actor $VP.VBD.foul ($NP.team.nonactor-possessive) $NP.coactor ($PP.time) .",

        "%CONNECTIVE.VBD> $NP.actor $VP.VBD.foul $NP.coactor ($NP.team.nonactor-possessive-post) "
        "[$PP.foul-cause-coref|and $S.attention-crowd] .",
        # 4
        "[%CONNECTIVE.VBD> $NP.coactor was| %CONNECTIVE.VBG> $NP.coactor being] withdrawn $PP.time with !PRPS $NP.bodypart in a brace following a "
        "($JJ.negative) challenge from $NP.actor .",

        "%CONNECTIVE.VBG> $NP.actor $VP.VBG.foul $NP.coactor [$PP.foul-cause-coref|$PP.foul-effect] .",

        "%CONNECTIVE.VBG> $NP.actor $VP.VBG.foul-effect , "
        "$VP.VBG.foul $NP.coactor near the $NP.position.box .",

        "%CONNECTIVE.VBD> $NP.actor [$VP.VBD.foul-elaboration-coref|$VP.VBD.foul-elaboration] with a ($JJ.negative) "
        "$NP.foul [$PP.foul-effect|$PP.foul-effect-coref|] .",

        "%CONNECTIVE.VBG> $NP.coactor winning the ball [in $NP.position.horizontal| $PP.position.vertical] "
        "(for $NP.team.actor) and drawing a $NP.foul from $NP.actor ."
    ]
}
