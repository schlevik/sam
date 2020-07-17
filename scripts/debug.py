import click

from scripts.utils import Domain
from stresstest.classes import Bundle
from stresstest.realize import Realizer
from tests.util import interactive_env_football_modifier, only


@click.command()
@click.argument("action", type=str, default=None)
@click.argument("sent-nr", type=int, default=None)
@click.option("--domain", type=Domain(), default='football')
@click.option("--config", default='conf/modifier.json')
@click.option("--n", default=1, type=int)
@click.option("--first-modification", type=int, default=0)
@click.option("--fill-with-modification", type=bool, default=None)
@click.option("--modify-event-types", type=str, default=None)
@click.option("--modifier-type", type=str, default=None)
@click.option("--modification-distance", type=int, default=1)
@click.option("--total-modifiable-actions", type=int, default=2)
def test_modifier(sent_nr, action, n, domain, config, first_modification, fill_with_modification, modify_event_types,
                  modifier_type, modification_distance, total_modifiable_actions):
    if modify_event_types:
        modify_event_types = [modify_event_types]
    if sent_nr is not None and action is not None:
        domain = only(domain, sent_nr, action)
    for i in range(n):
        _ = interactive_env_football_modifier(changed_bundle=domain, cfg=config, do_print=True, do_realise=True,
                                              first_modification=first_modification,
                                              fill_with_modification=fill_with_modification,
                                              modify_event_types=modify_event_types,
                                              modification_distance=modification_distance,
                                              total_modifiable_actions=total_modifiable_actions,
                                              modifier_type=modifier_type)


@click.command()
@click.option("--domain", type=Domain(), default='football')
def validate(domain: Bundle):
    print("Validating...")
    try:
        realizer = Realizer(**domain.templates_modifier, validate=True)
    except ValueError as e:
        print(str(e))
        click.secho("FAIL!", fg='red')
        return
    click.secho("SUCCESS!", fg='green')
