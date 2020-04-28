import click


@click.command()
@click.option("--input", type=str, default='data/stresstest.json')
@click.option("--output", type=str, default="data/predictions.json")
@click.option("--model", type=str, default='models/model.tar.gz')
@click.option("--cls", type=str, default=None)
def predict(input, output, model, cls):
    raise NotImplementedError()
