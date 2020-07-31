import glob
import importlib
import smtplib
import traceback
from email.message import EmailMessage

import click
from click import Command, Context
from loguru import logger
from tqdm import tqdm


class MyGroup(click.Group):
    def invoke(self, ctx):
        ctx.obj = tuple(ctx.args)
        super(MyGroup, self).invoke(ctx)


class Cli:
    notify = False
    ctx: Context = None

    @staticmethod
    @click.group(cls=MyGroup)
    @click.option('--debug', default=False, is_flag=True)
    @click.option('--notify', default=None, type=str)
    @click.pass_context
    def cli(ctx, debug, notify):
        Cli.notify = notify
        Cli.ctx = ctx
        if not debug:
            logger.remove(0)
            logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level='WARNING')
        else:
            logger.remove()
            logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
            logger.add("./logs/debug.log", level='DEBUG', rotation='50MB', compression="zip")
            logger.debug('Set up logging.')

    def __call__(self, *args, **kwargs):
        for f in glob.glob('scripts/*.py'):
            # self.cli: Group
            # for folder in scripts: import everything that is a command, add to main cli
            # python magic 🤪 (or more like undocumented interfaces)
            m = importlib.import_module(f[:-3].replace('/', '.'))
            for name, obj in m.__dict__.items():
                if isinstance(obj, Command):
                    self.cli.add_command(obj)

        content = f"succeeded!"
        try:
            self.cli()
        except Exception as e:
            content = f"failed! " \
                      f"error:\n{traceback.format_exc()}"
            click.secho("YOU FAIL", bold=True, fg='red')
            click.echo(content)
            raise e
        finally:
            cmd = f'"python main.py {self.ctx.invoked_subcommand} {" ".join(self.ctx.obj)}"'
            click.secho(f"Command {cmd} {content}", fg='green')
            if self.notify:
                self.send_email(content=content)

    def send_email(self, content, success=False):
        msg = EmailMessage()
        msg.set_content("content")
        msg['Subject'] = f'Your job {"failed" if not success else "succeeded"}!'
        msg['From'] = 'Robobert'
        msg['To'] = self.notify
        msg.set_content(content)
        with smtplib.SMTP('localhost') as s:
            s.send_message(msg)


if __name__ == '__main__':
    Cli()()
