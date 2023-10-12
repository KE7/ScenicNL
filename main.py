import sys
import click


@click.group()
def main():
    pass

def _launch():
    # to stop Click handling errors
    ctx = main.make_context("main", sys.argv[1:])
    with ctx:
        main.invoke(ctx)

if __name__ == '__main__':
    _launch()