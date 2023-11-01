from pathlib import Path
import sys
import click


@click.group()
def main():
    pass

@click.argument(
    "query-set",
    type=click.Path(
        path_type=Path,
        file_okay=True,
        dir_okay=False,
        exists=True,
    ),
    help="File containing natural language descriptions of scenes.",
)


@click.option(
    "--cache-path",
    type=click.Path(
        path_type=Path,
        file_okay=True,
        dir_okay=False,
    ),
    default="/users/ke/Documents/Cal/research/scenicNL/cache.db",
    help="Path to SQLite3 database for caching.",
)

@click.option(
    "--ignore-cache",
    is_flag=True,
    help="Ignore cache and recompute predictions from scratch. THIS IS VERY EXPENSIVE.",
)


def main(
    query_set: Path,
    cache_path: Path,
    ignore_cache: bool
) -> None:
    """
    Generate simulator scenes from natural language descriptions.
    """
    pass


def _launch():
    # to stop Click handling errors
    ctx = main.make_context("main", sys.argv[1:])
    with ctx:
        main.invoke(ctx)

if __name__ == '__main__':
    _launch()