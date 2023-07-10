# DeepCrunch Docs

We are using Sphinx with Napoleon extension and PyData theme.
Moreover, we set Google style to follow with type convention.

- [Napoleon formatting with Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [ReStructured Text (reST)](https://docs.pylonsproject.org/projects/docs-style-guide/)
- [Paragraph-level markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#paragraphs)

See following short example of a sample function taking one position string and optional

```python
from typing import Optional


def add_numbers(value1: int, value2: Optional[float] = None) -> str:
    """
    A function to add two numbers.

    Parameters:
        value1 (int): The primary number.
        value2 (float, optional): An additional number, defaults to None.

    Returns:
        str: The string representation of the sum of the two numbers.

    Example:
        >>> add_numbers(3, 4)
        '7'

    Remarks:
        Add any special cases or additional details here.
    """
    if value2 is None:
        value2 = 0
    return str(value1 + value2)
```

## Building Docs

When updating the docs, make sure to build them first locally and visually inspect the html files in your browser for
formatting errors. In certain cases, a missing blank line or a wrong indent can lead to a broken layout.
Run these commands

```bash
git submodule update --init --recursive
make docs
```

and open `docs/build/html/index.html` in your browser.

When you send a PR the continuous integration will run tests and build the docs.

Notes:

- You need to have LaTeX installed for rendering math equations. You can for example install TeXLive with the necessary extras by doing one of the following:
  - on Ubuntu (Linux) run `sudo apt-get update && sudo apt-get install -y texlive-latex-extra dvipng texlive-pictures`
  - use the [RTD docker image](https://hub.docker.com/r/readthedocs/build)
- You need to have pandoc installed for rendering Jupyter Notebooks. On Ubuntu (Linux), you can run: `sudo apt-get install pandoc`

## Developing docs

When developing the docs, building docs can be VERY slow locally because of the notebook tutorials.
To speed this up, enable this flag in before building docs:

```bash
# builds notebooks which is slow
export PL_FAST_DOCS_DEV=0

# fast notebook build which is fast
export PL_FAST_DOCS_DEV=1
```

## Customizing the Documentation Theme

In case you want to alter the CSS theme of the documentation, navigate to [this link](https://pydata-sphinx-theme.readthedocs.io/). Please be advised, it's a slightly complex task and requires a basic understanding of JavaScript and npm.
