
1. Create virtual env

    `python -m venv .venv/ --prompt rs_example_lib --upgrade-deps`

2. Activate venv

    1. Linux

        `source .venv/bin/activate`

    2. Windows

        `.venv\Scripts\activate`

3. Install dependencies

    `pip install .[dev]`

4. Install editable project

    `pip install --no-build-isolation --config-settings=editable-verbose=true --editable .`
