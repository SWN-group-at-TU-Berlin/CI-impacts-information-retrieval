#!/usr/bin/env python
"""Write the requirements.txt of the Hugging Face Space from hf_space/pyproject.toml.

Only the direct dependencies, with the version ranges as written in pyproject.toml (no fully pinned lock file):
the Space build installs gradio, spaces, uvicorn, websockets etc. itself with its own pins
(`pip install -r requirements.txt gradio[oauth,mcp]==<sdk_version> spaces==<x> ...`), exact pins of these packages
or of their dependencies would conflict with them. gradio and spaces are therefore left out completely.

usage: python hf_space/make_requirements.py > requirements.txt
"""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent / "pyproject.toml"
INSTALLED_BY_SPACE = {"gradio", "spaces"}  # version of gradio: sdk_version in hf_space/README.md


def package_name(requirement: str) -> str:
    return re.match(r"[A-Za-z0-9_.-]+", requirement).group(0).lower().replace("_", "-")


def requirements(pyproject: Path = PYPROJECT) -> list[str]:
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    return [d for d in dependencies if package_name(d) not in INSTALLED_BY_SPACE]


if __name__ == "__main__":
    print("# generated from hf_space/pyproject.toml by hf_space/make_requirements.py, do not edit")
    print("\n".join(requirements()))
