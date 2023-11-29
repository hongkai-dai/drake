"""
Generates the serialize.h header file, containing Clarabel's settings names.
"""

import argparse
from pathlib import Path

from bazel_tools.tools.python.runfiles import runfiles

_PROLOGUE = """\
#pragma once

#include "drake/common/name_value.h"

namespace clarabel {

template <typename Archive>
// NOLINTNEXTLINE(runtime/references)
void Serialize(Archive* a, DefaultSettings<double>& settings) {
#define DRAKE_VISIT(x) a->Visit(drake::MakeNameValue(#x, &(settings.x)))
"""

_EPILOGUE = """\
#undef DRAKE_VISIT
}

}  // namespace clarabel
"""


def _settings_names():
    """Returns the list of names of Clarabel.cpp's settings."""

    # Read the DefaultSettings.h header.
    manifest = runfiles.Create()
    headers_dir = "clarabel_cpp_internal/include/cpp"
    header = manifest.Rlocation(f"{headers_dir}/DefaultSettings.h")
    with open(header) as f:
        text = f.read()

    # Strip away the parts we don't need.
    needle = "struct DefaultSettings\n{"
    index = text.find(needle)
    assert index > 0
    text = text[index + len(needle):]
    needle = "}"
    index = text.find(needle)
    assert index > 0
    text = text[:index]

    # Parse the contents of the struct.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("static"):
            continue
        assert line.endswith(";")
        line = line[:-1]
        assert line.count(" ") == 1
        _, name = line.split()
        yield name


def _create_header_text():
    result = _PROLOGUE
    for name in _settings_names():
        result += f"  DRAKE_VISIT({name});\n"
    result += _EPILOGUE
    return result


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", metavar="FILE", required=True)
    args = parser.parse_args()
    text = _create_header_text()
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)


assert __name__ == "__main__"
_main()
