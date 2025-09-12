
import os
import re
import importlib

header = """
# Machine learning experiments
"""

files = [file[:-len(".py")] for file in os.listdir()
         if re.match(r"(_\d\d_.*)\.py", file)]

metadata = [{"Module": importlib.import_module(file),
             ".py": f"{file}.py",
             ".ipynb": f"{file}.ipynb" if os.path.exists(f"{file}.ipynb") else None}
            for file in sorted(files)]

def file_partition(filename):
    """
    Find `# [](block_name)` patterns and their associated code blocks.
    Uses literal code blocks, rather than an abstract syntax tree.
    """

    with open(filename) as f:
        lines = f.read().splitlines()

    annotations = [re.search(r"\[\]\((?P<block_name>.+)\)", line)
                   for line in lines]
    blank_lines = [re.search(r"^s*$", line)
                   for line in lines]

    stringified = "".join((a and "(") or
                          (b and ")") or
                          " "
                          for a, b in zip(annotations, blank_lines))

    blocks = {}
    line_numbers = {}
    for pair in re.finditer(r"\(.*?\)", stringified):
        block_start, block_end = pair.span()
        block_name = annotations[block_start]["block_name"]
        block_lines = lines[block_start+1:block_end-1]
        block_lines = deindent_lines(block_lines)

        blocks[block_name] = "\n".join(block_lines)
        line_numbers[block_name] = (block_start+1+1, block_end-2+1)

    return blocks, line_numbers

def deindent_lines(text):
    min_space_prefix = min(len(line) - len(line.lstrip())
                           for line in text)
    text = [line[min_space_prefix:]
            for line in text]

    return text

def ellipsize_lines(lines):
    if len(lines) > 15:
        lines = lines[:13] + ["..."]

    return lines

def template(metadata):
    module = metadata["Module"]
    py = f"[py]({metadata[".py"]})"
    ipynb = f"[nb]({metadata[".ipynb"]})" if metadata[".ipynb"] else ""

    model_description = str(module.Model())
    model_lines = model_description.splitlines()
    model_lines = model_lines[1:-1]

    model_lines = deindent_lines(model_lines)
    model_lines = ellipsize_lines(model_lines)
    model_description = "\n".join(model_lines)

    module_description = module.description
    if callable(module.description):
        partitions, _ = file_partition(metadata[".py"])
        module_description = module.description(partitions)

    # set python link as first block if there is one
    for block_start, block_end in (file_partition(metadata[".py"])[1] or {}).values():
        py = f"[py]({metadata[".py"]}#L{block_start})"
        break

    return f"""
### {module.name} {ipynb} {py}
{module_description}
```
{model_description}
```
"""

lines = [header] + list(map(template, metadata))
print("\n".join(lines))

with open("README.md", "w") as readme:
    readme.writelines(lines)
