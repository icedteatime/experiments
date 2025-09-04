
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

def template(metadata):
    module = metadata["Module"]
    py = f"[py]({metadata[".py"]})"
    ipynb = f"[nb]({metadata[".ipynb"]})" if metadata[".ipynb"] else ""

    model_description = str(module.Model())
    model_lines = model_description.splitlines()
    model_lines = model_lines[1:-1]

    min_space_prefix = min(len(line) - len(line.lstrip())
                           for line in model_lines)
    model_lines = [line[min_space_prefix:]
                   for line in model_lines]

    if len(model_lines) > 20:
        model_lines = model_lines[:18] + ["..."]

    model_description = "\n".join(model_lines)

    return f"""
### {module.name} {ipynb} {py}
{module.description}
```
{model_description}
```
"""

lines = [header] + list(map(template, metadata))
print("\n".join(lines))

with open("README.md", "w") as readme:
    readme.writelines(lines)
