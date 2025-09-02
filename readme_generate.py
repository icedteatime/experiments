
import os
import re
import importlib

files = [file[:-len(".py")] for file in os.listdir()
         if re.match(r"(_\d\d_.*)\.py", file)]

metadata = [{"Module": importlib.import_module(file),
             ".py": f"{file}.py",
             ".ipynb": f"{file}.ipynb" if os.path.exists(f"{file}.ipynb") else None}
            for file in sorted(files)]

def template(metadata):
    match metadata:
        case {"Module": module, ".py": py, ".ipynb": ipynb}:
            py = f"[py]({py})"
            ipynb = f"[nb]({ipynb})" if ipynb else ""
            model_description = str(module.Model())
            if len(model_description.splitlines()) > 15:
                model_description = "\n".join(model_description.splitlines()[:20]) + "\n..."

            return f"""
### {module.name} {ipynb} {py}
{module.description}
```
{model_description}
```
"""

print("\n".join(map(template, metadata)))
with open("README.md", "w") as readme:
    readme.writelines(map(template, metadata))
