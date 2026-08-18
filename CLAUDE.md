## Everywhere

Use a concise and precise language with accurate 
and descriptive words. Use ASD-STE100 style.

If a ``CLAUDE.md`` file exists in a directory, its directives and instructions will precedes current one when applying to all its files and subdirectories.

## Responses

Use ASD-STE100.

Avoid verbose language and lengthy sentences. Keep your responses concise and 
precise, with accurate and descriptive words.

## Codes

Always use short but descriptive names, either in elaborate or 
minimalistic coding styles.

When commenting a code, keep the comments concise and precise with accurate 
and descriptive words. Use ASD-STE100.

## Documentation

In the documents, keep a catalog of names of entities such as method, policy, schedule, etc. For each entity, provide an elaborate description and what it is and/or how it works, along with a code snippet if applicable. These entities can be things suchas augmentation policy, gradient optimizer function, gradient update schedule (row-wise, cell-wise, etc), force function, normalization method, etc. Provenance: Include links to the corresponding code block or file. 

The catalog is an incrementing document. Do not remove anything. For update, mention the reason for update and the include the updated version. Also include at which version of the code development or experiment or research step the update is adopted.

## Python Environment

This project's virtual environment lives at `.venv/`. 
Invoke its interpreter directly. Do not run `source .venv/bin/activate` as it does not persist between tool calls.

Use these exact commands:
- Run a script: `.venv/bin/python script.py`
- Install a package after granted permission: `.venv/bin/pip install <package>`
- Run a tool: `.venv/bin/pytest`

To store large files and artifacts use `/tmp/` for temporary ones, and `data_cache/` for others. These files can be `.npz` and other files. You may use a symlink to link to them locally.