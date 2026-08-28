All agents and subagents must be informed about this file and read it.

## Everywhere

Stay away from over engineering or over-doing. Only deliver what you asked or instructed to do. Add details and other functions ONLY AND ONLY if they help the delivery and if there is a very high chance it is needed. If you want to add extra features, brief the user about it and ask for permission.

Use a concise and precise language with accurate and descriptive words.
Avoid verbosity. 

Use accurate and descriptive words to generate short but information dense sentenses.

If a ``CLAUDE.md`` file exists in a directory, its directives and instructions will precedes current one when applying to all its files and subdirectories.

Use ``/plain-english`` skill, found at ``~/.claude/skills/plain-english/SKILL.md``.

To obtain date for timestamping, etc, always use the system date program ``date``.

## SKILL

For all large tasks use the ``/agentic-development`` skill. These tasks can be such as coding and monitoring more than a medium sized file, or ingesting and processing a large knowledge base and documents. Define the goal and objectives and have the subagents do it.

For coding and text generation tasks, make sure that you don't generate sloppy outputs. Use ``/unslop`` skill for this.

For design tasks use ``/design-restraint`` skill.

## Responses

Use ASD-STE100.

Avoid verbose language and lengthy sentences. Keep your responses concise and 
precise, with accurate and descriptive words. Generate short but information dense sentenses.

## Coding

Use minimalistic architectures and structures.

Always use short but descriptive information-dense names. Use abbreviations or acronyms only for famous names, for example ``acc`` for ``accuracy``, or ``*_fn`` or ``*_func`` for function names.

When commenting a code, keep the comments concise and precise with accurate 
and descriptive words. Create short and information dense sentences.

Use ASD-STE100. 

Use the ``plain-english`` skill for comments and docstrings, not only for 
messages to the user. A comment says what the code does and why, in words a 
reader outside this project can follow. Code stays exact: real names, real 
paths, real error text, real numbers. Prose gets plain words.

Swap the jargon, do not explain it. ``save`` not ``persist``. ``use`` not 
``leverage``. ``clean up`` not ``refactor``. ``what is public`` not ``API 
surface``. ``safe to run twice`` not ``idempotent``. ``later step`` not 
``downstream``.

A term with no short plain word, and that the reader must know, is defined one 
time in parentheses and then used plainly.

Do not lose accuracy. Simplify the words, not the facts. A comment that records 
a defect, a threshold or a reason keeps every number and every name.

## Documentation

In the documents, keep a catalog of names of entities such as method, policy, schedule, etc. For each entity, provide an elaborate description and what it is and/or how it works, along with a code snippet if applicable. These entities can be things suchas augmentation policy, gradient optimizer function, gradient update schedule (row-wise, cell-wise, etc), force function, normalization method, etc. Provenance: Include links to the corresponding code block or file. 

The catalog is an incrementing document. Do not remove anything. For update, mention the reason for update and the include the updated version. Also include at which version of the code development or experiment or research step the update is adopted.

## Agentics

A chat session agent must print in the chat windows the timestamp, sender name and ID, and the prompt body of any incoming prompt from other agents.

## Python Environment

This project's virtual environment lives at ``.venv/``. 
Invoke its interpreter directly. Do not run ``source .venv/bin/activate`` as it does not persist between tool calls.

Use these exact commands:
- Run a script: ``.venv/bin/python script.py``
- Install a package after granted permission: ``.venv/bin/pip install <package>``
- Run a tool: ``.venv/bin/pytest``

To store large files and artifacts use ``/tmp/`` for temporary ones, and ``data_cache/`` for others. These files can be ``.npz`` and other files. You may use a symlink to link to them locally.

