# Task Instruction

Complete the following task to update a HWPX renewal playbook document.

## Goal
Revise `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, then save the updated file to `/root/renewal_playbook_updated.hwpx`.

## Steps

### 1. Inspect the workspace
- List files in the task directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, `followups.csv`, and any test/verifier files.
- Read `renewal_update.json` to learn the new field values (customer name, current owner, renewal window, pricing band, escalation contact, pricing note).
- Read `followups.csv` to learn the replacement follow-up items and their `sequence` ordering.
- Examine the verifier (`test_output.py` or similar) to understand exactly what assertions will be checked.

### 2. Understand the HWPX structure
- A `.hwpx` file is an OPC (ZIP) package. Unzip it to a temporary directory.
- Identify the main content XML file(s) — typically under `Contents/` (e.g., `Contents/section0.xml`).
- Read the XML content and understand the namespace prefixes (commonly `hp:` for Hancom elements).
- Locate all editable paragraphs containing the old field values and the three existing follow-up lines.
- Locate the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` and note its position — it must remain untouched.

### 3. Write a Python script to perform the update
Create a Python script (e.g., `/root/update_hwpx.py`) that does the following:

```python
import json, csv, zipfile, shutil, os, copy
from lxml import etree

# Paths
SRC = '<path_to_renewal_playbook.hwpx>'
UPDATE_JSON = '<path_to_renewal_update.json>'
FOLLOWUPS_CSV = '<path_to_followups.csv>'
OUT = '/root/renewal_playbook_updated.hwpx'
TMP_DIR = '/tmp/hwpx_work'

# 1. Read update data
with open(UPDATE_JSON) as f:
    updates = json.load(f)
with open(FOLLOWUPS_CSV) as f:
    reader = csv.DictReader(f)
    followups = sorted(reader, key=lambda r: int(r['sequence']))

# 2. Unzip
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR)
with zipfile.ZipFile(SRC) as zf:
    zf.extractall(TMP_DIR)

# 3. Find and parse content XML (section0.xml or similar)
# Walk TMP_DIR to find XML files under Contents/
# Parse with lxml, register namespaces

# 4. For each paragraph:
#    a. Extract full text from <hp:t> elements
#    b. Perform text replacements for all fields from updates dict
#       (old_value -> new_value for each field)
#    c. Replace the three follow-up lines with CSV items in sequence order
#    d. Do NOT touch the appendix sentence
#    e. For any paragraph whose text was modified:
#       - Remove all <hp:lineSegArray> child elements to clear layout cache

# 5. Write modified XML back to the file in TMP_DIR

# 6. Re-zip TMP_DIR into OUT, preserving the original ZIP structure
#    Use zipfile.ZipFile with ZIP_DEFLATED, walk all files in TMP_DIR
```

Key implementation details:
- **Field replacements**: The JSON likely has old/new pairs or just new values. Inspect the JSON structure. You need to find old values by reading the original XML text. Replace every occurrence in editable sections.
- **Follow-up replacement**: Identify the three existing follow-up lines (they likely share a pattern or are consecutive paragraphs). Replace them with the CSV items sorted by `sequence`. The CSV likely has a `text` or `content` column — inspect it. If there are more or fewer than 3 CSV items, add/remove paragraph elements accordingly (clone structure from existing follow-up paragraphs).
- **Appendix preservation**: Before and after processing, verify the appendix sentence is present and unchanged.
- **Layout cache removal**: For every `<hp:p>` element where you changed any `<hp:t>` text, find and remove all `<hp:lineSegArray>` descendants. This is critical for clean rendering.
- **Namespace handling**: Use `lxml` namespace maps. Register all namespaces found in the document to avoid prefix changes on serialization.
- **ZIP recreation**: When re-zipping, iterate over all files in the extracted directory. Use `os.path.relpath` to maintain correct archive paths. Preserve `[Content_Types].xml` and relationship files.

### 4. Execute and validate
- Run the Python script.
- Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
- Unzip the output and check that:
  - All old field values are gone from the content XML.
  - All new field values are present.
  - Follow-up items appear in correct sequence order.
  - The appendix sentence is unchanged.
  - No `<hp:lineSegArray>` elements remain in modified paragraphs.
- Run the verifier/test suite (e.g., `cd <task_dir> && python -m pytest test_output.py -v`) to confirm the result passes.

### 5. Troubleshoot if needed
- If tests fail, read the specific assertion error, re-inspect the output XML, and fix the script.
- Common pitfalls: missing a replacement location, wrong follow-up ordering, accidentally modifying the appendix, leaving stale lineSegArray elements, or breaking ZIP structure.

## Critical Constraints
- Do NOT add duplicate lines — remove old values before inserting new ones.
- Do NOT modify the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`
- The output MUST be a valid `.hwpx` (ZIP/OPC) package.
- Strip `hp:lineSegArray` from every paragraph whose text content was changed.

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.