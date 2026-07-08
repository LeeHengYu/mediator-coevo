# Task Instruction

Complete the following task to update a HWPX renewal playbook document.

## Goal
Revise `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Step-by-step Plan

### Step 1: Inspect input files
1. List files in the working directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.
2. Read and display `renewal_update.json` to understand the field mappings (customer name, current owner, renewal window, pricing band, escalation contact, pricing note).
3. Read and display `followups.csv` to understand the follow-up items and their `sequence` column for ordering.

### Step 2: Inspect the HWPX structure
1. The HWPX file is a ZIP archive. List its contents using Python's `zipfile` module.
2. Extract and display the full XML content of `Contents/section0.xml` — this is where the editable text lives.
3. Identify the XML namespaces used (especially `hp:` namespace for paragraphs, text runs, and layout cache elements like `<hp:lineSegArray>`).
4. Identify the exact current text values that need to be replaced (old customer name, old owner, old renewal window, old pricing band, old escalation contact, old pricing note).
5. Identify the three existing follow-up lines that need to be replaced.
6. Locate the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` and note its position — it must NOT be modified.

### Step 3: Write and execute the update script
Write a single Python script that does all of the following:

```python
import zipfile
import json
import csv
import os
import io
import copy
from lxml import etree

# 1. Load renewal_update.json
with open('renewal_update.json', 'r', encoding='utf-8') as f:
    updates = json.load(f)

# 2. Load followups.csv, sorted by 'sequence' column
with open('followups.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    followups = sorted(list(reader), key=lambda r: int(r['sequence']))

# 3. Read the HWPX ZIP, extract all entries preserving order
original_zip = 'renewal_playbook.hwpx'
entries = {}  # name -> bytes
with zipfile.ZipFile(original_zip, 'r') as zf:
    for info in zf.infolist():
        entries[info.filename] = zf.read(info.filename)

# 4. Parse section0.xml with lxml, preserving namespaces
xml_bytes = entries['Contents/section0.xml']
tree = etree.fromstring(xml_bytes)
# ... (namespace-aware XPath to find text runs)
# ... Map old values to new values from updates dict
# ... Find follow-up paragraphs and replace with CSV items in sequence order
# ... Preserve the appendix sentence exactly
# ... Remove <hp:lineSegArray> (and similar layout-cache elements) from any paragraph whose text was modified

# 5. Serialize modified XML back
modified_xml = etree.tostring(tree, xml_declaration=True, encoding='UTF-8')
entries['Contents/section0.xml'] = modified_xml

# 6. Repackage as HWPX ZIP with mimetype as first uncompressed entry
output_path = '/root/renewal_playbook_updated.hwpx'
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
    # mimetype must be first and stored (not compressed)
    if 'mimetype' in entries:
        zout.writestr('mimetype', entries['mimetype'], compress_type=zipfile.ZIP_STORED)
    for name, data in entries.items():
        if name == 'mimetype':
            continue
        zout.writestr(name, data)
```

The above is a skeleton. You must adapt it based on what you actually observe in the XML. Key details:

**Field replacements:** For each key in `renewal_update.json`, identify the OLD value currently in the XML text runs and replace it with the NEW value. The JSON likely has a structure mapping field names to new values. You need to find the corresponding old values by reading the XML. Replace ALL occurrences in editable sections.

**Follow-up replacement:** Identify the three existing follow-up line paragraphs in the XML. Replace their text content with the CSV items ordered by `sequence`. Do NOT add duplicates — remove the old text and insert the new text in the same paragraph structures.

**Appendix protection:** Before any text replacement, check if a paragraph contains `이 부록 문단은 그대로 유지해야 합니다.` and skip it entirely.

**Layout cache removal:** For every paragraph element (`<hp:p>` or similar) where you modified any text run, find and remove child elements like `<hp:lineSegArray>`, `<hp:lineSeg>`, or any layout-cache elements. This prevents overlapping character rendering.

### Step 4: Validate the output
1. Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
2. Extract `Contents/section0.xml` from the output and verify:
   - All new values from `renewal_update.json` appear in the text.
   - None of the old values remain.
   - Follow-up items appear in the correct sequence order.
   - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged.
   - No `<hp:lineSegArray>` elements remain in paragraphs that were modified.
3. Verify `mimetype` is the first entry in the ZIP and is stored uncompressed.
4. If the task directory has a test file (e.g., `test_output.py`), run it with `pytest` to confirm.

### Critical Reminders
- Use `lxml` (not `xml.etree.ElementTree`) to properly handle namespaces.
- When searching for text, look inside `<hp:t>` elements within `<hp:run>` elements within `<hp:p>` elements. The namespace URI for `hp:` must be extracted from the XML root.
- Be careful with text that may be split across multiple `<hp:run>` elements in the same paragraph — you may need to concatenate runs to find a match, then update the appropriate run(s).
- Do NOT modify any paragraph containing the appendix sentence.
- Always re-read the XML after parsing to understand the exact structure before writing replacement logic.

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