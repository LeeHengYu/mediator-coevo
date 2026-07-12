# Task Instruction

Execute the following steps in order to produce `/root/safety_audit_brief_final.hwpx`.

## 1 – Understand the HWPX format
A `.hwpx` file is a ZIP archive (ODF-like package used by Hancom Hangul). Inside it the main document body is typically at `Contents/section0.xml` (or similar). You will manipulate the archive in-place using Python's `zipfile` module.

## 2 – Locate and inspect all inputs
```bash
find /root -type f | head -80
```
Identify:
- `safety_audit_template.hwpx` – the template
- `audit_overview.json` – overview/summary data
- `corrective_actions.json` – three corrective-action records

Read both JSON files:
```bash
cat /root/*.json 2>/dev/null || find / -name 'audit_overview.json' -o -name 'corrective_actions.json' 2>/dev/null
```

## 3 – Explore the HWPX template
```python
import zipfile, os
template = '<path to safety_audit_template.hwpx>'
with zipfile.ZipFile(template) as z:
    for name in z.namelist():
        print(name)
```
Then read every XML file inside, especially any `section*.xml`, looking for:
- `{{...}}` placeholder tokens
- The structure of the summary section and audit table
- Where the risk tier, inspection date, and corrective-action lines appear
- Any layout-cache or char-shape elements attached to paragraphs with placeholders

Print the full XML content of every file that contains `{{` so you can see exact element structure.

## 4 – Build the replacement logic (Python script)
Write a single Python script `/root/build_hwpx.py` that does the following:

### 4a – Load JSON data
```python
import json, zipfile, re, shutil, os
with open('<path>/audit_overview.json') as f:
    overview = json.load(f)
with open('<path>/corrective_actions.json') as f:
    actions = json.load(f)  # expect a list of 3 items
```

### 4b – Date rewriting
Convert every date value from `YYYY-MM-DD` → `YYYY.MM.DD`. Do this both in the JSON values you'll substitute AND in any already-present literal dates in the XML.

### 4c – Risk-tier severity note
Map: `High` → `즉시조치`, `Medium` → `계획보완`, `Low` → `모니터링`.
Wherever the risk tier value is inserted (or already present as a placeholder result), append the severity note immediately after it, e.g. `High 즉시조치`.

### 4d – Placeholder substitution
For each XML file inside the HWPX archive:
1. Read the XML as text (UTF-8).
2. Replace all `{{placeholder_name}}` tokens with the corresponding values from the JSON data. Be careful: the placeholder text may be split across multiple XML inline elements (e.g., `<hp:t>{{</hp:t><hp:t>field}}</hp:t>`). To handle this:
   - First, collapse adjacent text runs within the same paragraph into a single text run, then do the replacement. OR
   - Work on the concatenated text of each paragraph, do replacements, then put the result back in a single `<hp:t>` element per paragraph.
3. For corrective-action lines, fill them in the order they appear in `corrective_actions.json`.
4. After substitution, ensure **no `{{...}}`** remains anywhere.

### 4e – Remove stale layout-cache elements
For any paragraph whose text content was modified, remove layout-cache child elements. These are typically `<hp:linesegarray>` or `<hp:lineSegArray>` or similar elements. Inspect the actual XML namespace and element names first, then strip them from modified paragraphs. This prevents overlapping-character rendering issues.

### 4f – Rewrite the HWPX
```python
output = '/root/safety_audit_brief_final.hwpx'
shutil.copy(template, output)
# Rewrite modified entries
with zipfile.ZipFile(template, 'r') as zin:
    with zipfile.ZipFile(output, 'w') as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in modified_files:
                data = modified_files[item.filename]
            zout.writestr(item, data)
```

## 5 – Validate
After running the script:
1. Verify the output is a valid ZIP:
   ```bash
   python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx'); z.testzip(); print('OK')"
   ```
2. Extract and print every XML that was modified; confirm:
   - No `{{` or `}}` remains anywhere in any file
   - All overview fields are filled
   - Audit table value cells are filled
   - Three corrective-action lines present in correct order
   - Every occurrence of the risk tier has the severity note appended
   - All dates are in `YYYY.MM.DD` format (no hyphens in dates)
   - Section titles and row labels are unchanged
   - Layout-cache elements removed from modified paragraphs
3. If any check fails, fix and re-run.

## 6 – Final confirmation
```bash
ls -la /root/safety_audit_brief_final.hwpx
```
Confirm the file exists and has a reasonable size (> 1 KB).

## Key cautions
- **Namespace awareness**: HWPX XML uses namespaces like `urn:hancom:hwpml:...`. When searching for elements, use the actual namespace prefixes found in the file. Print them first.
- **Split placeholders**: The biggest risk is placeholders split across XML runs. Always concatenate paragraph text before matching.
- **Preserve structure**: Do not remove or rename any XML elements except layout-cache elements on modified paragraphs. Keep all section titles and row labels verbatim.
- **Encoding**: Write XML as UTF-8. Korean characters must be preserved correctly.
- **Order matters**: Corrective actions must appear in the same order as in the JSON array.

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