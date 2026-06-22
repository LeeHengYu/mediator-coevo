# Task Instruction

Execute the following steps to produce `/root/renewal_playbook_updated.hwpx`.

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name 'renewal_playbook.hwpx' -o -name 'renewal_update.json' -o -name 'followups.csv' 2>/dev/null
```
Identify the exact paths of the three input files.

### 2. Examine the input data
```bash
cat <path-to>/renewal_update.json
cat <path-to>/followups.csv
```
Note every field name and value in the JSON (customer name, current owner, renewal window, pricing band, escalation contact, pricing note). Note every row in the CSV and its `sequence` column for ordering.

### 3. Explore the HWPX package structure
```python
import zipfile, os
with zipfile.ZipFile('<path-to>/renewal_playbook.hwpx') as z:
    for info in z.infolist():
        print(info.filename, info.compress_type, info.file_size)
```
Identify the main content XML (likely `Contents/section0.xml` or similar) and other XML files that might contain editable text.

### 4. Read and understand the content XML
Extract and print the content XML. Identify:
- All paragraphs containing the old customer name, owner, renewal window, pricing band, escalation contact, pricing note.
- The three follow-up lines that must be replaced.
- The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` (must be preserved exactly).
- The namespace URIs used (especially the `hp:` prefix).

### 5. Write a Python script that performs the update

Create and run a single Python script (`/root/update_hwpx.py`) that does the following:

#### a. Setup
- `import zipfile, json, csv, xml.etree.ElementTree as ET, io, os, copy, re`
- Register all namespaces found in the XML via `ET.register_namespace(prefix, uri)` **before** any parsing, to prevent namespace prefix mangling on output.
- Load `renewal_update.json` into a dict.
- Load `followups.csv`, sort rows by the `sequence` column (ascending integer), collect the follow-up text items in order.

#### b. Parse the content XML
- Open the HWPX zip, read the content XML file into an ElementTree.

#### c. Build an old→new replacement map
From the JSON, build a mapping of every old value to its new value for: customer name, current owner, renewal window, pricing band, escalation contact, pricing note. Print this map for debugging.

#### d. Replace field values in editable paragraphs
- Walk every paragraph element. For each paragraph, collect all text runs (the `<hp:t>` or similar text elements).
- Concatenate the full paragraph text. If the paragraph text contains any old value, perform the replacement in the individual text-run elements.
- **Critical**: if any text in a paragraph was modified, remove all `<hp:lineSegArray>` child elements (and any `<hp:linesegarray>` case variants) from that paragraph. This prevents layout-cache artifacts.
- Do NOT modify the appendix paragraph containing `이 부록 문단은 그대로 유지해야 합니다.`.

#### e. Replace the three follow-up lines
- Identify the three existing follow-up paragraphs. They likely share a pattern (numbered follow-up items or a recognizable prefix).
- Replace their text content with the CSV items in `sequence` order (first CSV item → first follow-up paragraph, etc.).
- If there are more or fewer CSV items than existing follow-up paragraphs, clone or remove paragraph elements as needed to match the CSV count.
- Remove `<hp:lineSegArray>` from each modified follow-up paragraph.
- Ensure old follow-up text is fully removed (no duplicates).

#### f. Serialize and repackage
- Write the modified XML back to bytes.
- Create the output HWPX zip at `/root/renewal_playbook_updated.hwpx`.
- Write `mimetype` as the **first entry** with `ZIP_STORED` compression (no compression), matching OPC container requirements.
- Copy all other entries from the original zip, replacing the content XML with the modified version. Use appropriate compression (`ZIP_DEFLATED`) for non-mimetype entries.

#### g. Validate
- Open the output zip and confirm it is a valid zip.
- Re-parse the content XML from the output to confirm:
  - All old values are absent.
  - All new values are present.
  - Follow-up lines match the CSV in sequence order.
  - The appendix sentence is unchanged.
  - No `<hp:lineSegArray>` elements exist in any modified paragraph.
- Print a summary of validation results.

### 6. Run the script
```bash
python /root/update_hwpx.py
```

### 7. Final check
```bash
python -c "
import zipfile
z = zipfile.ZipFile('/root/renewal_playbook_updated.hwpx')
print('Valid zip, entries:', len(z.infolist()))
for i in z.infolist():
    print(i.filename, i.compress_type)
z.close()
print('Output file exists and is valid.')
"
```

Confirm the file exists at the exact path `/root/renewal_playbook_updated.hwpx` and is a valid zip/HWPX package.

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