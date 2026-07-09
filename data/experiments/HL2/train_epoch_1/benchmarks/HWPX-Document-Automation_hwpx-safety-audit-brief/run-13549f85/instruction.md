# Task Instruction

You must produce `/root/safety_audit_brief_final.hwpx` by filling a template HWPX document with data from two JSON files. Follow every step below in order.

## 1 – Inspect the workspace
```bash
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -30
```
Identify the paths for `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

## 2 – Understand the data
```bash
cat <path-to>/audit_overview.json
cat <path-to>/corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The inspection date (will be in `YYYY-MM-DD` format; you must rewrite it to `YYYY.MM.DD` everywhere).
- The risk tier value (e.g. `High`, `Medium`, or `Low`).
- The severity-note mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`.
- The corrective actions list (preserve their order).

## 3 – Explore the HWPX package structure
An `.hwpx` file is a ZIP archive. Unzip it to a temp directory and list all files:
```bash
mkdir -p /tmp/hwpx_work
cp <template>.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
python3 -c "
import zipfile, os
with zipfile.ZipFile('template.hwpx','r') as z:
    for info in z.infolist():
        print(info.filename, info.compress_type)
    z.extractall('extracted')
"
find extracted/ -type f
```

## 4 – Identify content XML files with placeholders
Search for `{{` in all extracted XML files:
```bash
grep -rl '{{' extracted/
```
For each file found, print its full contents so you can see the XML structure and every placeholder.

## 5 – Write the Python automation script
Create `/tmp/hwpx_work/build.py` that does the following:

### 5a – Load JSON data
Read both JSON files. Derive:
- All overview/summary fields.
- The corrective-action lines (keep original order from `corrective_actions.json`).
- The `inspection_date` reformatted from `YYYY-MM-DD` to `YYYY.MM.DD`.
- The `risk_tier` value.
- The severity note string from the mapping.

### 5b – Process each XML file that contains placeholders
For each such XML file:
1. Parse as raw text (UTF-8).
2. **Join fragmented `<hp:t>` text nodes**: Placeholders like `{{key}}` may be split across multiple `<hp:t>` elements inside the same `<hp:run>`. Before doing any replacement, you must consolidate: within each `<hp:run>`, merge all `<hp:t>` element text content into the first `<hp:t>` and remove the extra `<hp:t>` elements. Use an XML parser (e.g., `lxml.etree`) for this.
3. After consolidation, perform placeholder replacements. Replace every `{{placeholder}}` with the corresponding value from the JSON data.
4. **Risk tier + severity note**: After replacing the risk-tier placeholder, find every occurrence of the risk-tier value in the document text nodes and append the severity note immediately after it, separated by a space. For example, if risk tier is `High`, every text node containing `High` (that refers to the risk tier) should become `High 즉시조치`. Be careful to only append the note once per occurrence and only where it refers to the risk tier (i.e., where the placeholder was or where the value appears in context).
5. **Date rewriting**: Find every occurrence of the inspection date in `YYYY-MM-DD` format and replace with `YYYY.MM.DD`.
6. **Remove layout caches**: Delete every `<hp:lineSegArray>` element (and its children) from every `<hp:p>` paragraph that you modified. This prevents stale layout cache from causing overlapping characters. Use the XML parser to find and remove these elements.
7. Ensure no `{{` or `}}` remains anywhere in any text node.
8. Serialize the XML back to UTF-8.

### 5c – Repack the HWPX ZIP
Rebuild the `.hwpx` ZIP archive:
- The `mimetype` file MUST be the **first entry** in the ZIP and stored with `ZIP_STORED` (no compression, compression type 0).
- All other files use `ZIP_DEFLATED`.
- Preserve the original directory structure exactly.
- Save to `/root/safety_audit_brief_final.hwpx`.

## 6 – Run the script
```bash
cd /tmp/hwpx_work
python3 build.py
```
Fix any errors that arise.

## 7 – Validate the output
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx','r') as z:
    names = z.namelist()
    print('First entry:', names[0])
    # Check mimetype is first and stored
    info = z.getinfo('mimetype')
    print('mimetype compress_type:', info.compress_type)
    # Search for leftover placeholders
    for name in names:
        data = z.read(name)
        if b'{{' in data:
            print('LEFTOVER PLACEHOLDER in', name)
            print(data.decode('utf-8','replace')[:2000])
    print('Total files:', len(names))
    print('All entries:', names)
"
```

Also verify:
- The date appears in `YYYY.MM.DD` format (not `YYYY-MM-DD`).
- The risk tier has the severity note appended.
- Corrective actions appear in the correct order.
- No `<hp:lineSegArray>` elements remain in modified paragraphs.
- Section titles and row labels are unchanged.

Print relevant text content from the XML to confirm all substitutions are correct.

## Critical reminders
- Placeholders are often fragmented across multiple `<hp:t>` elements. You MUST join them before replacement.
- Remove `<hp:lineSegArray>` from every `<hp:p>` you touched.
- `mimetype` must be first in ZIP with `ZIP_STORED`.
- Do not leave any `{{...}}` text.
- Keep all existing section titles and row labels verbatim.
- The severity note goes immediately after the risk tier text (e.g., `High 즉시조치`), everywhere the risk tier appears.
- The date format change (`-` to `.`) applies everywhere the date appears.

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