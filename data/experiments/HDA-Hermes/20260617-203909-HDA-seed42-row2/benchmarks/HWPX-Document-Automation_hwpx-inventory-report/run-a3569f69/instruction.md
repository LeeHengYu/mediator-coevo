# Task Instruction

Complete the inventory status report by replacing placeholders in a .hwpx template with JSON data.

## Steps

### 1. Inspect the workspace and understand the data
```bash
ls -la /root/
cat /root/inventory_data.json
```

### 2. Understand the .hwpx package structure
`.hwpx` is a ZIP archive. Extract and inspect it:
```bash
mkdir -p /tmp/hwpx_work
cp /root/inventory_report_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_extracted
find template_extracted -type f | sort
```

### 3. Find where placeholders live
Search all XML files for `{{` to locate every placeholder:
```bash
grep -r '{{' template_extracted/
```
Also check if placeholders might be split across XML tags:
```bash
grep -rn '{\|}' template_extracted/ | head -60
```
Examine the main content XML file(s) carefully — typically `Contents/section0.xml` or similar:
```bash
cat template_extracted/Contents/section*.xml
```

### 4. Write a Python script to perform the replacement
Create `/tmp/hwpx_work/process.py` that:

a) Reads `inventory_data.json` to get all key-value pairs (handle nested keys if needed — check the JSON structure first).

b) Opens the .hwpx as a ZIP, iterates over all entries.

c) For XML files (especially content XMLs), performs placeholder replacement:
   - First, normalize the XML text: if `{{` and `}}` delimiters are split across multiple XML run elements within the same paragraph, concatenate the text, do the replacement, then reconstruct. This is the hardest part.
   - A robust approach: parse with `lxml` or `xml.etree.ElementTree`. For each paragraph element, collect all text content, check if it contains `{{...}}`, and if so, do the replacement on the joined text, then redistribute back into the run elements (or consolidate into fewer runs).
   - After modifying a paragraph's text, remove any layout-cache child elements from that paragraph. These are typically elements like `<hp:linesegarray>` or elements related to cached glyph/line layout. Inspect the XML to identify the exact element names. Remove them only from paragraphs where text was changed.

d) Preserve empty paragraphs — do not delete any paragraph elements, even if they have no text.

e) Preserve all Korean labels and static note lines — only modify text that contains `{{...}}`.

f) Write the result as a new ZIP to `/root/inventory_report_ready.hwpx`, preserving the same ZIP structure and compression.

### 5. Important implementation details for the Python script
- Use `zipfile` module to read/write.
- Use `lxml.etree` (preferred) or `xml.etree.ElementTree` for XML parsing. Register all namespaces found in the XML to avoid namespace prefix changes.
- When looking for layout cache elements, inspect the actual XML first. Common HWPX layout cache element names include `linesegarray`, `lineSegArray`, or similar. Look for any child of `<p>` (paragraph) elements that seem to be layout/rendering caches.
- The placeholder format is `{{key_name}}`. Map these to JSON keys. Check if the JSON is flat or nested — if nested, use dot-notation or flatten appropriately.
- Convert non-string JSON values (numbers, etc.) to strings for replacement.
- After writing the output, verify: unzip it and grep for `{{` to confirm zero remaining placeholders.

### 6. Execute and verify
```bash
cd /tmp/hwpx_work
python3 process.py
```

Then verify:
```bash
mkdir -p /tmp/hwpx_work/output_check
cp /root/inventory_report_ready.hwpx /tmp/hwpx_work/output_check/out.zip
cd /tmp/hwpx_work/output_check
unzip out.zip -d out_extracted
# Must return nothing:
grep -r '{{' out_extracted/
# Verify Korean text is preserved:
grep -r '재고' out_extracted/ | head -5
# Verify it's a valid zip:
python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_ready.hwpx'); print('Valid ZIP, entries:', len(z.namelist())); z.testzip()"
# Check empty paragraphs are preserved by counting <hp:p> or <p> elements:
grep -c '<hp:p\|<p ' out_extracted/Contents/section*.xml
```

### 7. Final sanity check
Compare paragraph counts between template and output to ensure no paragraphs were dropped:
```bash
for f in template_extracted/Contents/section*.xml; do echo "Template $(basename $f):"; grep -o '<hp:p[> ]' "$f" | wc -l; done
for f in out_extracted/Contents/section*.xml; do echo "Output $(basename $f):"; grep -o '<hp:p[> ]' "$f" | wc -l; done
```

If any `{{` remains or paragraph counts differ, debug and fix before finishing.

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