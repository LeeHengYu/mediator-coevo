# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in `inventory_report_template.hwpx` with values from `inventory_data.json`, then save the result to `/root/inventory_report_ready.hwpx`.

## Step-by-step plan

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name 'inventory_report_template.hwpx' -o -name 'inventory_data.json' 2>/dev/null
```
Identify the exact paths of both input files.

### 2. Understand the HWPX package structure
A `.hwpx` file is a ZIP archive (like OOXML). Unzip it to a temporary directory:
```bash
mkdir -p /tmp/hwpx_work
cp <path_to_template> /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip template.hwpx -d template_contents
```
List all files inside, paying special attention to XML files under `Contents/` (typically `section0.xml` or similar). These contain the document body text.

### 3. Load and inspect the JSON data
```bash
cat <path_to>/inventory_data.json
```
Note every key-value pair. These keys will appear as `{{key_name}}` in the XML.

### 4. Find all placeholders
```bash
grep -rn '{{' /tmp/hwpx_work/template_contents/
```
This will show every file and line containing `{{...}}` placeholders. Typically they are in section XML files.

### 5. Write a Python script to perform the replacement
Create `/tmp/hwpx_work/replace.py` that:

a. Reads `inventory_data.json` into a dict.

b. For each XML file that contains `{{...}}`:
   - Parse it (use `lxml.etree` or `xml.etree.ElementTree`).
   - Walk every text node (`.text` and `.tail` of every element).
   - **Critical**: Placeholders may be split across multiple `<hp:t>` (or similar inline text) elements within a single `<hp:run>`. Before replacing, concatenate the text of sibling inline text elements within each run, perform the replacement on the concatenated string, then set the first text element to the replaced result and clear/remove the remaining ones.
   - Replace every `{{key}}` with the corresponding JSON value (convert numbers to strings).
   - **Remove stale layout-cache elements**: After modifying any paragraph's text, find and remove any `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:charShapeIdRef>`, or similar layout-cache/glyph-run child elements within that paragraph element. These are cached layout data that will cause overlapping characters if left stale. Look for elements with local names like `linesegarray`, `LineSegArray`, `lineSeg`, or any element under a namespace that appears to be layout caching. Inspect the XML structure first to identify the exact element names.
   - Preserve all other content: Korean labels, static note lines, empty paragraphs (paragraphs with no text or only whitespace).

c. Write the modified XML back with the same XML declaration and encoding.

d. Re-pack everything into a new ZIP file at `/root/inventory_report_ready.hwpx`, preserving the original directory structure and using `ZIP_DEFLATED` compression. Make sure to include all files from the extracted contents (mimetype, META-INF, Contents, etc.).

### 6. Run the script
```bash
python3 /tmp/hwpx_work/replace.py
```

### 7. Validate the output

a. Verify it's a valid ZIP:
```bash
unzip -t /root/inventory_report_ready.hwpx
```

b. Verify no placeholders remain:
```bash
mkdir -p /tmp/hwpx_verify
unzip /root/inventory_report_ready.hwpx -d /tmp/hwpx_verify
grep -rn '{{' /tmp/hwpx_verify/
```
This must return nothing.

c. Verify Korean labels and static note are preserved:
```bash
grep -c '참고' /tmp/hwpx_verify/Contents/*.xml  # or whatever the Korean content is
```

d. Verify empty paragraphs are preserved by comparing paragraph counts:
```bash
grep -c '<hp:p ' /tmp/hwpx_work/template_contents/Contents/section0.xml
grep -c '<hp:p ' /tmp/hwpx_verify/Contents/section0.xml
```
Counts should match.

e. Verify layout-cache elements were removed from modified paragraphs:
```bash
# Check that linesegarray or similar elements don't appear in modified paragraphs
# (they may still appear in unmodified paragraphs, which is fine)
```

### Key cautions
- **Split placeholders**: HWPX editors often split text across multiple `<hp:t>` elements within a run or across runs. You MUST handle this by joining text within runs before replacement.
- **Layout cache removal**: Only remove layout-cache elements from paragraphs where text was actually modified. Leave unmodified paragraphs untouched.
- **Encoding**: Preserve UTF-8 encoding for Korean text.
- **ZIP structure**: The `mimetype` file, if present, should ideally be stored uncompressed (first entry, no compression) as per ODF/package conventions. Check the original ZIP to replicate its compression settings.
- **Namespace handling**: When parsing XML, be careful with namespaces. Use the namespace map from the root element. Don't strip or alter namespaces.

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