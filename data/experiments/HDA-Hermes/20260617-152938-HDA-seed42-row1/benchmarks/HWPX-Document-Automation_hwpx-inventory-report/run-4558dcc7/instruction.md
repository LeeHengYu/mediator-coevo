# Task Instruction

Complete the inventory status report by replacing placeholders in a .hwpx template with values from a JSON file.

## Steps

### 1. Inspect the workspace
```bash
ls -la /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```

### 2. Read the JSON data file
```bash
cat /root/inventory_data.json
```
Note all keys and values. These will map to `{{key}}` placeholders in the template.

### 3. Explore the .hwpx package structure
A `.hwpx` file is a ZIP archive containing XML files.
```bash
cd /root
mkdir -p hwpx_work
cp inventory_report_template.hwpx hwpx_work/template.zip
cd hwpx_work
unzip -o template.zip -d template_contents
find template_contents -type f
```

### 4. Identify content XML files containing placeholders
```bash
grep -rl '{{' template_contents/
```
For each file found, inspect its full contents:
```bash
cat <file>
```
Pay close attention to:
- The XML structure and namespace prefixes used
- Where `{{...}}` placeholders appear (they should be inside text run elements)
- Layout cache elements associated with paragraphs (elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, `<hp:lineseg>`, or any element whose tag contains 'LineSeg', 'lineseg', 'lineSegArray', 'linesegarray', or similar cached layout data)
- Empty paragraphs (preserve them)
- Korean text labels (preserve them)

### 5. Write a Python script to perform replacements
Create `/root/hwpx_work/fix_hwpx.py` that:

a) Reads `inventory_data.json` to get the replacement mapping.

b) For each XML file that contains `{{` placeholders:
   - Parses the XML (use `lxml.etree` if available, otherwise `xml.etree.ElementTree`). Be careful to preserve all namespaces and the XML declaration.
   - For every text node (`.text` and `.tail` of all elements), replaces `{{key}}` with the corresponding JSON value (convert numbers to strings).
   - For any paragraph element that had a replacement performed in any of its descendant text nodes, removes all layout-cache child elements. These are typically elements whose local tag name (ignoring namespace) matches patterns like: `linesegarray`, `LineSegArray`, `lineseg`, `LineSeg`. Look at the actual XML to identify the exact tag names. The key indicator is elements that cache glyph positions/widths/line layout.
   - Preserves all other elements, attributes, empty paragraphs, and Korean text unchanged.

c) Writes the modified XML back, preserving the XML declaration and encoding.

d) Rebuilds the .hwpx ZIP package from the modified contents.

**CRITICAL details for the script:**
- When parsing XML, register all namespaces found in the document BEFORE parsing to avoid `ns0:` prefix mangling. Use `iterparse` or scan for `xmlns` declarations first.
- After all replacements, verify NO `{{` remains in any text node across all XML files.
- Use `zipfile.ZipFile` to repackage. Copy all files from the extracted directory back into a new ZIP. Preserve the directory structure exactly.
- Save the final output to `/root/inventory_report_ready.hwpx`.

### 6. Run the script
```bash
cd /root/hwpx_work
python3 fix_hwpx.py
```

### 7. Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/inventory_report_ready.hwpx

# Check no placeholders remain
mkdir -p /root/hwpx_work/verify
unzip -o /root/inventory_report_ready.hwpx -d /root/hwpx_work/verify
grep -r '{{' /root/hwpx_work/verify/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'

# Verify Korean text is preserved (spot check)
grep -r '재고' /root/hwpx_work/verify/ | head -5

# Verify JSON values appear in output
cat /root/inventory_data.json
# Then grep for a few key values in the output XML
```

### 8. If any issues, debug and fix
- If namespace prefixes are mangled, fix the namespace registration approach.
- If placeholders remain, check if they are split across multiple XML text nodes (e.g., `{{` in one run and `key}}` in another). If so, implement run-merging logic: concatenate adjacent text runs, perform replacement, then set the merged text on the first run and clear/remove subsequent runs.
- If the ZIP structure is wrong, compare with the original template structure.

The final deliverable is `/root/inventory_report_ready.hwpx`.

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