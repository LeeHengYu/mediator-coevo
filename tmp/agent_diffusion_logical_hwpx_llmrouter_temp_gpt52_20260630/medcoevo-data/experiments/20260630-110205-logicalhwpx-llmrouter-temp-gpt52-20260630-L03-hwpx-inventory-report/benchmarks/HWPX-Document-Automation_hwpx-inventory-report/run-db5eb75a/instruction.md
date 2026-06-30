# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in the HWPX template with values from the JSON data file.

## Step-by-step plan

### 1. Inspect the workspace
```bash
ls /root/
ls /root/HWPX-Document-Automation/hwpx-inventory-report/
```
Identify the template file (`inventory_report_template.hwpx`) and data file (`inventory_data.json`).

### 2. Read the JSON data
```bash
cat /root/HWPX-Document-Automation/hwpx-inventory-report/inventory_data.json
```
Note every key-value pair. These keys correspond to the `{{key}}` placeholders in the template.

### 3. Understand the HWPX package structure
A `.hwpx` file is a ZIP archive containing XML files. The main document content is typically in `Contents/section0.xml` (or similar).
```bash
cd /root/HWPX-Document-Automation/hwpx-inventory-report/
cp inventory_report_template.hwpx /tmp/template_backup.hwpx
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip -o /root/HWPX-Document-Automation/hwpx-inventory-report/inventory_report_template.hwpx
find . -type f | sort
```

### 4. Locate all placeholders
Search every XML file for `{{` patterns:
```bash
grep -r '{{' /tmp/hwpx_work/
```
Record every placeholder found and which file(s) contain them.

### 5. Examine the content XML carefully
Read the full content of each file that contains placeholders:
```bash
cat <file_with_placeholders>
```
Pay attention to:
- The exact XML structure around each placeholder
- Korean labels and static note lines (must be preserved)
- Empty paragraphs (must be preserved)
- Layout cache elements (`<hp:linesegarray>`, `<hp:lineseg>`, `<hp:lineSegArray>`, or similar elements that cache glyph/line layout)

### 6. Write a Python script to perform the replacement
Create a Python script that:
1. Reads `inventory_data.json` to get the replacement mapping.
2. Copies the template HWPX to the output path.
3. Opens the output HWPX as a ZIP, finds all XML content files.
4. For each XML file containing `{{...}}` placeholders:
   a. Parses the XML.
   b. Replaces every `{{key}}` with the corresponding JSON value (converted to string).
   c. **Critical**: For any `<hp:run>` or paragraph element whose text was modified, removes all child layout-cache elements. These are typically `<hp:linesegarray>` or `<hp:lineSegArray>` or similar elements. Inspect the actual element names in step 5 and target them precisely. The layout cache elements store pre-computed glyph positions that become stale after text replacement and cause overlapping characters.
   d. Preserves all other elements, empty paragraphs, Korean text, and static content exactly.
5. Writes the modified XML back into the ZIP.
6. Saves the result to `/root/inventory_report_ready.hwpx`.

IMPORTANT implementation notes:
- Use `zipfile` module. Read the template, create a new ZIP for output, copy all entries, replacing only the modified XML files.
- Use `lxml.etree` or `xml.etree.ElementTree` for XML parsing. Preserve namespaces carefully. When using ElementTree, register all namespaces found in the document before parsing to avoid namespace prefix mangling.
- The placeholder `{{...}}` text may be split across multiple XML text nodes within a single `<hp:t>` or run element due to HWPX editor behavior. Check if placeholders are split. If they are, you may need to concatenate adjacent text nodes, perform replacement, then set the result back. Inspect the actual XML to determine the exact approach.
- Convert numeric JSON values to strings before replacement.
- Handle the case where a placeholder like `{{key}}` might appear inside a `<hp:t>` element's text content.

### 7. Run the script
```bash
python3 /tmp/replace_placeholders.py
```

### 8. Validate the output
```bash
# Verify it's a valid ZIP
unzip -t /root/inventory_report_ready.hwpx

# Check no placeholders remain
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify
unzip -o /root/inventory_report_ready.hwpx
grep -r '{{' /tmp/hwpx_verify/
# This should return NO results

# Verify Korean labels and structure are preserved
# Compare structure with original
diff <(unzip -l /root/HWPX-Document-Automation/hwpx-inventory-report/inventory_report_template.hwpx) <(unzip -l /root/inventory_report_ready.hwpx)
# File list should be identical

# Verify empty paragraphs are preserved - count paragraph elements
grep -c '<hp:p ' /tmp/hwpx_verify/Contents/section0.xml
# Compare with original count from step 5

# Verify no stale layout cache elements in modified paragraphs
# (The script should have removed them)
```

### 9. If validation fails
- If placeholders remain: check if they were split across XML nodes and fix the script.
- If the ZIP is invalid: check the repackaging logic.
- If layout cache elements remain in modified paragraphs: fix the cache removal logic.
- Re-run and re-validate.

### Key pitfalls to avoid (from cross-task feedback)
- **Stale layout cache**: This was a failure mode in a similar task. You MUST remove layout cache elements (`lineSegArray` or similar) from any paragraph whose text content was modified.
- **Unreplaced placeholders**: Another failure mode. Ensure ALL placeholders are replaced, even if text is split across XML nodes.
- **Namespace handling**: HWPX XML uses multiple namespaces. Preserve them exactly. Register them before serialization.
- **Empty paragraphs**: Do not accidentally remove paragraphs that have no text content. They serve as spacing.

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