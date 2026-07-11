# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in the HWPX template with values from the JSON data file, then save the result as a valid `.hwpx` package.

## Step-by-step plan

### 1. Inspect the workspace
```bash
ls -la /root/
ls -la /root/*.hwpx /root/*.json 2>/dev/null || true
find /root/ -maxdepth 2 -name '*.hwpx' -o -name '*.json' 2>/dev/null
```
Identify the exact paths of `inventory_report_template.hwpx` and `inventory_data.json`.

### 2. Understand the HWPX structure
A `.hwpx` file is a ZIP package (like DOCX). Extract it to inspect its contents:
```bash
mkdir -p /tmp/hwpx_work
cp <path_to_template> /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip template.hwpx -d template_extracted
find template_extracted -type f
```
Identify which XML files contain the document body text. Typically the main content is in a file like `Contents/section0.xml` or similar.

### 3. Read the JSON data
```bash
cat <path_to_inventory_data.json>
```
Note every key-value pair. These keys correspond to `{{key}}` placeholders in the XML.

### 4. Inspect all XML files for placeholders
Search for every `{{` occurrence across all extracted files:
```bash
grep -rn '{{' template_extracted/
```
This will show exactly which files and lines contain placeholders, and what the placeholder names are. Cross-reference every placeholder name against the JSON keys to ensure complete coverage.

### 5. Read the full content of each file containing placeholders
Before editing, read the complete content of each file that contains `{{...}}` patterns. Pay special attention to:
- Korean labels that must remain unchanged
- Empty paragraphs (spacing elements) that must be preserved
- Layout-cache elements (e.g., `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:charShapeArray>`, or similar cached layout data) within paragraphs that contain placeholders

### 6. Perform replacements using Python
Write a Python script that:
1. Loads the JSON data file.
2. For each XML file containing placeholders:
   a. Reads the file content as a UTF-8 string.
   b. Replaces every `{{key}}` with the corresponding JSON value (converted to string).
   c. **Important**: For any paragraph (`<hp:p>` element or equivalent) whose text was modified (i.e., contained a placeholder), remove any layout-cache child elements. These are typically elements like `<hp:linesegarray>...</hp:linesegarray>` or `<hp:lineSegArray>...</hp:lineSegArray>` that cache glyph positions. Removing them forces the application to recalculate layout on open, preventing overlapping characters.
   d. Writes the modified content back.
3. Verifies that no `{{` remains in any file in the extracted directory.

Here is the approach for step 6c in more detail:
- Parse the XML with `xml.etree.ElementTree` (register all namespaces first to avoid namespace prefix changes).
- Walk all paragraph elements.
- For each paragraph, collect its full text content. If the text differs from the original (was modified), find and remove child elements that are layout caches.
- Alternatively, if the XML structure is complex, use a regex-based approach: after doing string replacements, use regex to find paragraph blocks that were modified and strip `<hp:linesegarray>...</hp:linesegarray>` (case-insensitive tag matching) from those paragraphs.
- Be careful with namespaces; inspect the actual XML to determine the exact tag names and namespace prefixes used.

### 7. Repack the HWPX
```bash
cd /tmp/hwpx_work/template_extracted
zip -r /root/inventory_report_ready.hwpx . -x '*.DS_Store'
```
Use `zip` with stored (or deflate) compression matching the original. The key is that the resulting ZIP must be a valid HWPX package.

### 8. Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/inventory_report_ready.hwpx

# Check no placeholders remain
unzip -p /root/inventory_report_ready.hwpx | grep -c '{{' 
# This should output 0

# Verify the file exists at the correct path
ls -la /root/inventory_report_ready.hwpx
```

## Critical constraints
- **Do NOT change Korean text labels** — only replace `{{...}}` patterns.
- **Do NOT remove empty paragraphs** — they serve as spacing.
- **DO remove layout-cache elements** (like `linesegarray` or similar) from any paragraph whose text content was modified by placeholder replacement.
- **No `{{...}}` may remain** in any file inside the package.
- The output file must be at exactly `/root/inventory_report_ready.hwpx`.
- The output must be a valid ZIP archive with the same internal structure as the original HWPX.

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