# Task Instruction

Complete the inventory status report by replacing placeholders in a .hwpx template with JSON data.

## Steps

### 1. Inspect the workspace and understand the files
```bash
cd /root
ls -la
cat inventory_data.json
```

### 2. Examine the .hwpx template structure
A `.hwpx` file is a ZIP archive containing XML files. Unzip it to inspect:
```bash
mkdir -p /tmp/hwpx_template
cp inventory_report_template.hwpx /tmp/hwpx_template/template.zip
cd /tmp/hwpx_template
unzip template.zip -d template_contents
find template_contents -type f | sort
```

### 3. Identify where placeholders live
Search all XML files inside the extracted archive for `{{` patterns:
```bash
grep -r '{{' template_contents/ --include='*.xml' -l
grep -r '{{' template_contents/ --include='*.xml'
```
Also check if placeholders might be split across multiple XML text runs within the same paragraph. If `{{` and `}}` appear in different `<hp:t>` or similar text elements within one run or across sibling runs, you will need to merge them before replacement.

### 4. Read the content XML files carefully
For each file that contains `{{` placeholders, read the full file content to understand:
- The XML namespace declarations
- The paragraph structure (elements like `<hp:p>`, `<hp:run>`, `<hp:t>`, etc.)
- Layout cache elements (like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:LineSeg>`, or `<hp:lineseg>` elements within paragraphs)
- How Korean text and labels are structured
- Empty paragraphs that must be preserved

### 5. Write a Python script to perform the replacement
Create a Python script `/tmp/hwpx_template/process.py` that:

a. Reads `inventory_data.json` from `/root/`.

b. Opens `inventory_report_template.hwpx` as a ZIP file.

c. For each file entry in the ZIP:
   - If the file is an XML file that contains placeholders:
     1. Parse the XML content.
     2. **Handle split placeholders**: Before doing replacements, check if `{{...}}` patterns are split across multiple consecutive text nodes within the same paragraph/run. If so, merge the text content, perform the replacement, then put the merged text in the first text node and remove the now-empty sibling nodes (or clear their text).
     3. Replace all `{{key}}` patterns with the corresponding value from the JSON data. The JSON keys should match the placeholder names (e.g., `{{report_date}}` matches key `report_date`).
     4. **Remove layout-cache elements from modified paragraphs**: For any paragraph (`<hp:p>` or similar) whose text content was modified, find and remove child elements that represent layout caches. These are typically `<hp:linesegarray>` or `<hp:lineSegArray>` or similar elements. Removing them ensures the document re-renders cleanly without overlapping characters.
     5. Serialize the XML back, preserving the original XML declaration and encoding.
   - If the file does not contain placeholders, copy it as-is (preserving binary content for images, etc.).

d. Write the output as `/root/inventory_report_ready.hwpx`.

**Critical details for the Python script:**
- Use `zipfile.ZipFile` for reading and writing.
- When writing the output ZIP, preserve the compression type of each entry.
- Use `xml.etree.ElementTree` with proper namespace handling. Register namespaces before parsing to avoid `ns0:` prefix pollution. Read the XML files first to identify all namespace URIs and prefixes, then register them with `ET.register_namespace()`.
- Handle the XML declaration line properly (e.g., `<?xml version="1.0" encoding="UTF-8"?>`).
- For JSON values that are numbers, convert them to strings for text replacement.
- Ensure UTF-8 encoding throughout.

### 6. Run the script
```bash
cd /tmp/hwpx_template
python3 process.py
```

### 7. Validate the output
```bash
# Check the output exists and is a valid ZIP
file /root/inventory_report_ready.hwpx
python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_ready.hwpx'); z.testzip(); print('ZIP OK'); z.close()"

# Extract and verify no placeholders remain
mkdir -p /tmp/hwpx_output
cd /tmp/hwpx_output
unzip /root/inventory_report_ready.hwpx -d output_contents
grep -r '{{' output_contents/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'

# Verify Korean labels are preserved (compare structure)
# Show the content XML to visually confirm Korean text is intact and values are filled in
for f in $(grep -rl '재고' output_contents/ 2>/dev/null || find output_contents -name '*.xml' -path '*/section*'); do echo "=== $f ==="; cat "$f"; done

# Verify empty paragraphs are preserved - count paragraphs in input vs output
for f in $(find output_contents -name '*.xml' -path '*/section*' -o -name '*.xml' -path '*/Section*'); do
  echo "Output paragraph count in $f:"
  grep -c '<hp:p ' "$f" 2>/dev/null || grep -c '<hp:p>' "$f" 2>/dev/null || echo 0
done
```

### 8. If validation fails
- If placeholders remain, inspect which ones and check if they were split across XML nodes or if the JSON key names don't match. Fix the script accordingly.
- If the ZIP is invalid, check the script's ZIP writing logic.
- If Korean text is corrupted, check encoding handling.
- Re-run and re-validate after any fix.

**Output file must be at:** `/root/inventory_report_ready.hwpx`

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