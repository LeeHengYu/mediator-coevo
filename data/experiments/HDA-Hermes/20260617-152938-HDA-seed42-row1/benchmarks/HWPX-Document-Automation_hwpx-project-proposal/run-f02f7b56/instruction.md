# Task Instruction

You must produce the file `/root/project_proposal_ready.hwpx` by filling in a template HWPX document with values from a JSON file. Follow these steps precisely:

## 1. Explore the workspace
```bash
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -30
```
Identify the exact paths of `project_proposal_template.hwpx` and `project_proposal.json`.

## 2. Understand the HWPX structure
A `.hwpx` file is a ZIP archive containing XML files. Unzip the template to a temporary directory and inspect its contents:
```bash
mkdir -p /tmp/hwpx_template
cp <path_to_template> /tmp/hwpx_template/template.hwpx
cd /tmp/hwpx_template
unzip template.hwpx -d template_contents
find template_contents -type f
```
Read every XML file inside (especially files like `section0.xml`, `section1.xml`, or any file under `Contents/`). Search for `{{` placeholders across ALL files:
```bash
grep -r '{{' template_contents/
```
Also look at the full text content to understand the document structure, Korean labels, phase lines with date ranges, and the note line.

## 3. Read the JSON data
```bash
cat <path_to_json>
```
Note every key-value pair. Pay special attention to the budget value (it likely has commas that must be removed while keeping the currency symbol like ₩ or $).

## 4. Write a Python script to perform the transformation
Create a Python script `/tmp/fill_template.py` that:

a) **Copies the template HWPX** to the output path.
b) **Opens it as a ZIP**, reads each XML file, performs replacements, and writes a new ZIP.
c) For each XML file in the archive:
   - Replaces every `{{placeholder}}` with the corresponding JSON value.
   - For the budget field: removes commas from the numeric part but keeps the currency symbol (e.g., `₩1,500,000,000` → `₩1500000000`).
   - After placeholder replacement, for lines containing phase information (단계1, 단계2, 단계3), calculates the month span from the date range present in the line and appends it in parentheses. The month span calculation: parse the start and end dates (likely in YYYY.MM.DD or YYYY-MM-DD format), compute the difference in months, and append `(N개월)` after the phase text. Based on the task description: 단계1 → (3개월), 단계2 → (3개월), 단계3 → (1개월). However, compute these from the actual dates if possible; if computation is ambiguous, use the values specified in the requirements.
   - **Removes stale layout-cache elements**: In the HWPX XML, there may be elements like `<hp:linesegarray>` or `<hp:lineSegArray>` or similar layout cache tags within paragraphs you modify. For any `<hp:p>` (paragraph) element whose text content you changed, remove all child elements that represent layout caches. These are typically `<hp:linesegarray>...</hp:linesegarray>` or `<hp:lineSegArray>...</hp:lineSegArray>` blocks. Use an XML parser (lxml or xml.etree.ElementTree) with proper namespace handling to do this cleanly.
   - Preserves all Korean labels and the static note line unchanged.
d) Ensures no `{{...}}` text remains anywhere in any file in the output.
e) Writes the result as a valid ZIP (HWPX) to `/root/project_proposal_ready.hwpx`, preserving the original ZIP structure (directory entries, compression method).

**Critical implementation details:**
- Use `zipfile` module to read/write.
- Use `xml.etree.ElementTree` or `lxml` for XML parsing. Be careful with namespaces — inspect the actual namespace URIs in the XML files and handle them properly.
- When searching for placeholders, search in text nodes of XML elements (both `.text` and `.tail` attributes).
- For non-XML files in the ZIP (like `mimetype`, `META-INF/*`), copy them as-is.
- The `mimetype` entry, if present, should be stored (not compressed) as the first entry.

## 5. Run and validate
```bash
python3 /tmp/fill_template.py
```
Then validate:
```bash
# Check file exists
ls -la /root/project_proposal_ready.hwpx

# Check it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/project_proposal_ready.hwpx'); print(z.namelist())"

# Check no placeholders remain
mkdir -p /tmp/hwpx_output
cd /tmp/hwpx_output
unzip /root/project_proposal_ready.hwpx -d output_contents
grep -r '{{' output_contents/ || echo 'No placeholders found - GOOD'

# Print all text content to verify replacements and month spans
cat output_contents/Contents/section*.xml 2>/dev/null || find output_contents -name '*.xml' -exec cat {} \;
```

Verify specifically:
- All `{{...}}` placeholders are gone
- Budget value has no commas but retains currency symbol
- Phase lines have the parenthesized month spans: `(3개월)`, `(3개월)`, `(1개월)`
- Korean labels and static note line are unchanged
- Layout cache elements are removed from modified paragraphs

## 6. Run the test suite if available
```bash
find /root/ -name 'test_output*' -o -name 'test_*' | head -10
```
If tests exist, run them:
```bash
cd /root && python3 -m pytest tests/ -v 2>&1 || python3 -m pytest test_output.py -v 2>&1
```
Fix any failures before finishing.

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