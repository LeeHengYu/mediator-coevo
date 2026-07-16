# Task Instruction

## Task: Fill in training feedback HWPX template

### Goal
Fill in the training feedback sheet `training_feedback_template.hwpx` using values from `training_feedback.json`, then save the result to `/root/training_feedback_ready.hwpx`.

### Step-by-step Plan

#### 1. Explore the workspace
```bash
find /root -maxdepth 3 -type f | head -80
```
Identify the location of `training_feedback_template.hwpx` and `training_feedback.json`.

#### 2. Read the JSON data
```bash
cat <path>/training_feedback.json
```
Note every key-value pair. You will need all of them.

#### 3. Understand the HWPX structure
A `.hwpx` file is a ZIP archive. Unzip it to a temporary working directory:
```bash
mkdir -p /tmp/hwpx_work
cp <path>/training_feedback_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip template.hwpx -d template_extracted
```
List all extracted files. The main content is typically in `Contents/section0.xml` (or similar). There may also be other XML files.

#### 4. Find all placeholders
Search every XML file for `{{` patterns:
```bash
grep -rn '{{' /tmp/hwpx_work/template_extracted/
```
Document every placeholder found and which file it's in. Map each `{{...}}` to the corresponding JSON key.

#### 5. Understand the XML structure around each placeholder
For each file containing placeholders, read the full file content. Pay attention to:
- The XML namespace declarations
- How text runs (`<hp:t>`, `<w:t>`, or similar) contain the placeholder text
- Any layout-cache elements (`<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:layoutCache>`, `<hp:charPr>` with positioning, etc.) associated with paragraphs containing placeholders

#### 6. Write a Python script to perform all replacements
Create `/tmp/hwpx_work/fill_template.py` that:

a. Reads `training_feedback.json`.

b. Extracts the template HWPX (ZIP) to a working directory.

c. For each XML file that contains `{{...}}` placeholders, parses it with `xml.etree.ElementTree` (preserving namespaces) and performs replacements.

d. **Replacement rules** (apply these transformations to the JSON values before substituting):
   - `참석자수` (attendee count): Convert to digits only. E.g., if JSON has `"25명"` or `"25"`, output just `"25"` (strip any non-digit characters, or if it's already a number, convert to string of digits).
   - `만족도` (satisfaction): Rewrite as `"X.X점 (5.0점 만점)"` format, where X.X is the numeric score from JSON. E.g., if JSON has `4.5` or `"4.5/5.0"`, output `"4.5점 (5.0점 만점)"`.
   - For the overall opinion field: take the value from JSON and append ` 후속 심화반 검토 요망.` at the end (with a space before it if the original doesn't end with a space). Make sure this is the final overall-opinion sentence in the document.
   - All other placeholders: substitute the JSON value directly.

e. **Critical: Handle split placeholders.** HWPX/OOXML-style editors often split `{{placeholder}}` across multiple text runs (e.g., `<hp:t>{{</hp:t>`, `<hp:t>placeholder</hp:t>`, `<hp:t>}}</hp:t>`). The script must:
   - Collect consecutive text elements within a paragraph
   - Join their text content to find `{{...}}` patterns
   - When a match is found spanning multiple runs, put the replacement value in the first run and clear the others

f. **Critical: Remove stale layout-cache elements.** For any paragraph (`<hp:p>` or equivalent) where text was modified, remove child elements that are layout caches. Look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:LineSeg>`, `<hp:layoutCache>`, or any element whose tag contains `lineseg` or `LineSeg` (case-insensitive). This prevents overlapping characters when the document is opened.

g. Write the modified XML back, preserving the XML declaration and encoding.

h. Re-pack the modified directory into a valid ZIP file at `/root/training_feedback_ready.hwpx`, preserving the original directory structure and `mimetype` file (if present, store it uncompressed as the first entry, similar to ODF/EPUB conventions).

#### 7. Run the script
```bash
python3 /tmp/hwpx_work/fill_template.py
```

#### 8. Validate the output

a. Verify it's a valid ZIP:
```bash
unzip -t /root/training_feedback_ready.hwpx
```

b. Check NO placeholders remain:
```bash
unzip -o /root/training_feedback_ready.hwpx -d /tmp/hwpx_verify
grep -rn '{{' /tmp/hwpx_verify/
```
This must return nothing.

c. Verify the specific transformations:
```bash
# Check 참석자수 is digits only
grep -r '참석자수' /tmp/hwpx_verify/ | head -5
# Check 만족도 format
grep -r '만족도\|점 (5.0점 만점)' /tmp/hwpx_verify/ | head -5  
# Check the appended sentence
grep -r '후속 심화반 검토 요망' /tmp/hwpx_verify/ | head -5
# Check Korean labels are preserved
grep -r '교육명\|교육일시\|교육장소' /tmp/hwpx_verify/ | head -5
```

d. Verify no `lineseg` or layout cache elements remain in modified paragraphs:
```bash
grep -ri 'lineseg\|layoutcache' /tmp/hwpx_verify/Contents/ | head -10
```
Ideally these should be absent from any paragraph that was modified. (They may exist in unmodified paragraphs - that's OK.)

#### 9. Run any existing test/verifier
```bash
# Check if there's a test file
find /root -name 'test_*' -o -name '*_test.*' | head -10
# If found, run it
cd /root && python3 -m pytest test_output.py -v 2>&1 || python3 -m pytest -v 2>&1
```

### Key Pitfalls to Avoid
- **Split text runs**: `{{placeholder}}` may be split across multiple XML text elements. You MUST handle this by joining text across runs within a paragraph before matching.
- **Namespace handling**: Use namespace-aware XML parsing. Register all namespaces found in the document before parsing/writing to avoid `ns0:` prefix pollution.
- **Layout cache removal**: Only remove from paragraphs you actually modified. Don't strip them globally (that could break unmodified content).
- **ZIP structure**: The output must be a proper ZIP. Use Python's `zipfile` module. If there's a `mimetype` file, add it first with `ZIP_STORED` compression.
- **Encoding**: HWPX XML files are typically UTF-8. Preserve this encoding.
- **No leftover placeholders**: Double-check by grepping the final output for `{{`.
- **String formatting**: From cross-task feedback, be precise with number formatting (commas, decimal places) to match expected patterns exactly.

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