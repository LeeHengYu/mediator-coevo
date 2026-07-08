# Task Instruction

Complete the following task to fill in a training feedback HWPX template with JSON data.

## Goal
Fill in `training_feedback_template.hwpx` using values from `training_feedback.json`, then save the result to `/root/training_feedback_ready.hwpx`.

## Step-by-step Plan

### 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like OOXML). It contains XML files inside. First, explore the structure:
```bash
cd /root
cp training_feedback_template.hwpx training_feedback_ready.hwpx
mkdir -p hwpx_work
cd hwpx_work
unzip -o ../training_feedback_ready.hwpx
find . -type f | sort
```

### 2. Read the JSON data
```bash
cat /root/training_feedback.json
```
Note all key-value pairs. You will need to map each key to a `{{key}}` placeholder in the XML content.

### 3. Find all placeholders
Search all extracted XML files for `{{` patterns:
```bash
grep -r '{{' . --include='*.xml' -l
grep -r '{{' . --include='*.xml'
```
This tells you which files contain placeholders and what they look like.

### 4. Perform replacements with these specific rules
For each `{{...}}` placeholder, replace it with the corresponding JSON value, applying these transformations:

- **`참석자수`**: Convert to digits only. If the JSON says `"32명"`, write `32`. If it says `32`, write `32`. Strip any non-digit characters.
- **`만족도`**: Rewrite as `X.X점 (5.0점 만점)` style. For example, if JSON has `4.5`, write `4.5점 (5.0점 만점)`. If JSON has `"4.5/5.0"`, extract the score and write `4.5점 (5.0점 만점)`.
- **Overall opinion / 종합의견**: After substituting the JSON comment value, append ` 후속 심화반 검토 요망.` at the end (with a space before it if needed). Make sure the appended text is inside the same text run/paragraph.
- **All other placeholders**: Replace directly with the JSON value as-is.

### 5. Remove stale layout-cache elements
After modifying any paragraph's text content, check if that paragraph (or its parent elements) contains layout-cache elements. In HWPX XML, these are typically elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hc:lineseg>`, or similar cached layout data within or near `<hp:p>` paragraph elements. **Remove all such layout-cache elements from any paragraph whose text you modified.** This prevents overlapping characters when the document is opened.

Specifically, look for and remove:
- `<hp:linesegarray>...</hp:linesegarray>` (case-insensitive tag matching)
- Any element that appears to be a line-segment cache or character-position cache within modified paragraphs

### 6. Repackage the HWPX
After all edits are done:
```bash
cd /root/hwpx_work
zip -r -X ../training_feedback_ready.hwpx . -x '.*'
```
Make sure to use `zip` from within the extracted directory so paths are relative and match the original structure. Use the same compression method as the original if possible.

### 7. Validate
Verify the output:
```bash
# Check it's a valid zip
unzip -t /root/training_feedback_ready.hwpx

# Check no placeholders remain
mkdir -p /root/hwpx_verify
cd /root/hwpx_verify
unzip -o /root/training_feedback_ready.hwpx
grep -r '{{' . --include='*.xml'
```
The grep must return NO results. If any `{{...}}` patterns remain, go back and fix them.

Also verify the specific transformations:
- `참석자수` value is digits only (no Korean unit suffix)
- `만족도` is in `X.X점 (5.0점 만점)` format
- The overall opinion text ends with `후속 심화반 검토 요망.`
- Korean labels and static note lines are unchanged
- Layout cache elements are removed from modified paragraphs

## Important Notes
- Work carefully with the XML. The text content may be split across multiple `<hp:t>` or similar text run elements within a paragraph. A single `{{placeholder}}` might span multiple text runs. If so, you need to consolidate or handle the replacement across runs.
- Use Python for the XML manipulation if sed/awk would be fragile. Python's `lxml` or `xml.etree.ElementTree` with namespace handling would be appropriate.
- Preserve all XML namespaces, attributes, and structure that you don't need to modify.
- The output file must be at exactly `/root/training_feedback_ready.hwpx`.

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