# Task Instruction

Complete the following task step-by-step:

## Goal
Fill in the training feedback sheet `training_feedback_template.hwpx` using the values in `training_feedback.json`, then save the result to `/root/training_feedback_ready.hwpx`.

## Steps

### 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like `.docx`). It contains XML files inside. The text content is typically in XML files under a path like `Contents/` (e.g., `Contents/section0.xml` or similar). Explore the structure first.

### 2. Explore the workspace
```bash
find /root -type f -name '*.hwpx' -o -name '*.json' 2>/dev/null
```
Locate `training_feedback_template.hwpx` and `training_feedback.json`. Read the JSON file to understand all available values.

### 3. Inspect the HWPX package structure
```bash
mkdir -p /tmp/hwpx_work
cp <path_to_template> /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -l template.hwpx
```
List all files in the archive. Then extract:
```bash
unzip -o template.hwpx -d template_extracted
```

### 4. Find all placeholder locations
```bash
grep -rn '{{' template_extracted/
```
This will show every `{{...}}` placeholder and which XML files contain them. Document all placeholders found.

### 5. Read the JSON data
```bash
cat <path_to_json>
```
Map each `{{placeholder_name}}` to its corresponding JSON value.

### 6. Perform replacements with special rules
Write a Python script to do all replacements. The script must:

a) **Parse the JSON** file to get all values.

b) **For each XML file** in the extracted HWPX that contains `{{...}}` placeholders, read it and perform substitutions.

c) **Special transformations:**
   - `참석자수` (attendee count): Convert to digits only. E.g., if JSON has `"25명"` or `"25"`, write just `"25"` (digits only, no unit suffix like 명).
   - `만족도` (satisfaction): Rewrite as `X.X점 (5.0점 만점)` style. E.g., if JSON has `4.5`, write `4.5점 (5.0점 만점)`.
   - **Overall opinion sentence**: Find the placeholder for the overall comment/opinion. After substituting the JSON value, append ` 후속 심화반 검토 요망.` (with a space before it) to that sentence.
   - All other placeholders: substitute the JSON value directly.

d) **Remove stale layout-cache elements**: For any `<hp:linesegarray>` (or similar line-segment/layout-cache elements) that belong to paragraphs whose text was modified, remove them entirely. These are layout cache elements that, if left stale, cause overlapping characters when the document is opened. Specifically:
   - In HWPX XML, paragraphs (`<hp:p>`) may contain `<hp:linesegarray>` or `<hp:lineSegArray>` child elements. If the paragraph's text run (`<hp:t>`, `<hp:run>`, etc.) was modified (i.e., contained a placeholder), remove the `<hp:linesegarray>`/`<hp:lineSegArray>` element from that paragraph.
   - Use an XML parser (like `xml.etree.ElementTree` or `lxml`) rather than regex for this structural removal to avoid breaking the XML.

e) **Verify no `{{` remains** in any XML file after substitution.

f) **Keep all Korean labels and static note lines unchanged** — only replace placeholder tokens.

### 7. Repackage the HWPX
After modifying the XML files in place within the extracted directory, repackage into a valid HWPX (ZIP) file:
```bash
cd /tmp/hwpx_work/template_extracted
zip -r /root/training_feedback_ready.hwpx . -x '*.DS_Store'
```
IMPORTANT: The ZIP must be created from inside the extracted directory so that paths are relative (e.g., `Contents/section0.xml`, not `template_extracted/Contents/section0.xml`). Also ensure `mimetype` file (if present) is stored first and uncompressed, as is standard for OPC-like packages:
```bash
cd /tmp/hwpx_work/template_extracted
if [ -f mimetype ]; then
  zip -0 /root/training_feedback_ready.hwpx mimetype
  zip -r /root/training_feedback_ready.hwpx . -x mimetype -x '*.DS_Store'
else
  zip -r /root/training_feedback_ready.hwpx . -x '*.DS_Store'
fi
```

### 8. Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/training_feedback_ready.hwpx

# Check no placeholders remain
unzip -p /root/training_feedback_ready.hwpx | grep -c '{{'
# Should output 0

# Verify the special transformations are present
unzip -p /root/training_feedback_ready.hwpx | grep '점 (5.0점 만점)'
unzip -p /root/training_feedback_ready.hwpx | grep '후속 심화반 검토 요망'

# Verify attendee count is digits only (check the context around it)
unzip -p /root/training_feedback_ready.hwpx | grep -oP '참석자수.*?\d+'
```

### 9. Final check
Confirm `/root/training_feedback_ready.hwpx` exists and is non-empty:
```bash
ls -la /root/training_feedback_ready.hwpx
```

## Critical Reminders
- The HWPX XML namespace handling is important. When parsing XML with ElementTree, register namespaces before parsing to avoid namespace prefix mangling on output. Use `lxml` if available, or carefully handle namespaces with `xml.etree.ElementTree`.
- When writing XML back, preserve the original encoding declaration and XML declaration.
- The `linesegarray` / `lineSegArray` removal is essential — any paragraph you touched must have its layout cache stripped.
- Do NOT modify paragraphs that don't contain placeholders.
- The placeholder pattern is `{{...}}` — match with regex `\{\{[^}]+\}\}` to find all of them.
- Be careful that placeholders might be split across multiple XML text nodes within a single run or across runs. If `grep` finds `{{` in a file, also check whether the placeholder text is contiguous in one text node or split. If split, you'll need to join text nodes in the same paragraph/run before replacing.

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