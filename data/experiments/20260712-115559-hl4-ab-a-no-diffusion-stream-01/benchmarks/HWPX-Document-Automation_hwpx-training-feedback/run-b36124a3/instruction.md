# Task Instruction

Execute the following steps in order to fill in the HWPX training feedback template and produce `/root/training_feedback_ready.hwpx`.

## Step 0 – Understand the HWPX format
A `.hwpx` file is a ZIP-based ODF-like package used by Hancom Hangul. Inside, the main body content is typically in XML files under `Contents/` (e.g., `Contents/section0.xml`). The placeholders `{{...}}` will appear as text runs inside these XML files.

## Step 1 – Inspect the workspace
```bash
ls -la /root/
find /root/ -name '*.hwpx' -o -name '*.json' 2>/dev/null
```
Identify the exact paths of `training_feedback_template.hwpx` and `training_feedback.json`.

## Step 2 – Read the JSON data
```bash
cat <path_to>/training_feedback.json
```
Record every key-value pair. Pay special attention to:
- `참석자수` – must be converted to digits only (e.g., "25명" → "25", "스물다섯" → "25").
- `만족도` – must be reformatted as `X.X점 (5.0점 만점)` using the numeric score from JSON.
- The overall-opinion/종합의견 value – the sentence from JSON must have `후속 심화반 검토 요망.` appended (with a space before it if the original doesn't end with one).

## Step 3 – Explore the HWPX package structure
```python
import zipfile, os
template_path = '<path_to>/training_feedback_template.hwpx'
with zipfile.ZipFile(template_path, 'r') as z:
    for info in z.infolist():
        print(info.filename, info.file_size)
```
Then read and print the contents of every XML file inside (especially files under `Contents/`). Identify:
- Which XML files contain `{{` placeholder text.
- The exact XML element structure around each placeholder.
- Any layout-cache elements (e.g., `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hc:lineseg>`, `<hp:parameterset>` with layout-related names, or similar cached glyph/position data within the same paragraph elements that contain placeholders).

## Step 4 – Build the replacement map
Using the JSON values, build a Python dictionary mapping each `{{placeholder}}` string to its replacement value, applying these transformations:
1. **참석자수**: Strip all non-digit characters. If the value is a Korean number word, convert it to Arabic digits.
2. **만족도**: Format as `{score}점 (5.0점 만점)` where `{score}` is the numeric value from JSON (keep one decimal place).
3. **종합의견 / overall opinion**: Take the JSON value and append ` 후속 심화반 검토 요망.` (ensure exactly one space before the appended sentence if the original text doesn't already end with a space; if it ends with a period, add a space then append).
4. All other placeholders: use the JSON values as-is.

## Step 5 – Perform replacements and clean layout caches
Write a Python script that:
1. Copies the template HWPX (ZIP) to `/root/training_feedback_ready.hwpx`.
2. Opens the copy as a ZIP, iterates over all entries.
3. For each XML file that contains `{{`:
   a. Perform all placeholder replacements using the map from Step 4.
   b. **Critical**: Handle the case where a `{{placeholder}}` might be split across multiple XML text runs within the same paragraph. If a simple string replace on the full XML text doesn't find a placeholder, concatenate adjacent text runs, do the replacement, then write the result back into the first run and clear the subsequent runs.
   c. After replacing text in a paragraph, **remove any layout-cache child elements** from that paragraph element. These are typically elements like `<hp:linesegarray>` or `<hp:lineSegArray>` (or any element whose local name contains `lineSeg` or `lineSegArray`). Use an XML parser (lxml or xml.etree.ElementTree) to remove these elements rather than regex, to avoid breaking the XML structure.
4. For non-XML entries, copy them unchanged.
5. Write the result back as a valid ZIP.

## Step 6 – Validate the output
```python
import zipfile
output_path = '/root/training_feedback_ready.hwpx'
# 1. Confirm it's a valid ZIP
with zipfile.ZipFile(output_path, 'r') as z:
    bad = z.testzip()
    assert bad is None, f'Corrupt entry: {bad}'
    # 2. Check no {{...}} placeholders remain
    for info in z.infolist():
        data = z.read(info.filename)
        try:
            text = data.decode('utf-8')
        except:
            continue
        assert '{{' not in text and '}}' not in text, f'Placeholder remains in {info.filename}: {[s for s in text.split("{{") if "}}" in s][:3]}'
    print('All checks passed.')
```

```bash
# 3. Confirm the file exists and has reasonable size
ls -la /root/training_feedback_ready.hwpx
```

## Step 7 – Final content verification
Read and print the text content of the modified XML section files from the output HWPX to visually confirm:
- 참석자수 is digits only (no unit suffix like 명).
- 만족도 follows the `X.X점 (5.0점 만점)` format.
- 종합의견 ends with `후속 심화반 검토 요망.`
- All Korean labels and the static note line are unchanged.
- No `{{` or `}}` remains.

## Important Notes
- Use `lxml` if available, otherwise `xml.etree.ElementTree` for XML parsing.
- When writing the ZIP, preserve the compression type and all original entries.
- Be careful with XML namespaces – use namespace-aware parsing.
- If placeholders span multiple text runs in the XML, you MUST handle run merging; do not assume each placeholder is in a single text node.

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