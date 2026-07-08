# Task Instruction

Execute the following steps in order to fill in the training feedback HWPX template and produce `/root/training_feedback_ready.hwpx`.

## Background
A `.hwpx` file is an OPC/ZIP package (like `.docx`). The actual document content lives in XML files inside the archive, typically under `Contents/` (e.g., `Contents/section0.xml`). Placeholders like `{{교육명}}` appear as text runs inside those XML files. You must find every `{{...}}` placeholder across ALL XML files in the package, replace them with the correct values from the JSON, and repackage the ZIP.

## Step-by-step

### 1. Read the JSON data
```bash
cat /root/training_feedback.json
```
Parse and understand every key-value pair.

### 2. Inspect the HWPX package structure
```bash
python3 -c "
import zipfile, sys
with zipfile.ZipFile('/root/training_feedback_template.hwpx','r') as z:
    for name in z.namelist():
        print(name)
"
```
Identify all files, especially XML content files.

### 3. Find all placeholders
For every file in the ZIP, search for `{{` to locate placeholders. Print the filename and the surrounding text:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/training_feedback_template.hwpx','r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8')
        except:
            continue
        if '{{' in data:
            print(f'=== {name} ===')
            # Print lines containing placeholders
            for i, line in enumerate(data.split('\n')):
                if '{{' in line:
                    print(f'  Line {i}: ...{line[max(0,line.index("{{")-80):line.index("{{")+120]}...')
"
```

**CRITICAL**: Placeholders may be split across multiple XML text runs (e.g., `<hp:t>{{교육</hp:t><hp:t>명}}</hp:t>`). You MUST handle this. After reading each XML file's full text, check if `{{` and `}}` appear in the raw XML. If a placeholder is split across tags, you need to merge the text runs or handle the replacement at the raw XML string level carefully.

### 4. Build the replacement map
From the JSON, build a Python dictionary mapping placeholder names to final display values. Apply these transformations:

- **`참석자수`**: Convert to digits only. E.g., if JSON says `"32명"`, write `"32"`. If JSON says `"32"`, keep `"32"`. Strip any non-digit characters.
- **`만족도`**: Rewrite as `"X.X점 (5.0점 만점)"` format. E.g., if JSON value is `4.5` or `"4.5"`, output `"4.5점 (5.0점 만점)"`.
- **Overall opinion / `종합의견` (or similar)**: Take the JSON comment value and append ` 후속 심화반 검토 요망.` at the end (with a space before `후속`).
- All other values: use as-is from the JSON.

### 5. Perform replacements carefully
Write a Python script that:
1. Opens the template HWPX as a ZIP.
2. For each file in the ZIP:
   a. If it's an XML file containing `{{`, read it as UTF-8 text.
   b. **Handle split placeholders**: Use regex to find all `{{...}}` patterns even when XML tags are interspersed. Strategy: first, extract a "tag-stripped" version of the text to find placeholder boundaries, then perform replacements on the original XML. Alternatively, a simpler approach: remove all XML tags temporarily, find placeholders, then map back. **Recommended approach**: Use regex `r'\{\{[^}]*\}\}'` on the raw XML first. If that doesn't find all expected placeholders, then use a more aggressive approach: collect all text content between `{{` and `}}` even across tags.
   c. Replace each `{{placeholder_name}}` with the transformed value.
   d. **Remove stale layout-cache elements**: For any `<hp:linesegarray>...</hp:linesegarray>` or `<hp:lineSegArray>...</hp:lineSegArray>` elements within paragraphs whose text was modified, remove them entirely. This prevents overlapping characters when opening the document. Use regex: `r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>'` with `re.DOTALL` flag, but only in paragraphs that contained placeholders.
   e. Similarly remove any `<hp:lineseg .../>` or `<hp:lineSeg .../>` self-closing tags in modified paragraphs.
3. Write all files (modified and unmodified) to the output ZIP at `/root/training_feedback_ready.hwpx`, preserving compression method.

### 6. Validate the output
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/training_feedback_ready.hwpx','r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8')
        except:
            continue
        if '{{' in data:
            print(f'FAIL: placeholder remains in {name}')
            import re
            for m in re.finditer(r'\{\{[^}]*\}\}', data):
                print(f'  Found: {m.group()}')
        else:
            if name.endswith('.xml'):
                print(f'OK: {name} - no placeholders')
print('Validation complete')
"
```

Also verify the file is a valid ZIP:
```bash
python3 -c "import zipfile; print('Valid ZIP:', zipfile.is_zipfile('/root/training_feedback_ready.hwpx'))"
```

### 7. Verify specific content transformations
Extract text from the output XML and verify:
- `참석자수` value is digits only (no `명` suffix or other text)
- `만족도` appears as `X.X점 (5.0점 만점)`
- The overall opinion ends with `후속 심화반 검토 요망.`
- Korean labels are intact
- The static note line is unchanged

## Key Warnings
- Do NOT change Korean label text (e.g., `교육명:`, `강사:`, etc.) — only replace the `{{...}}` placeholder portions.
- Handle the case where placeholders span multiple XML text runs.
- Remove `lineSegArray` / `lineseg` cache elements from modified paragraphs to prevent rendering issues.
- Preserve the ZIP structure exactly (same filenames, same directory structure).
- If the template has `mimetype` or `META-INF` files, preserve them as-is.
- Use `zipfile.ZIP_DEFLATED` for XML files and match original compression for other files.

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