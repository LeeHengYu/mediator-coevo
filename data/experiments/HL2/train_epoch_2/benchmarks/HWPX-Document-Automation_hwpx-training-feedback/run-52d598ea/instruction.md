# Task Instruction

Complete the following task to fill in a HWPX training feedback template and save the result.

## Goal
Fill in `training_feedback_template.hwpx` using values from `training_feedback.json`, then save to `/root/training_feedback_ready.hwpx`.

## Step-by-step Instructions

### Step 1: Inspect the input files
1. Find the template file `training_feedback_template.hwpx` and the JSON file `training_feedback.json` in the working directory (check `/root/` or current directory).
2. Read and display the JSON file to understand all key-value pairs.
3. Since HWPX is a ZIP archive, list its contents: `python3 -c "import zipfile; z=zipfile.ZipFile('training_feedback_template.hwpx'); print(z.namelist())"`
4. Extract and display all XML files inside the HWPX archive (especially files under `Contents/` like `section0.xml`, `content.hpf`, etc.) to find where `{{...}}` placeholders appear.

### Step 2: Write a Python script to perform the replacement
Create a Python script (`fill_template.py`) that does the following:

#### 2a: Extract the HWPX archive
- Copy the template to the output path first.
- Use `zipfile` to read all entries.

#### 2b: For each XML file in the archive
- Parse with `xml.etree.ElementTree`, registering all namespaces properly first (critically important: scan the XML for namespace declarations and register them with `ET.register_namespace` BEFORE parsing, to preserve prefixes like `hp:`, `hc:`, etc.).
- For namespace registration, read the raw XML bytes first, use regex to find all `xmlns:prefix="uri"` declarations, and register each one.

#### 2c: Merge fragmented text runs in paragraphs
- HWPX often splits placeholder text like `{{교육명}}` across multiple `<hp:t>` tags (e.g., `{{`, `교육명`, `}}`). Before doing replacements:
  - For each paragraph element (e.g., `<hp:p>`), collect all `<hp:t>` text nodes within that paragraph.
  - Concatenate them into a single string.
  - Check if this concatenated string contains any `{{...}}` placeholder.
  - If it does, put all the concatenated text into the FIRST `<hp:t>` element and clear/remove the remaining `<hp:t>` elements' text (or remove the extra `<hp:run>` elements if safe, but at minimum clear the text from subsequent `<hp:t>` tags).

#### 2d: Apply replacements with transformations
For each `{{key}}` placeholder found in the merged text:
- Look up `key` in the JSON data.
- Apply these specific transformations:
  - **참석자수**: Extract digits only from the JSON value (e.g., "42명" → "42", or if it's already a number, convert to string of digits).
  - **만족도**: Format as `X.X점 (5.0점 만점)` where X.X is the numeric score from JSON. For example, if JSON has `4.5` or `"4.5"`, output `4.5점 (5.0점 만점)`.
  - **종합의견** (or whatever key represents the overall opinion): After substituting the JSON value, append ` 후속 심화반 검토 요망.` to the end. Make sure there's a space before `후속` if the original comment doesn't end with one.
  - **All other keys**: Direct substitution with the JSON string value.

#### 2e: Remove layout cache from modified paragraphs
- For any `<hp:p>` paragraph where text was modified, find and REMOVE all `<hp:linesegarray>` child elements (and their children). This prevents stale layout data from causing overlapping characters when the document is opened.

#### 2f: Write back the modified XML files into a new HWPX ZIP
- Create the output HWPX file at `/root/training_feedback_ready.hwpx`.
- Copy all non-modified entries from the original ZIP as-is (preserving binary files like images).
- For modified XML files, serialize with `ET.tostring()` using `xml_declaration=True` and `encoding='utf-8'`.
- Ensure the XML declaration matches the original (typically `<?xml version="1.0" encoding="UTF-8"?>`).

### Step 3: Run the script
```bash
cd /root && python3 fill_template.py
```

### Step 4: Verify the output
1. List the contents of the output HWPX to confirm it's a valid ZIP:
   ```bash
   python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx'); print(z.namelist())"
   ```
2. Extract and search ALL XML files in the output for any remaining `{{` markers:
   ```bash
   python3 -c "
import zipfile
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx')
for name in z.namelist():
    if name.endswith('.xml') or name.endswith('.hpf'):
        content = z.read(name).decode('utf-8', errors='replace')
        if '{{' in content:
            print(f'FAIL: {{{{ found in {name}')
            # Print context around the marker
            idx = content.find('{{')
            print(content[max(0,idx-50):idx+50])
        else:
            print(f'OK: {name}')
"
   ```
3. Verify the specific transformations were applied correctly:
   - Check that 참석자수 appears as digits only (no Korean unit suffix).
   - Check that 만족도 appears in the `X.X점 (5.0점 만점)` format.
   - Check that the overall opinion sentence ends with `후속 심화반 검토 요망.`
   - Check that Korean labels and static note lines are unchanged.
   - Check that no `<hp:linesegarray>` elements remain in paragraphs that were modified.
4. Verify no `{{` or `}}` markers remain anywhere.

### Step 5: If any verification fails
- Re-read the problematic XML section from the output.
- Identify the root cause (fragmented runs not merged, wrong key name, transformation not applied).
- Fix the script and re-run.

## Important Notes
- The JSON keys may use slightly different names than the placeholder names. Inspect both carefully and map them correctly.
- Some placeholders might appear in multiple XML files (e.g., header/footer sections). Check ALL XML files.
- Namespace handling is critical. Always register namespaces before parsing to avoid prefix changes.
- The script must handle the case where a placeholder spans multiple `<hp:run>` elements within a single `<hp:p>`, not just multiple `<hp:t>` within one `<hp:run>`.
- When removing extra text after merging, be careful not to break the XML structure.

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