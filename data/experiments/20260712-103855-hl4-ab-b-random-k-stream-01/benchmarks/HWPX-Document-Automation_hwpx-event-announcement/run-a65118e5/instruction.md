# Task Instruction

Complete the following task to prepare an event announcement HWPX document.

## Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, and save the result to `/root/event_announcement_ready.hwpx`.

## Step-by-step Plan

### Step 1: Inspect the workspace
```bash
ls -la /root/
```
Locate `event_announcement_template.hwpx` and `event_data.json`. They may be in `/root/` or a subdirectory. Find them.

### Step 2: Read the JSON data
```bash
cat <path_to>/event_data.json
```
Note every key-value pair. These keys correspond to `{{key}}` placeholders in the template.

### Step 3: Understand the HWPX package structure
A `.hwpx` file is a ZIP archive. Unzip it to a temporary directory to inspect its contents:
```bash
mkdir -p /tmp/hwpx_work
cp <path_to>/event_announcement_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```
List all files inside the package to understand the structure.

### Step 4: Find all files containing placeholders
```bash
grep -rl '{{' template_contents/
```
This identifies which XML files contain `{{...}}` placeholders.

### Step 5: Examine the XML files with placeholders
For each file found in Step 4, read its contents carefully:
```bash
cat <file>
```
Pay special attention to:
- How placeholders appear (they may be split across multiple XML text runs/spans)
- Layout-cache elements such as `<hp:linesegarray>`, `<hp:lineseg>`, `<hp:charshapeidarray>`, or similar cached layout data within paragraphs
- The overall XML namespace and structure

### Step 6: Write a Python script to perform the replacements
Create a Python script `/tmp/hwpx_work/process.py` that:

1. Reads `event_data.json` to get the replacement values.
2. Copies the original `.hwpx` file (which is a ZIP) and processes it.
3. For each XML file inside the ZIP that contains `{{...}}` patterns:
   a. Reads the XML content.
   b. **CRITICAL**: Placeholders like `{{event_name}}` might be split across multiple XML text run elements (e.g., `<hp:t>{{</hp:t><hp:t>event_name</hp:t><hp:t>}}</hp:t>`). The script must handle this by:
      - First, concatenating all text content within a paragraph to find placeholders
      - Then performing replacements at the paragraph level, consolidating split runs if needed
      - OR: Working at the raw XML text level if placeholders are not split across tags
      - Check the actual XML first to determine which approach is needed
   c. Replaces every `{{key}}` with the corresponding value from the JSON.
   d. **CRITICAL**: For any paragraph (`<hp:p>` element or equivalent) whose text content was modified, removes stale layout-cache child elements. These are typically elements like `<hp:linesegarray>` (and its children `<hp:lineseg>`), or `<hp:charshapeidarray>` — any element that caches glyph positions or line-break info. Inspect the actual XML to identify the exact element names. Remove them from modified paragraphs only.
   e. Verifies no `{{...}}` patterns remain in the output.
4. Writes the modified files back into a new ZIP archive saved as `/root/event_announcement_ready.hwpx`.
5. The ZIP must preserve the original directory structure, file ordering if possible, and not add extra compression artifacts.

IMPORTANT considerations for the script:
- Use `zipfile` module in Python.
- Preserve all files that don't need modification (copy them as-is).
- Use UTF-8 encoding for reading/writing XML files.
- Keep all Korean labels and static note lines unchanged — only replace `{{...}}` patterns.
- After replacement, do a final scan of ALL text content in ALL XML files to confirm zero `{{` or `}}` patterns remain.

### Step 7: Run the script
```bash
cd /tmp/hwpx_work
python3 process.py
```

### Step 8: Validate the output
```bash
# Check the output file exists
ls -la /root/event_announcement_ready.hwpx

# Verify it's a valid ZIP
unzip -t /root/event_announcement_ready.hwpx

# Check no placeholders remain
mkdir -p /tmp/hwpx_verify
unzip /root/event_announcement_ready.hwpx -d /tmp/hwpx_verify
grep -r '{{' /tmp/hwpx_verify/ || echo 'No placeholders found - GOOD'

# Inspect the modified XML to confirm replacements are correct
# and layout-cache elements were removed from modified paragraphs
grep -rl 'lineseg\|charshapeid' /tmp/hwpx_verify/ | head -5
# Compare with original to see if cache elements were properly removed from modified paragraphs
```

### Step 9: Final content verification
Read the main content XML from the output package and verify:
- All JSON values appear correctly in the document
- Korean labels are preserved
- The static note line is unchanged
- No `{{...}}` placeholders remain

## Key Pitfalls to Avoid
- Do NOT assume placeholder text is contiguous in XML. Inspect the actual XML structure first.
- Do NOT remove layout-cache elements from paragraphs that were NOT modified.
- Do NOT alter Korean text, static notes, or any content outside of `{{...}}` placeholders.
- Do NOT use `str.replace()` blindly on XML — be aware of XML entities and encoding.
- Ensure the output `.hwpx` is written as a proper ZIP file, not just renamed.

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