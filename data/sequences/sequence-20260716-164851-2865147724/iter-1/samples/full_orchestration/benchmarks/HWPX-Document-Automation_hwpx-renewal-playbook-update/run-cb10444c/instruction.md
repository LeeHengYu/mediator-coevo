# Task Instruction

## Task: Update renewal_playbook.hwpx with new data and save to /root/renewal_playbook_updated.hwpx

### Overview
You need to update a `.hwpx` document (which is a ZIP-based OOXML-like Korean word-processor package) using data from `renewal_update.json` and `followups.csv`, then save the result as `/root/renewal_playbook_updated.hwpx`.

### Step 0: Explore the workspace
```bash
find /root -maxdepth 2 -type f | head -60
ls -la /root/
```
Identify the location of `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.

### Step 1: Examine the input data files
1. `cat` the `renewal_update.json` file completely. Note every field name and value — these are the replacement values.
2. `cat` the `followups.csv` file completely. Note the columns; there should be a `sequence` column that determines ordering.

### Step 2: Unpack and inspect the HWPX package
```bash
mkdir -p /tmp/hwpx_work
cp <path_to>/renewal_playbook.hwpx /tmp/hwpx_work/original.hwpx
cd /tmp/hwpx_work
mkdir extracted
cd extracted
unzip ../original.hwpx
find . -type f
```
HWPX files are ZIP archives containing XML files (typically under `Contents/` with section XML files like `section0.xml`). Identify ALL XML files, especially section XMLs.

### Step 3: Read and understand the section XML(s)
For each section XML file found (e.g., `Contents/section0.xml`):
```bash
cat Contents/section0.xml
```
Read the FULL content. Identify:
- The XML structure (namespaces, paragraph elements, text run elements)
- Where the customer name, current owner, renewal window, pricing band, escalation contact, and pricing note appear
- Where the three follow-up lines are
- Where the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` appears (DO NOT modify this)
- Any layout-cache elements (e.g., `<linesegarray>`, `<lineSegArray>`, `<hp:linesegarray>`, or similar elements containing cached glyph/position data)

### Step 4: Write a Python script to perform the updates
Write a Python script `/tmp/hwpx_work/update_hwpx.py` that:

1. **Copies** the original hwpx to the output path first (to preserve the ZIP structure).
2. **Reads** `renewal_update.json` to get replacement values.
3. **Reads** `followups.csv` and sorts rows by the `sequence` column.
4. **Parses** each section XML using `xml.etree.ElementTree` with proper namespace handling.
5. **Performs text replacements** for all fields from the JSON:
   - For each field in the JSON, find the OLD value currently in the document and replace it with the NEW value.
   - To find old values: inspect the XML text content carefully. The JSON likely has keys like `customer_name`, `current_owner`, `renewal_window`, `pricing_band`, `escalation_contact`, `pricing_note` with new values. You need to identify the CURRENT (old) values in the XML by reading the document content.
   - Replace ALL occurrences in ALL editable sections.
6. **Replaces follow-up lines**: Identify the three existing follow-up text lines in the XML. Replace them with the CSV items in `sequence` order. The number of follow-up items in the CSV may differ from 3 — adjust paragraph elements accordingly (add or remove paragraph elements as needed, cloning structure from existing ones).
7. **Removes layout-cache elements**: For ANY paragraph whose text content was modified, remove child elements that represent layout caches. These are typically `<linesegarray>` or similar elements (check the actual namespace and tag name in the XML). This is CRITICAL — the requirement says edited paragraphs must not retain stale layout-cache elements.
8. **Preserves** the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` exactly as-is.
9. **Writes** the modified XML back, preserving XML declaration and encoding.
10. **Rebuilds** the HWPX ZIP at `/root/renewal_playbook_updated.hwpx` by replacing the modified XML(s) in the archive while keeping all other files intact.

**CRITICAL implementation notes:**
- Register all XML namespaces BEFORE parsing to avoid namespace prefix mangling in output. Use `xml.etree.ElementTree.register_namespace()` for each namespace found.
- When writing XML back, preserve the original encoding declaration.
- Use `zipfile` module to rebuild: iterate over original ZIP entries, for non-modified entries copy bytes directly, for modified XML entries write the new content.
- Ensure exact parentheses, spacing, and formatting match what the verifier expects. Do NOT strip parentheses or alter formatting of replacement values from the JSON.
- When replacing text in XML runs, handle cases where text may be split across multiple `<t>` or text-run elements within a paragraph. If a value spans multiple runs, you may need to consolidate or handle carefully.

### Step 5: Run the script
```bash
cd /tmp/hwpx_work
python3 update_hwpx.py
```

### Step 6: Validate the output
1. Verify the output file exists and is a valid ZIP:
```bash
ls -la /root/renewal_playbook_updated.hwpx
python3 -c "import zipfile; z=zipfile.ZipFile('/root/renewal_playbook_updated.hwpx'); print(z.namelist()); z.close()"
```
2. Extract and inspect the updated section XML(s):
```bash
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify
unzip /root/renewal_playbook_updated.hwpx
cat Contents/section0.xml
```
3. Verify:
   - All JSON field values appear in the updated XML
   - Follow-up lines match CSV items in sequence order
   - Old values do NOT appear (no duplicates)
   - Appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged
   - No stale layout-cache elements remain on modified paragraphs
   - The file is a valid ZIP

### Step 7: Run the verifier test if present
```bash
find /root -name 'test_*' -o -name '*test*.py' | head -10
# If a test file exists, run it:
python3 -m pytest <test_file> -v
```
If any test fails, read the error carefully, fix the issue, and re-run.

### Important Warnings (from cross-task feedback)
- Do NOT alter formatting of replacement values. If the JSON says `High (즉시조치)`, keep the parentheses exactly. Same for any value with special characters.
- Be extremely careful with namespace handling in XML. Print the root tag and all namespaces before making changes.
- After editing, always verify the output by reading it back and checking for both new values and absence of old values.

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