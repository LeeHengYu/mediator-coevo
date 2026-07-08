# Task Instruction

Complete the project proposal HWPX document by following these steps precisely:

## Step 1: Inspect the workspace
```bash
ls -la /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```

## Step 2: Examine the JSON data file
```bash
cat /root/project_proposal.json
```

## Step 3: Examine the HWPX template structure
```bash
mkdir -p /tmp/hwpx_inspect
cp /root/project_proposal_template.hwpx /tmp/hwpx_inspect/
cd /tmp/hwpx_inspect
unzip -l project_proposal_template.hwpx
```

## Step 4: Extract and inspect the XML content files
```bash
unzip -o project_proposal_template.hwpx -d template_contents/
find template_contents/ -name '*.xml' | while read f; do echo "=== $f ==="; cat "$f"; echo; done
```
Pay special attention to:
- Which XML files contain `{{...}}` placeholders
- The namespace declarations used (especially `hp:` namespace)
- The structure of text runs within paragraphs
- Any `<hp:lineSegArray>` or similar layout-cache elements
- The phase lines (단계1, 단계2, 단계3) and their date range formats

## Step 5: Write a Python script to process the HWPX

Create `/root/process_hwpx.py` that does the following:

1. **Load JSON data** from `/root/project_proposal.json`.

2. **Extract the HWPX** (ZIP) template into a temporary directory, preserving all entries.

3. **For each XML file** in the extracted contents (especially section XML files under `Contents/`):
   a. Parse the XML preserving all namespaces.
   b. For each paragraph element, concatenate all text content from its child run elements to form the full paragraph text.
   c. Check if the concatenated text contains any `{{...}}` placeholders.
   d. If placeholders are found, replace them with the corresponding JSON values. **Important**: Since placeholders may be split across multiple `<hp:t>` elements within runs, you must:
      - Collect all `<hp:t>` elements in the paragraph
      - Concatenate their text to find placeholders
      - After replacement, put the entire replaced text into the first `<hp:t>` element and clear/remove the remaining ones (or set their text to empty string)
   e. **Budget normalization**: When replacing a budget placeholder, remove commas from the numeric value but keep the leading currency symbol (e.g., `₩1,500,000,000` becomes `₩1500000000`).
   f. **Month span appending**: For lines containing 단계1, 단계2, or 단계3 with date ranges, calculate the month span from the date range in that line and append ` (N개월)` at the end. To calculate months:
      - Parse the start and end dates from the line (likely in format like `2024.01 ~ 2024.03` or `2024-01 ~ 2024-03`)
      - Calculate the difference in months (end_month - start_month, accounting for year differences, inclusive counting if needed — check the expected values: 단계1 -> 3개월, 단계2 -> 3개월, 단계3 -> 1개월)
   g. **Remove layout-cache elements**: For any paragraph whose text was modified, remove `<hp:lineSegArray>` elements and any other layout-cache child elements (like `<hp:lineseg>` arrays) so the document opens cleanly.

4. **Verify no `{{...}}` placeholders remain** in any XML file.

5. **Re-zip** all contents back into a valid HWPX file at `/root/project_proposal_ready.hwpx`, preserving the original ZIP structure (directory entries, compression method).

## Step 6: Run the script
```bash
cd /root && python3 process_hwpx.py
```

## Step 7: Validate the output
```bash
# Check it's a valid ZIP/HWPX
unzip -l /root/project_proposal_ready.hwpx

# Extract and check for remaining placeholders
mkdir -p /tmp/hwpx_verify
unzip -o /root/project_proposal_ready.hwpx -d /tmp/hwpx_verify/
grep -r '{{' /tmp/hwpx_verify/ || echo 'No placeholders remaining - GOOD'

# Check the content of section XML files to verify replacements
find /tmp/hwpx_verify/ -name '*.xml' | while read f; do echo "=== $f ==="; cat "$f"; echo; done

# Verify month spans are present
grep -r '개월' /tmp/hwpx_verify/ || echo 'WARNING: No month spans found'

# Verify budget has no commas
grep -r ',' /tmp/hwpx_verify/Contents/ | grep -i 'budget\|예산\|₩' || echo 'Budget appears normalized'
```

## Step 8: Run the verifier if available
```bash
cd /root && ls test_*.py 2>/dev/null && python3 -m pytest test_*.py -v
```

## Key Reminders:
- **Namespace handling**: Register all namespaces before parsing/writing XML to avoid ns0/ns1 prefix pollution. Use `ET.register_namespace()` for each namespace found.
- **Placeholder splitting**: HWPX editors often split text across multiple `<hp:t>` elements even within a single word. Always concatenate paragraph text before matching placeholders.
- **Layout cache removal**: Only remove `<hp:lineSegArray>` (and its children) from paragraphs you actually modified. Do NOT remove them from unmodified paragraphs.
- **Korean text preservation**: Do not alter any Korean label text or the static note line.
- **ZIP recreation**: Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression. Preserve the directory structure exactly.

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