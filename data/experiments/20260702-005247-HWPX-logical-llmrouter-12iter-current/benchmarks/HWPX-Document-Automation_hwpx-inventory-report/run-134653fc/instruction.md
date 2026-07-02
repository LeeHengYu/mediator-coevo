# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in `inventory_report_template.hwpx` with values from `inventory_data.json`, then save the result to `/root/inventory_report_ready.hwpx`.

Step-by-step:

1. **Inspect the input files.**
   - `cat inventory_data.json` to see all key-value pairs.
   - `mkdir /tmp/hwpx_work && cp inventory_report_template.hwpx /tmp/hwpx_work/template.zip && cd /tmp/hwpx_work && unzip template.zip -d template_extracted`
   - `grep -rn '{{' template_extracted/` to find every placeholder and which XML files contain them.
   - Also inspect the XML files containing placeholders to understand the structure (especially `<hp:t>` elements and `<hp:lineSegArray>` elements).

2. **Write a Python script** (`/tmp/hwpx_work/build_report.py`) that:
   a. Loads `inventory_data.json`.
   b. Extracts the template HWPX (ZIP) into a working directory.
   c. For each XML file in the extracted contents (especially under `Contents/`):
      - Parses the XML.
      - **Merges split `<hp:t>` elements**: Within each `<hp:run>` (or equivalent parent), if there are multiple consecutive `<hp:t>` elements, concatenate their text into a single `<hp:t>` and remove the extras. This is critical because HWPX editors sometimes split text across multiple `<hp:t>` tags, which can split a `{{placeholder}}` across elements.
      - After merging, performs string replacement of every `{{key}}` with the corresponding JSON value on the merged text content of `<hp:t>` elements.
      - **Strips all `<hp:lineSegArray>` elements** (layout cache) from any paragraph (`<hp:p>`) whose text was modified. This prevents overlapping characters when the document is opened.
      - Preserves all other structure, including empty paragraphs, Korean labels, and the static note line.
      - Writes the modified XML back.
   d. Re-packages the modified directory into a new ZIP file with `.hwpx` extension at `/root/inventory_report_ready.hwpx`, preserving the original ZIP structure (no extra top-level directory).

3. **Run the script**: `cd /tmp/hwpx_work && python3 build_report.py`

4. **Validate the output**:
   - `unzip -l /root/inventory_report_ready.hwpx` to confirm it's a valid ZIP.
   - `mkdir /tmp/hwpx_work/output_check && cd /tmp/hwpx_work/output_check && unzip /root/inventory_report_ready.hwpx`
   - `grep -rn '{{' /tmp/hwpx_work/output_check/` — must return NO results (no remaining placeholders).
   - `grep -rn 'lineSegArray' /tmp/hwpx_work/output_check/` on modified files — confirm layout cache was stripped from edited paragraphs.
   - Spot-check a few replaced values in the XML to confirm correctness.

**Key implementation details for the Python script:**
- Use `xml.etree.ElementTree` with proper namespace handling. Register all namespaces found in the XML before parsing to avoid namespace prefix mangling on output.
- When merging `<hp:t>` elements, handle the namespace URI for `hp` (likely `http://www.hancom.co.kr/hwpml/2011/paragraph` or similar — read it from the actual XML).
- When re-zipping, use `zipfile.ZipFile` and walk the extracted directory, writing files with paths relative to the extraction root (matching the original archive structure). Use `ZIP_DEFLATED` compression.
- Convert numeric JSON values to strings before replacement.
- Handle nested JSON values if present (check the actual JSON structure first).

**Constraints:**
- Do NOT remove or alter any Korean text labels.
- Do NOT remove empty paragraphs.
- Do NOT leave any `{{...}}` placeholders in the output.
- Do NOT bypass or weaken any verification checks.

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