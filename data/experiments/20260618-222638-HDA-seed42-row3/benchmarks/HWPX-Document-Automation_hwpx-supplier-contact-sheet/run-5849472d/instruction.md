# Task Instruction

Complete the following task to update a HWPX supplier contact sheet with values from a JSON file.

## Goal
Replace all `{{...}}` placeholders in `supplier_contact_template.hwpx` with values from `supplier_contact.json`, then save the result to `/root/supplier_contact_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read `supplier_contact.json` to understand the keys and values available.
- The HWPX file is a ZIP archive. Unzip `supplier_contact_template.hwpx` to a temporary directory and list all files inside.
- Identify which XML files contain `{{` placeholders. The most likely candidate is `Contents/section0.xml`, but check all XML files.
- Print the full content of every XML file that contains `{{` so you can see the exact placeholder names and surrounding XML structure.

### 2. Build a mapping from placeholders to JSON values
- Parse `supplier_contact.json`.
- For each `{{key}}` placeholder found in the XML, confirm there is a matching key in the JSON. The placeholder name inside `{{...}}` should correspond to a JSON key (possibly with dots or nested paths—inspect carefully).
- If the JSON has nested structure, flatten it appropriately to match the placeholder names.

### 3. Write a Python script to perform the substitution
Write and run a Python script that:

a. Extracts the HWPX ZIP to a temp directory.

b. For each XML file in the archive that contains `{{`:
   - Parse it with `lxml.etree` (use the appropriate namespace handling).
   - Walk all text-bearing elements (e.g., `<hp:t>` tags or any element with `.text` or `.tail` containing `{{`).
   - Replace every `{{key}}` occurrence with the corresponding JSON value.
   - **Important**: Be aware that a single placeholder may be split across multiple XML elements (e.g., `<hp:t>{{</hp:t><hp:t>name</hp:t><hp:t>}}</hp:t>`). If you detect this, you need to merge the text, perform the substitution, and place the result in the first element while clearing the others. Check for this by looking at the raw XML.
   - After modifying any `<hp:p>` paragraph element's descendant text, remove all `<hp:lineSegArray>` child elements (and any similar layout-cache elements like `<hp:lineseg>`) from that `<hp:p>` to prevent stale layout caches from causing rendering issues.

c. Serialize the modified XML back to the file, preserving the XML declaration and encoding.

d. Repackage everything into a new ZIP file at `/root/supplier_contact_ready.hwpx`:
   - If a `mimetype` file exists in the archive root, write it first with `ZIP_STORED` (no compression).
   - Write all other files with `ZIP_DEFLATED`.
   - Preserve the original directory structure.

### 4. Validate the output
- Unzip `/root/supplier_contact_ready.hwpx` to a new temp directory.
- Read all XML files and confirm:
  - **No `{{` or `}}` patterns remain anywhere** in any XML file.
  - The Korean field labels that were already in the document are still present.
  - The static note line is unchanged.
  - All JSON values appear in the XML content.
  - No `<hp:lineSegArray>` elements exist in any paragraph whose text was modified.
- Print a summary of checks passed/failed.

### 5. Important details
- Keep all Korean text labels intact—only replace the `{{...}}` placeholder portions.
- If a placeholder appears inside a larger string (e.g., `담당자: {{contact_name}}`), replace only the `{{contact_name}}` part, keeping `담당자: ` intact.
- Values from JSON should be inserted as-is (strings as strings, numbers converted to strings).
- Do NOT remove or modify any XML elements beyond the text substitution and layout-cache cleanup.
- The final file must be a valid ZIP that can be opened as an HWPX document.

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