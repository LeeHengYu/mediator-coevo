# Task Instruction

Complete the inventory status report by replacing placeholders in the HWPX template with data from the JSON file. Follow these steps precisely:

## Step 1: Inspect the input files
- Read `inventory_data.json` to understand the data structure and available keys.
- Examine the HWPX template `inventory_report_template.hwpx` by unzipping it and inspecting its structure, especially `Contents/section0.xml`.
- Identify all `{{...}}` placeholders in the XML content.
- Check which namespaces are used (e.g., `hp:`, `hc:`, etc.).

## Step 2: Write and run a Python script that does the following:

### 2a: Namespace Registration
- Before any XML parsing, register ALL namespace prefixes found in the XML file using `ET.register_namespace()`. This prevents namespace prefix loss (e.g., `hp:` becoming `ns0:`) during serialization.

### 2b: Unzip the template
- Extract `inventory_report_template.hwpx` to a temporary directory.

### 2c: Load JSON data
- Load `inventory_data.json` and build a flat mapping of placeholder names to values.
- If the JSON has nested structures, flatten them appropriately so that each `{{key}}` maps to a string value.

### 2d: Handle split placeholders
- Read the raw XML string of `Contents/section0.xml`.
- CRITICAL: Placeholders like `{{item_name}}` may be split across multiple `<hp:t>` tags due to HWP's internal formatting (e.g., `<hp:t>{{item</hp:t><hp:t>_name}}</hp:t>`). 
- Strategy: First, do a regex-based replacement on the raw XML string to replace all `{{...}}` patterns. Use a regex like `\{\{([^}]+)\}\}` and replace with the corresponding value from the JSON mapping.
- If the placeholders are split across tags, concatenate all text content within each paragraph's `<hp:run>` elements, perform replacement, and write back. Alternatively, use a string-level approach: remove XML tags between split placeholder parts by detecting incomplete `{{` patterns.
- RECOMMENDED ROBUST APPROACH: Parse the XML, for each `<hp:p>` paragraph element, extract all text from all `<hp:t>` child elements (in order), concatenate into a single string, perform all `{{...}}` replacements on that concatenated string, then place the result into the first `<hp:t>` element and clear the remaining `<hp:t>` elements (set their text to empty string). This handles splits reliably.

### 2e: Remove layout cache from modified paragraphs
- For every `<hp:p>` paragraph where text was modified (i.e., a placeholder was replaced), find and REMOVE any `<hp:lineSegArray>` child elements. This prevents overlapping/garbled text rendering.
- Do NOT remove `<hp:lineSegArray>` from unmodified paragraphs.

### 2f: Preserve document structure
- Do NOT remove or modify empty paragraphs (they serve as spacing).
- Do NOT modify Korean label text or the static note line.
- Only modify text that contains `{{...}}` placeholders.

### 2g: Write back the modified XML
- Serialize the modified XML tree back to `Contents/section0.xml` in the extracted directory.
- Use `xml_declaration=True, encoding='UTF-8'` for serialization.

### 2h: Re-zip into a valid .hwpx
- Create `/root/inventory_report_ready.hwpx` as a ZIP file.
- IMPORTANT: Zip from within the extracted directory root so that paths like `mimetype`, `Contents/section0.xml`, `META-INF/`, etc. are at the archive root (not nested under an extra directory).
- Use `zipfile.ZIP_DEFLATED` compression, but for the `mimetype` file specifically, use `ZIP_STORED` (no compression) if it exists.

## Step 3: Validate the output
- Unzip `/root/inventory_report_ready.hwpx` and read `Contents/section0.xml`.
- Verify that NO `{{...}}` placeholders remain anywhere in the XML text.
- Verify that Korean labels and the static note line are still present.
- Verify that empty paragraphs are preserved.
- Verify that `<hp:lineSegArray>` elements are removed from modified paragraphs.
- Print a summary of replacements made and validation results.

If any `{{...}}` placeholders remain after the first pass, investigate whether they were split across tags and apply the concatenation approach described in 2d.

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