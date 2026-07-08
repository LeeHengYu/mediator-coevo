# Task Instruction

Prepare the event announcement document by replacing all `{{...}}` placeholders with values from `event_data.json` and saving the result to `/root/event_announcement_ready.hwpx`.

## Step-by-step

### 1. Inspect the input files
- Read `/root/event_data.json` and note every key-value pair.
- Unzip `/root/event_announcement_template.hwpx` to a temporary directory (e.g., `/tmp/hwpx_work/`).
- List the extracted contents to understand the package structure.
- Read `Contents/section0.xml` (the main content XML) carefully. Look for:
  - All `{{...}}` placeholders — note their exact text and which `<hp:t>` elements they appear in.
  - Whether any placeholder is split across multiple `<hp:t>` runs (e.g., `{{` in one run and `}}` in another). This is a common HWPX issue.
  - Korean labels and static note lines that must be preserved.
  - `<hp:linesegarray>` or `<hp:lineSegArray>` elements (layout cache).

### 2. Write and run a Python script to perform the replacement

Use Python with `xml.etree.ElementTree` or `lxml` for XML manipulation. The script must:

#### 2a. Handle split placeholders
Before doing replacements, merge adjacent `<hp:t>` text runs within each paragraph (`<hp:p>`) into a single conceptual string. Specifically:
- For each `<hp:p>`, collect all `<hp:t>` elements in document order.
- Concatenate their `.text` values into one string.
- Check if this concatenated string contains any `{{...}}` pattern.
- If a placeholder spans multiple `<hp:t>` elements, consolidate: put the full merged text into the first `<hp:t>` element and set the remaining `<hp:t>` elements' text to empty string `""`.
- Then perform the `{{key}}` → value substitution on the merged text.

#### 2b. Replace placeholders
- For each `{{key}}` found in the XML text, replace it with the corresponding value from `event_data.json`.
- Use exact key matching (the key inside `{{...}}` must match a JSON key).
- Ensure all values are inserted as strings.

#### 2c. Remove layout cache elements
- After all text modifications, remove ALL `<hp:linesegarray>` elements (case-insensitive tag matching, or match with namespace) from any `<hp:p>` paragraph whose text was modified.
- To be safe, remove `<hp:linesegarray>` from ALL paragraphs in the document. This prevents stale layout cache from causing overlapping characters.

#### 2d. Preserve everything else
- Do NOT modify Korean labels, static note lines, or any other content.
- Preserve XML namespaces, attributes, and structure.

### 3. Repackage the HWPX file
- Write the modified `section0.xml` back to the extracted directory.
- Also check if there are other section XML files (section1.xml, etc.) and process them the same way if they contain placeholders.
- Re-zip the entire directory structure into `/root/event_announcement_ready.hwpx` using Python's `zipfile` module.
- Use `ZIP_DEFLATED` compression.
- Ensure the zip structure matches the original (same directory layout, no extra root folder).

### 4. Validate the output
- Unzip `/root/event_announcement_ready.hwpx` to a separate temp directory.
- Read the section XML(s) and verify:
  - No `{{` or `}}` patterns remain anywhere in any text content.
  - All expected values from `event_data.json` appear in the XML.
  - No `<hp:linesegarray>` elements remain in modified paragraphs.
  - Korean labels and static notes are intact.
- Print a summary of checks.

### Important notes from prior experience
- **Split placeholders are the #1 failure mode.** Always merge `<hp:t>` runs before matching placeholders.
- **Always remove `<hp:linesegarray>`** after modifying text — if these remain, HWP viewers display old cached text.
- **Namespace handling:** HWPX XML uses namespaces like `http://www.hancom.co.kr/hwpml/2011/paragraph`. Register namespaces before parsing to avoid `ns0:` prefix pollution in output. Use `ET.register_namespace()` for all namespaces found in the file.
- **Do not use simple regex on raw XML text** for placeholder replacement — use proper XML parsing to avoid breaking tags.

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