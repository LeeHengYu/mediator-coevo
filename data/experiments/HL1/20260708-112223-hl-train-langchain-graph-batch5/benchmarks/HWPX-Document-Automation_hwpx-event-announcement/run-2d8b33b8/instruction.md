# Task Instruction

Complete the following task step by step:

## Goal
Prepare the event announcement document `event_announcement_template.hwpx` using the values in `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Explore the workspace
- List files in the current directory and any subdirectories to find `event_announcement_template.hwpx` and `event_data.json`.
- Read `event_data.json` to understand all the key-value pairs available for substitution.

### 2. Inspect the HWPX template
- HWPX files are ZIP archives. Unzip `event_announcement_template.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`).
- List all files in the extracted archive to understand its structure.
- Examine all XML files (especially files like `Contents/section0.xml`, `Contents/section1.xml`, etc.) to find `{{...}}` placeholders.
- Also check other XML files (e.g., header, content.hpf, etc.) for any placeholders.
- Note: Placeholders like `{{event_name}}` may be **split across multiple `<hp:t>` nodes** within a single `<hp:p>` element. For example, `{{event` might be in one `<hp:t>` and `_name}}` in another. You MUST handle this.

### 3. Write a Python script to perform the replacements
Create a Python script that:

a) **Reads `event_data.json`** to get all key-value pairs.

b) **For each XML file in the extracted HWPX** that contains placeholder text:
   - Parse each `<hp:p>` paragraph element.
   - For each paragraph, concatenate all `<hp:t>` text nodes to form the full paragraph text.
   - Check if the concatenated text contains any `{{...}}` patterns.
   - If it does, perform all replacements using the JSON data.
   - Then, place the fully replaced text into the **first `<hp:t>` node** of that paragraph, and **clear or remove the remaining `<hp:t>` nodes** (set their text to empty string, or remove the extra `<hp:run>` elements while keeping the first one with its formatting).
   - **Important**: After modifying a paragraph's text, remove any `<hp:lineSegArray>` element (case-insensitive check: also `<hp:linesegarray>`, `<hp:lineSegArray>`) from that `<hp:p>` element. This clears stale layout cache and prevents overlapping characters when the document is opened.

c) **Write the modified XML back** to the extracted directory, preserving XML declarations and encoding.

d) **Repack the HWPX** as a ZIP file:
   - If a `mimetype` file exists, add it first as the first entry with `ZIP_STORED` (no compression).
   - Add all other files with `ZIP_DEFLATED` compression.
   - Save to `/root/event_announcement_ready.hwpx`.

### 4. Run the script
Execute the Python script.

### 5. Validate the output
- Unzip `/root/event_announcement_ready.hwpx` to a temporary validation directory.
- Grep all XML files for `{{` to confirm **no placeholders remain**.
- Grep for a few expected values from `event_data.json` to confirm they appear in the XML.
- Check that Korean labels and static note lines are still present (grep for a few Korean strings from the original).
- Verify the file is a valid ZIP by listing its contents.

### 6. Fix any issues
- If any `{{...}}` placeholders remain, investigate why (likely split across nodes that weren't properly merged) and fix the script.
- Re-run and re-validate until clean.

## Critical Technical Notes
- **Fragmented placeholders**: The most common failure mode is placeholders split across multiple `<hp:t>` XML nodes. You MUST merge all text within a paragraph before doing replacements.
- **Layout cache removal**: Any `<hp:p>` element whose text content is modified MUST have its `<hp:lineSegArray>` child element removed (look for the element with local name `lineSegArray` regardless of namespace prefix).
- **ZIP packaging**: The output must be a valid ZIP. Use Python's `zipfile` module. If `mimetype` exists, store it uncompressed as the first entry.
- **Preserve formatting**: Keep the first `<hp:run>` element's character properties (`<hp:charPr>`) intact so formatting is preserved. Only modify text content.
- **Do not change Korean labels or static note lines** — only replace `{{placeholder}}` patterns with their corresponding JSON values.

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