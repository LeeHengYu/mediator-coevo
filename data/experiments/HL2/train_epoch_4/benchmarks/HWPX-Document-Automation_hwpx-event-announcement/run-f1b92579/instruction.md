# Task Instruction

Complete the following task to prepare an HWPX event announcement document.

## Objective
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Step-by-step Plan

### Step 1: Inspect the input files
1. List the files in the task directory to locate `event_announcement_template.hwpx` and `event_data.json`.
2. Read `event_data.json` fully to understand all available keys and their values.
3. Since `.hwpx` is a ZIP-based package, list its contents using Python's `zipfile` module to identify all member files, especially XML files under `Contents/`.

### Step 2: Examine the XML content for placeholders
1. Read every XML file inside the HWPX archive (especially files like `Contents/section0.xml` or similar section files, plus `Contents/content.hpf` if present).
2. Search for all `{{...}}` placeholder patterns across ALL files in the archive, not just the main section XML. Print each match with its filename and surrounding context.
3. Be aware that placeholders may be **fragmented across XML tags** — e.g., `<hp:t>{{</hp:t></hp:run><hp:run ...><hp:t>event_name</hp:t></hp:run><hp:run ...><hp:t>}}</hp:t>`. You must handle this.

### Step 3: Write a Python script to perform the substitution
Create and run a Python script that:

1. Opens the template HWPX as a ZIP archive.
2. Loads the JSON data file.
3. For each file in the archive:
   a. If it's an XML file (or any text-based file), read its content as text.
   b. **Defragment placeholders**: Use a regex approach that works on the full XML text to reconstruct placeholders that may be split across multiple `<hp:t>` or `<hp:run>` elements. Specifically:
      - For each paragraph (`<hp:p>...</hp:p>`), extract the full concatenated text content.
      - If the concatenated text contains a `{{...}}` pattern, perform the replacement at the paragraph level.
      - A robust approach: within each `<hp:p>` block, use regex to find sequences of `<hp:run>` elements whose combined `<hp:t>` text forms a `{{placeholder}}` pattern, then consolidate them into a single `<hp:run>` with the replacement value in its `<hp:t>` tag.
      - Alternatively, use a simpler but effective approach: strip all XML tags from within a paragraph to get plain text, find placeholders, then do targeted string replacements on the raw XML (handling the tag-fragmented versions).
   c. Also handle the simple case where `{{placeholder}}` appears intact within a single `<hp:t>` element — do a direct string replacement.
   d. After substitution, **remove `<hp:linesegarray>` elements** (and their contents, i.e., `<hp:linesegarray>...</hp:linesegarray>`) from any paragraph (`<hp:p>`) that was modified. This is critical to prevent stale layout-cache causing overlapping characters when the document is opened.
   e. If the file is not text-based (binary), copy it unchanged.
4. Write all files into a new ZIP archive at `/root/event_announcement_ready.hwpx`, preserving the original directory structure and compression settings.

### Step 4: Validate the output
1. Open `/root/event_announcement_ready.hwpx` as a ZIP and verify it's a valid archive.
2. Read all XML files from the output and search for any remaining `{{...}}` patterns. Print results. There must be ZERO remaining placeholders.
3. Verify that Korean labels and the static note line are still present (spot-check a few known Korean strings from the original).
4. Confirm that `<hp:linesegarray>` elements have been removed from modified paragraphs.
5. Print a summary of all replacements made (placeholder → value).

## Critical Reminders
- **Placeholder fragmentation**: This is the #1 failure mode. Placeholders WILL be split across XML tags. You MUST handle this. Test by printing the concatenated text of each paragraph and checking for `{{`.
- **Layout cache cleanup**: Remove `<hp:linesegarray>...</hp:linesegarray>` from every modified `<hp:p>` element. Use regex: `re.sub(r'<hp:linesegarray>.*?</hp:linesegarray>', '', paragraph_xml, flags=re.DOTALL)`.
- **Preserve everything else**: Do not modify paragraphs that don't contain placeholders. Do not change Korean text, static notes, or any non-placeholder content.
- **Valid ZIP**: The output must be a proper ZIP file. Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression.
- **Check ALL archive members**: Placeholders might exist in files other than the main section XML. Check every text file in the archive.

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