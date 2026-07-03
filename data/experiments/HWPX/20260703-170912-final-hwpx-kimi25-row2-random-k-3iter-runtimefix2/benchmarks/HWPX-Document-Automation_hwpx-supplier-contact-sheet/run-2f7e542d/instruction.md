# Task Instruction

Complete the following task to fill in a HWPX supplier contact sheet template.

## Goal
Replace all `{{...}}` placeholders in `supplier_contact_template.hwpx` with values from `supplier_contact.json`, then save the result to `/root/supplier_contact_ready.hwpx`.

## Step-by-step Plan

### Step 1: Inspect the workspace
- List files in the current directory and locate `supplier_contact_template.hwpx` and `supplier_contact.json`.
- Read `supplier_contact.json` fully to understand all available key-value pairs.

### Step 2: Understand the HWPX structure
- HWPX files are ZIP archives. Unzip `supplier_contact_template.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`).
- List all files in the extracted archive to understand the package structure.
- Identify which XML files contain document content. Typically the main content is in `Contents/section0.xml` or similar paths. Also check for files like `content.hpf`, `header.xml`, `Preview/`, etc.

### Step 3: Find all placeholders
- Search all extracted files (especially XML files) for `{{` to find every placeholder.
- Note: Placeholders like `{{company_name}}` might be split across multiple XML text runs within a single paragraph. For example, `{{` might be in one `<hp:t>` element and `company_name}}` in another.
- Print the content of each XML file that contains `{{` so you can see the full context.

### Step 4: Write a Python script to perform the replacement
Write a Python script that:

1. Extracts the HWPX ZIP to a temp directory.
2. Loads the JSON data.
3. For each XML file in the extracted archive:
   a. Reads the raw XML text.
   b. Checks if it contains `{{`.
   c. If yes, processes it carefully:
      - **Critical**: Placeholders may be split across multiple XML elements. To handle this robustly:
        - Parse the XML.
        - For each paragraph element (typically `<hp:p>` or similar), concatenate ALL text content from all child text runs to reconstruct the full text.
        - Find `{{...}}` patterns in the concatenated text.
        - Map each placeholder back to the XML text nodes and replace accordingly. One approach: collect all text nodes in a paragraph into a list, join them, do regex replacement on the joined string, then redistribute the replaced text back into the first text node and clear the rest. Or alternatively, work at the raw string level if placeholders are not split.
        - First check if simple string replacement on the raw XML works (i.e., placeholders appear intact within single elements). If `{{` and `}}` always appear in the same element, simple replacement suffices.
      - Replace each `{{key}}` with the corresponding value from the JSON.
   d. **Remove layout cache elements** from any paragraph that was modified. Layout cache elements are typically `<hp:linesegarray>` or `<hp:lineSegArray>` elements (or similar names like `<lineseg>`, `<hp:LineSeg>`). These contain cached glyph positioning that becomes stale after text changes. For every `<hp:p>` paragraph where text was changed, find and remove all `<hp:linesegarray>` / `<hp:lineSegArray>` / `<linesegarray>` child elements (check the actual element names in the XML namespace).
   e. Writes the modified XML back.
4. Re-packages everything into a new ZIP file at `/root/supplier_contact_ready.hwpx`, preserving the original directory structure and using ZIP_DEFLATED compression. Make sure to preserve the exact same file paths within the ZIP.

### Step 5: Validate the output
- Unzip `/root/supplier_contact_ready.hwpx` to a second temp directory.
- Search ALL files for `{{` — there must be ZERO remaining placeholders.
- Verify that Korean text labels are still present (search for a few Korean characters that appeared in the original).
- Verify the static note line is unchanged.
- Verify the file is a valid ZIP.
- Print a summary of all replacements made.

## Important Notes
- Do NOT skip any placeholder. Every `{{...}}` must be replaced.
- Do NOT alter Korean labels or the static note line.
- Handle the case where placeholder text spans multiple XML text nodes within a paragraph.
- Always remove layout cache elements (`linesegarray`, `lineSegArray`, `LineSeg`, or whatever the actual tag name is) from modified paragraphs. Inspect the actual XML to determine the correct element names before writing removal code.
- The output must be a valid ZIP file with `.hwpx` extension.
- If you encounter any issues with split placeholders, print detailed debug info showing the paragraph structure before attempting the fix.

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