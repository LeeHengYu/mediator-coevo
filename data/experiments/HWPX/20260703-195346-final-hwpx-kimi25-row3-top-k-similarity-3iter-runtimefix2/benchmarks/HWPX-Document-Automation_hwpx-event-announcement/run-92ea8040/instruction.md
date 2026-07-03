# Task Instruction

Complete the following task to prepare an event announcement HWPX document.

## Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, and save the result to `/root/event_announcement_ready.hwpx`.

## Steps

1. **Inspect the workspace**: List files in the current directory and locate `event_announcement_template.hwpx` and `event_data.json`. Read `event_data.json` to understand all key-value pairs.

2. **Understand the HWPX structure**: A `.hwpx` file is a ZIP archive containing XML files (typically under `Contents/`). Unzip the template to a temporary directory (e.g., `/tmp/hwpx_work/`) and list all files. Identify which XML files contain `{{` placeholder text by grepping recursively.

3. **Write and run a Python script** that does the following:
   - Reads `event_data.json` to get the replacement mapping.
   - Unzips `event_announcement_template.hwpx` to a temp directory.
   - For each XML file in the archive, reads the raw XML text.
   - **Handles tag-split placeholders**: Placeholders like `{{event_name}}` may be split across multiple XML inline tags (e.g., `<hp:t>{{event</hp:t><hp:t>_name}}</hp:t>`). To handle this:
     a. First, collapse/merge adjacent `<hp:t>` runs that together form a placeholder. A robust approach: concatenate all text content, perform replacements on the concatenated text, then redistribute. Alternatively, use regex to find `{{` and `}}` across tag boundaries and merge the relevant text nodes.
     b. A simpler proven approach: strip all XML tags to get raw text, check if placeholders exist, then in the actual XML, use a regex that matches `{{` ... `}}` spanning across XML tags (e.g., `re.sub(r'\{\{[^}]*\}\}', ...)` won't work across tags). Instead, for each XML file:
        - Remove all text between `>` and `<` markers to find tag structure, OR
        - Use a two-pass approach: (1) merge split `<hp:t>` elements that together contain `{{...}}` patterns, (2) then do simple text replacement.
     c. **Recommended robust method**: Read the XML as a string. Use regex to find and merge consecutive `<hp:t>...</hp:t>` elements where the combined text contains a `{{...}}` pattern. Specifically:
        - Find all sequences of `<hp:t>text1</hp:t></hp:run><hp:run ...><hp:t>text2</hp:t>` (or just `</hp:t><hp:t>` within the same run) where text1+text2 contains part of a `{{...}}` placeholder.
        - A practical approach: temporarily strip all XML tags between `{{` and `}}` to collapse the placeholder, then replace.
        - Pattern: `re.sub(r'(\{\{)([^}]*?)(\}\})', lambda m: ..., xml_with_tags_stripped_inside_placeholders)`
        - Even simpler: replace the XML content by first removing any XML tags that appear between `{{` and `}}`. Use: `re.sub(r'\{\{(?:(?!\}\}).)*\}\}', ...)` but this won't match across tags. So first do: `content = re.sub(r'(\{\{)(.*?)(\}\})', lambda m: m.group(0), content)` — this won't help either.
        - **Best proven approach from prior success**: Strip interior tags. Do `content = re.sub(r'(<hp:t>)(.*?)(</hp:t>)', ...)` is complex. Instead:
          1. Use `re.sub(r'</hp:t>(.*?)<hp:t>', lambda m: m.group(1) if not re.search(r'<', m.group(1).replace('</hp:run>','').replace('<hp:run','')) else m.group(0), content)` — too fragile.
          2. **Simplest robust method**: Read XML as string. Remove all tags *inside* placeholder boundaries. Do this iteratively: `while re.search(r'\{\{[^}]*<[^>]+>[^}]*\}\}', content): content = re.sub(r'(\{\{[^}]*)<[^>]+>([^}]*\}\})', r'\1\2', content)`. This strips any XML tag found between `{{` and `}}`.
   - After merging, perform straightforward string replacement for each `{{key}}` → value from JSON.
   - **Remove layout cache elements**: For any XML file that was modified, remove all `<hp:lineSegArray>...</hp:lineSegArray>` elements (use `re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', content, flags=re.DOTALL)`). This prevents stale layout from causing overlapping characters.
   - Write modified XML files back.
   - Re-zip everything into `/root/event_announcement_ready.hwpx`, preserving the original directory structure and using ZIP_DEFLATED compression.

4. **Verify the output**:
   - Unzip `/root/event_announcement_ready.hwpx` to another temp directory.
   - Grep recursively for `{{` — there must be zero matches.
   - Grep for a few expected values from the JSON to confirm they appear.
   - Grep for `<hp:lineSegArray>` in modified files — should be absent.
   - Confirm the file is a valid ZIP.

5. **Run the test suite** if a test file exists (e.g., `pytest test_output.py -v` or similar) to confirm the verifier passes.

## Key Constraints
- All Korean labels and static note lines must remain unchanged.
- No `{{...}}` placeholders may remain in any file within the HWPX package.
- The output must be a valid `.hwpx` (ZIP) package.
- Remove `<hp:lineSegArray>` from any paragraph whose text was modified.
- Do NOT remove or weaken any test files or verifier scripts.

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