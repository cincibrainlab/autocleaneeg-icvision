# Agent Instructions

## Windows Split-Root Sandbox: Root-Cause Rule

The Windows error `windows unelevated restricted-token sandbox cannot enforce split writable root sets directly` is a task-launch/sandbox-policy defect: the task advertises multiple writable roots, but the Windows restricted-token policy compiler cannot enforce that collection.

Do not describe any of these as fixes: moving or nesting a worktree, changing `cwd`, retrying `apply_patch`, shrinking a patch, waiting, proving shell/Git access, escalating one command, or applying a diff through `git apply`. They may unblock one delivery path, but they do not repair the launcher.

The platform fix must occur at the Codex task-launch and Windows sandbox-policy boundary:

- canonicalize and deduplicate the entire writable-root collection using Windows-aware path rules;
- preserve distinct roots without widening access to a common ancestor or drive;
- validate and apply every grant atomically before starting the child process;
- keep undeclared parents, siblings, and other paths inaccessible;
- fail at launch with structured diagnostics when a policy cannot be enforced;
- never fall back to an unrestricted token or silently discard roots;
- cover single-root compatibility, multi-root enforcement, aliases, junctions/reparse points, UNC paths, cleanup, and outside-root denial in Windows CI.

A platform issue is complete only when ordinary sandboxed tools, including `apply_patch`, work across all declared roots and outside-root writes remain denied. Successful exact-root relaunch or reviewed-diff integration is an interim unblock and must not be reported as the root-cause fix.

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

