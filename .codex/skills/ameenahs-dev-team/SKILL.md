---
name: ameenahs-dev-team
description: "Project-local AmeenahsDevTeam amendment for truthful standing-thread routing and the Windows split-root sandbox root cause. Use for agent coordination, worktree editing, sandbox failures, and governance in autocleaneeg-icvision."
---

# AmeenahsDevTeam - project amendment

Use the installed AmeenahsDevTeam skill for the full roster, routing, security, loop, worktree, review, and correspondence rules.

## Windows split-root root cause

Treat `windows unelevated restricted-token sandbox cannot enforce split writable root sets directly` as a platform defect at the task-launch/Windows sandbox-policy boundary.

The platform-neutral task policy can advertise multiple writable roots. The Windows restricted-token policy path must canonicalize, validate, and enforce that complete collection atomically before launching a tool. It must retain least privilege and reject unsupported policies at launch with structured diagnostics.

Do not call these root-cause fixes:

- placing a worktree beneath the repository;
- changing task or shell cwd;
- retrying, waiting, or shrinking the patch;
- showing that shell or Git writes succeed;
- escalating a single command;
- using `git apply`, a unified-diff handoff, or another integration owner.

Those are interim delivery unblocks only. They must remain labeled as such and cannot close the platform issue.

A real fix must:

1. support all declared roots or fail before tool execution;
2. normalize Windows aliases without widening permissions;
3. apply all grants atomically and deny undeclared locations;
4. preserve zero-root and single-root behavior;
5. avoid unrestricted, common-ancestor, or first-root-only fallbacks;
6. handle drive differences, UNC paths, reparse points, spaces, non-ASCII paths, cleanup, and partial failures;
7. pass Windows integration tests proving `apply_patch` and ordinary shell writes inside every declared root while outside-root writes are denied.

When the platform defect blocks delivery, allow at most one exact-root task attempt. A reviewed unified-diff handoff may then unblock delivery, with maker and applying integration owner recorded separately. Do not present that handoff as a platform repair.

