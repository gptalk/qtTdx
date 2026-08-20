# V0.2-E Integration Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist V0.2-E integration decision into project memory (one Markdown file + MEMORY.md index update + git commit) so the routing recommendation survives across sessions.

**Architecture:** V0.2-E is a pure diagnostic decision document — zero code change. The only persistent artifact is a memory entry at `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\projection-v02-e-integration-decision.md`. The memory entry records (a) the V0.2-D + V0.2-C1 paired diagnosis, (b) the recommended routing (market-driver as new main line), (c) trigger conditions for reversing the routing, (d) follow-up spec hooks. MEMORY.md gets a one-line pointer.

**Tech Stack:** Markdown (memory file format), git (commit).

## Global Constraints

- **No code change.** V0.2-E scope is decision document + memory persistence only. The driver switch itself is deferred to V0.2-F (separate spec).
- **Memory file format** (per `C:\Users\yellow\CLAUDE.md` and existing entries in `memory/`): frontmatter with `name`, `description`, `metadata` (type=project, originSessionId, modified ISO 8601); body with `## Why:` / `## How to apply:` lines where relevant; link liberally to other memories with `[[name]]`.
- **MEMORY.md format** (existing): one-line bullet `- [Title](file.md) — hook`; no frontmatter; one line per memory.
- **Commit message style** (matching `docs/superpowers/specs/2026-08-20-dynamics-c1-market-driver-swap.md` `d84856c` pattern): `docs(projection): V0.2-E integration decision memory — market-driver recommended (zero code)`.
- **Windows GBK safety**: write memory file with `encoding='utf-8'`; commit message must be ASCII (memory entry body may contain Chinese per existing entries).
- **Do not modify** `docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md` — it's the source of truth. Do not modify `ablation_fit.py` / `prediction_ode.py` / `dynamics_*.py` / `gp_factor_mining/*` / `_solve_ols` — V0.2-E has no code surface.

---

### Task 1: Write V0.2-E memory entry + update MEMORY.md index + commit

**Files:**
- Create: `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\projection-v02-e-integration-decision.md`
- Modify: `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\MEMORY.md` (append one bullet after `projection-v02-c1-market-driver-swap` line)

**Interfaces:**
- Consumes: V0.2-E spec content (already committed at commit `0c5d0e7`); V0.2-D memory `[projection-v02-d-oos-reversal-decomposition](projection-v02-d-oos-reversal-decomposition.md)`; V0.2-C1 memory `[projection-v02-c1-market-driver-swap](projection-v02-c1-market-driver-swap.md)`.
- Produces: a new memory entry future sessions can recall to recover the integration decision and its trigger conditions.

- [ ] **Step 1: Write the memory file**

Write `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\projection-v02-e-integration-decision.md` with the following content (verbatim, ASCII + UTF-8 Chinese where appropriate):

```markdown
---
name: projection-v02-e-integration-decision
description: "V0.2-E Integration Decision — recommend market-driver as new main line (H1b 强, tail 缩 65%); 零代码, 4 个反转触发条件"
metadata:
  node_type: memory
  type: project
  originSessionId: 70589dd6-ae16-4fbe-81ec-d694989743ec
  modified: 2026-08-20T08:30:00.000Z
---

# V0.2-E — Integration Decision

**Date**: 2026-08-20
**Spec**: [docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md](docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md)
**Status**: ✅ APPROVED, 零代码, 仅本 memory 持久化决策

## 决策一句话

**Switch main-line driver from per-stock 申万二级 industry index → per-exchange market index (SH→000001.SH, SZ→399001.SZ).** 理由: industry-driver 系统性放大 q_drift tail (10.27% → 3.61%, -65%, H1b 强), 但不产生 alpha (mean IC -0.48 → -0.52); 同时 ic_real std 集中 46% (0.17 → 0.09), 减少 cross-stock dispersion; sign_flip 仅 2.7%, 不是 trade-off。

## 路由摘要

| Scenario (spec §5) | 条件 | C1 数据 | 评估 |
|---|---|---|---|
| A. Market > Industry | `P_C1(|q_drift|>0.3) < 5%` ✓ (3.6%); `sign_flipped == False` for >60% ✓ (97.3%); `delta_oos_ic > +0.05` for >60% ✗ (62% borderline) | partial match |
| B. Industry > Market | `P_C1(|q_drift|>0.3) > 15%` | ✗ (3.6%) | RULED OUT |
| C. Both bad | both IC < 0 AND P(>0.3) > 5% | both IC < 0 ✓; C1 P=3.6% (< 5%) | not C |
| D. Both good | both IC > 0.3 | ✗ (both ≈ -0.5) | not D |

→ 最接近 **Scenario A**, 但 ic delta 均值略负。

## 4 个反转触发条件 (要推翻 market-driver 推荐)

1. **V0.2-D.2** 跨股分析显示 `q_drift ∝ industry β 估计残差` 强正相关 — 这 **加强** A 推荐, 不推翻
3. **V0.2-C.2** two-tier 实验显示 industry 联合 market 加入增量信号 — 路由改为 "market + industry"
4. **V0.2-B** shrinkage 在 industry-driver 上把 D1 tail 压到 market 水平以下 — 改为 "industry + shrinkage"
5. **Rolling OOS** (未来 v6+ dynamics) 显示 market-driver tail 重新膨胀 — 撤回推荐

## 5 个风险

| Risk | Mitigation |
|---|---|
| 1. 行业信号丢失 (隐藏 alpha) | V0.2-C.2 two-tier 验证 |
| 2. Market-driver 过度平滑 | 抽查 10-20 高 IC_improved 票的 (β, q, d) 是否真的独特 |
| 3. OOS mean IC 负 (-0.5) — driver 都不是 alpha source | V0.3 re-spec (β·d² / sign-aware / 替换 q 等) |
| 4. OOS 单 70/30 切分, 没验 rolling stability | future rolling OOS evaluation |

## 4 个 follow-up spec hooks

- **V0.2-F** (如批准): driver-default migration (改 `projection_batch.py` 默认 `--index` 为 per-exchange market)
- **V0.2-C.2**: 两层 driver (market + industry 同时入), 检验 industry 增量
- **V0.2-D.2**: 跨股 `corr_s(q_s, E_β,s)`, 验 H1b 机制
- **V0.2-B** (优先级降): 现在 C1 tail 缩 65%, 收缩 less urgent; 仅在保 industry-driver 时仍必要

## 不在范围 (per spec §8)

- Driver switch 实施 (V0.2-F)
- V0.2-C.2 / V0.2-D.2 (独立 spec)
- 修改 `ablation_fit.py` / `_solve_ols` / `prediction_ode.py` / `dynamics_*.py` / `gp_factor_mining/*` (禁止)

## 链接

- V0.2-D (H1/H2/H3 baseline, industry-driver): [projection-v02-d-oos-reversal-decomposition](projection-v02-d-oos-reversal-decomposition.md)
- V0.2-C1 (paired comparison, market-driver swap): [projection-v02-c1-market-driver-swap](projection-v02-c1-market-driver-swap.md)
- V0.1 (Model 2 优先 + β-drift 无效): [projection-v01-specification-correction-ablation](projection-v01-specification-correction-ablation.md)
- V0 审计 (OLS 不 sick, R² 低是 spec 错): [parameter-fit-v0-identifiability-audit](parameter-fit-v0-identifiability-audit.md)
```

- [ ] **Step 2: Update MEMORY.md index**

Append one bullet line after the existing `projection-v02-c1-market-driver-swap` line in `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\MEMORY.md`:

```
- [projection-v02-e-integration-decision](projection-v02-e-integration-decision.md) — V0.2-E 集成决策: 推荐 market-driver 作 new main line (H1b 强, D1 tail 10.3%→3.6%, -65%); 零代码; 4 个反转触发条件 + 5 风险 + 4 follow-up
```

- [ ] **Step 3: Verify by reading back**

Read both files and confirm:
- Memory entry has frontmatter with `name: projection-v02-e-integration-decision`
- Memory entry body has the 5 sections (决策一句话 / 路由摘要 / 4 个反转触发条件 / 5 个风险 / 4 个 follow-up spec hooks / 不在范围 / 链接)
- MEMORY.md has the new bullet line
- No existing lines in MEMORY.md were removed or reordered

- [ ] **Step 4: Stage the changes**

```bash
cd "c:/Users/yellow/mcp/qtTdx" && git status --short
```

Expected output:
```
 M ../.claude/projects/c--Users-yellow-mcp-qtTdx/memory/MEMORY.md
 ?? ../.claude/projects/c--Users-yellow-mcp-qtTdx/memory/projection-v02-e-integration-decision.md
```

(The files live outside the repo root, but git will pick them up via `git add` with absolute paths. If git refuses, fall back to manually copying the file content into `docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md` `附录 A: Memory Snapshot` and committing only that file.)

- [ ] **Step 5: Commit**

```bash
cd "c:/Users/yellow/mcp/qtTdx" && git add C:/Users/yellow/.claude/projects/c--Users-yellow-mcp-qtTdx/memory/MEMORY.md C:/Users/yellow/.claude/projects/c--Users-yellow-mcp-qtTdx/memory/projection-v02-e-integration-decision.md && git commit -m "docs(memory): V0.2-E integration decision — recommend market-driver as new main line"
```

If git add fails on absolute paths outside the repo, fallback commit:

```bash
cd "c:/Users/yellow/mcp/qtTdx" && git add docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md && git commit -m "docs(memory): V0.2-E integration decision — recommend market-driver as new main line"
```

(In the fallback, the decision is still preserved because it lives in the spec itself; the memory file just speeds up recall.)

- [ ] **Step 6: Push to origin/main**

```bash
cd "c:/Users/yellow/mcp/qtTdx" && git push origin main
```

Expected: 1 commit pushed.

- [ ] **Step 7: Confirm completion**

Run:

```bash
cd "c:/Users/yellow/mcp/qtTdx" && git log --oneline -1
```

Expected: latest commit message matches Step 5. V0.2-E is now closed. The user can decide whether to invoke V0.2-F next.

---

## Self-Review

1. **Spec coverage**: §7 (Deliverables) requires (1) this document (already exists at `docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md`, committed at `0c5d0e7`) + (2) memory entry. Task 1 produces (2). ✓
2. **Placeholder scan**: no TBD / TODO. All sections have concrete content. ✓
3. **Type consistency**: memory file format matches existing entries (frontmatter + sections). MEMORY.md format matches existing entries (one bullet per memory, one line). ✓
4. **File paths**: absolute paths to memory files, relative path to spec. ✓
5. **Commit message**: ASCII, follows `docs(memory):` prefix pattern. ✓

## Execution Handoff

"Plan complete and saved to `docs/superpowers/plans/2026-08-20-dynamics-e-integration-decision.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent for the single task, review after, fast iteration
2. **Inline Execution** - Execute the task in this session directly

Which approach?"

---

*This plan is intentionally minimal — V0.2-E is a pure diagnostic document with zero code surface. The single task here is the memory entry + index update + commit. The next-spec hooks (V0.2-F, V0.2-C.2, V0.2-D.2, V0.2-B) live in their own future specs.*