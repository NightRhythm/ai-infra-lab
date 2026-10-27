# Mini-SGLang KV-cache 学习记录

## 整体结构
- MHAKVCache 保存真正的 K/V 张量，形状为 [K/V, layer, page, page内token, KV head, head_dim]。
- page_table[请求行, token位置] 记录该请求每个 token 对应的 KV 物理位置。它是 Engine 持有的常驻表，每个 batch 更新和使用其中相关表项。
- CacheManager 负责分配、回收 page，并协调前缀缓存。
- RadixPrefixCache 保存“token 前缀 → KV 物理位置”的索引，本身不存 K/V 张量。
- free_slots 存的是空闲 page 的起始 token 位置。比如 page_size=4 时，8 表示第 2 页起点，对应 token 位置 8–11。

## 请求与计算流程
- 新请求先查找可复用前缀。命中长度是 cached_len；本轮新算的 token 数是 extend_len = device_len - cached_len。
- Scheduler 分配新 page，更新请求的 page table，并设置 batch.out_loc。
- Attention backend 用 out_loc 把当前层新算出的 K、V 写进 KV 池；历史 KV 则按映射读取。
- Prompt 阶段通常称为 Prefill。采样出的第一个输出 token 会在后续 Decode forward 中作为输入，计算它自己的 KV。
- 项目中的 uid 是单次推理请求 ID，不是聊天会话 ID。后续请求若带上相同历史前缀，可以通过 Radix cache 复用 KV。

## Radix cache 与回收
- 插入和匹配都按 page_size 对齐。匹配时，match_req() 最多检查输入的前 input_len - 1 个 token，之后有效匹配长度还会向下对齐到 page 边界。
- Radix tree 把相同前缀合并成共享节点，不同后缀从分叉处分开。
- Handle 的引用计数保护正在使用的缓存节点。共享节点要等所有使用者都 unlock、计数回到 0，才可能淘汰。
- 淘汰优先处理未锁定的叶子节点，避免破坏仍被分支共享的前缀。
- 请求结束后，table slot（请求行编号）归还给 TableManager；KV page 则由 CacheManager 单独处理。完整缓存前缀可保留，未缓存尾部可归还。
- page 只有在没有活跃请求或缓存索引持有时，才能放回 free_slots。项目不会靠“失效标记”自动避免物理位置冲突。