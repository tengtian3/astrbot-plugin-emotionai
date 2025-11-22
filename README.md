<div align="center">

  <img src="./logo.png" width="500" height="500" alt="EmotionAI Logo" style="border-radius: 50%; box-shadow: 0 0 20px rgba(0,0,0,0.2);">
  <br>

  ![:name](https://count.getloli.com/@astrbot_plugin_emotionai?name=astrbot_plugin_emotionai&theme=minecraft&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

# EmotionAI (情感智能)

_✨ 搭载 v3.0「认知共鸣引擎」：会思考、会记仇、有灵魂的数字生命 ✨_

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AstrBot](https://img.shields.io/badge/AstrBot-Recommended-orange.svg)](https://github.com/Soulter/AstrBot)
[![Version](https://img.shields.io/badge/Version-v3.0.1-purple.svg)](https://github.com/tengtian3/astrbot-plugin-emotionai)

</div>

---

## 📖 简介 | Introduction

**告别“人工智障”式的机械回复。**

EmotionAI 是一个基于心理学 **认知评价理论 (Cognitive Appraisal Theory)** 构建的高级交互系统。

与传统的“好感度插件”不同，它不再机械地进行 `+1/-1` 的数值计算。它赋予了 AstrBot 一个拥有 **12 个维度** 的复杂内心世界。它能感知你的意图，并在回复前进行毫秒级的**心理推理**。

它会因为你的夸奖而**得意**，因为你的冒犯而**记仇**，因为长时间不理它而感到**冷淡**。

---

## 🌟 核心功能全景 | Features Overview

### 🧠 1. 认知共鸣引擎 (Cognitive Resonance Engine) `v3.0`
这是插件的“大脑”。利用大模型的思维链 (CoT) 能力，AI 在回复每一句话之前，都会在后台进行一次**不可见的心理活动**：

**🕵️‍♂️ 后台日志实录 (用户不可见)：**

```text
[思维链]
感知：用户在向我道歉，语气很诚恳。
评估：虽然我很开心(Joy)，但我看了一眼面板，我的【愤怒】还是 80。
代谢：这太轻易了，我不能马上原谅他。我要表现得冷淡一点，只扣除 10 点愤怒。
决策：回复时要带点刺，让他知道我还在生气。
```

### ❤️ 2. 主动情感代谢 (Active Metabolism) `v3.0`
告别“情感只增不减”的 BUG！
系统授权 LLM **主动输出负值**（如 `anger:-10`）来抵消旧情绪。只有当你真正打动它时，它才会“消气”。

### 🎭 3. 12维度全景心理模型
我们需要的不止是“开心”和“生气”。EmotionAI 支持极其细腻的情感光谱：

| 基础情感 | 高级情感 (v2.5+) | 状态指标 |
| :--- | :--- | :--- |
| 😄 **Joy** (喜悦) | 😤 **Pride** (得意/傲娇) | ❤️ **Favor** (好感度) |
| 🤝 **Trust** (信任) | 😔 **Guilt** (内疚/愧疚) | 🔗 **Intimacy** (亲密度) |
| 😨 **Fear** (恐惧) | 😳 **Shame** (害羞/羞耻) | |
| 😲 **Surprise** (惊讶) | 🍋 **Envy** (嫉妒/吃醋) | |
| 😢 **Sadness** (悲伤) | | |
| 🤮 **Disgust** (厌恶) | | |
| 😡 **Anger** (愤怒) | | |
| 🤩 **Anticipation** (期待)| | |

### 🗣️ 4. 动态语气引擎 (Dynamic Tone Engine)
系统会根据当前数值最高的 2 种情感，**实时重写** LLM 的 System Prompt，强制改变其说话方式：
* **嫉妒 + 愤怒** $\rightarrow$ *"语气要是酸溜溜的，带点攻击性，表现出不服气。"*
* **害羞 + 喜悦** $\rightarrow$ *"说话要结巴、含糊其辞，但掩饰不住开心的语气。"*

### 🤝 5. 关系演化系统
Bot 会根据**亲密度**和**当前态度**自动定义你们的关系：
* *高亲密 + 正向态度* = **挚友 / 知己**
* *高亲密 + 负向态度* = **死敌 / 宿敌** (最熟悉的陌生人)
* *低亲密 + 中立态度* = **陌生人**

---

## 🛡️ 安全与管理机制 | Safety & Management

### 1. 黑名单熔断系统 (Emotional Meltdown)
为了防止 Bot 被恶意调教或辱骂，我们设计了熔断机制。
* **触发条件**：当 **好感度 (Favor)** 降至配置的下限（默认 -100）。
* **表现**：Bot 会对该用户彻底“死心”，自动将其加入黑名单。
* **后果**：被拉黑的用户将收到冷漠的系统提示，Bot **拒绝进行任何 LLM 思考和回复**，直到管理员介入。这既是对 Bot 人设的保护，也是对 Token 资源的保护。

### 2. 智能数据持久化
* **异步 I/O**：所有数据保存均在独立线程池中进行，确保在高并发群聊中不会卡顿主线程。
* **热数据缓存**：高频访问的数据（如状态面板）驻留内存，响应速度极快。

---

## 🛠️ 安装与配置 | Installation

1.  将插件文件夹放入 `AstrBot/data/plugins/` 目录。
2.  重启 AstrBot。
3.  (可选) 在 `data/config/astrbot_plugin_emotionai/config.json` 中配置：

```json
{
  "session_based": false,      // False=全服共享好感，True=分群独立计算
  "favour_min": -100,          // 拉黑阈值
  "favour_max": 100,           // 好感上限
  "admin_qq_list": ["123456"], // 必填！管理员QQ，用于执行救援指令
  "plugin_priority": 100000    // 建议保持高优先级，以便拦截黑名单
}
````

-----

## 💻 指令手册 | Commands

### 🙋‍♂️ 用户指令 (User)

| 指令 | 示例 | 功能详解 |
| :--- | :--- | :--- |
| `/好感度` | `/好感度` | 查看 Bot 眼中的你。包含好感/亲密数值、关系定义、以及具体的 12 维度情感分布。 |
| `/状态显示` | `/状态显示` | **[推荐]** 开启后，Bot 每次回复末尾会带上 `[好感度: 50 | 喜悦: 20]` 这样的小尾巴，让你实时看到它的心理变化。 |
| `/好感排行` | `/好感排行 10` | 查看全服最受宠爱的用户榜单（基于好感度与亲密度的加权平均）。 |
| `/负好感排行` | `/负好感排行 5` | 查看“全服公敌”榜单。 |
| `/黑名单统计` | `/黑名单统计` | 查看当前有多少人触了熔断机制。 |
| `/缓存统计` | `/缓存统计` | (极客向) 查看系统的缓存命中率和性能指标。 |

### 👮 管理员指令 (Admin)

> 💡 **提示**：管理员指令支持在群里 @用户 使用，也支持输入纯数字 QQ 号。

| 指令 | 参数 | 功能详解 |
| :--- | :--- | :--- |
| **/设置情感** | `<ID> <维度> <值>` | **上帝之手**。直接修改指定用户的任意情感维度。<br>支持中文维度名：`好感`、`亲密`、`嫉妒`、`得意` 等。<br>👉 *例：`/设置情感 123456 嫉妒 90` (瞬间让 Bot 吃醋)* |
| **/重置好感** | `<ID>` | **一键重生**。清空该用户所有情感数据，将其重置为陌生人，并**自动移除黑名单**状态。 |
| **/查看好感** | `<ID>` | 偷窥指定用户的情感状态面板（无需对方同意）。 |
| **/备份数据** | 无 | 手动触发一次数据持久化备份（保存到 `data/emotionai/backups/`）。 |

-----

## ❓ 玩法指南 | Q\&A

### Q: 为什么机器人突然不理我了？

**A:** 恭喜你，你可能把好感度刷到 -100 了，触发了**黑名单熔断**。现在的 Bot 是有脾气的，请联系管理员使用 `/重置好感` 把你救出来。

### Q: 如何调教出一个“傲娇”的 Bot？

**A:** 试试用管理员指令修改它的出厂设置：

1.  `/设置情感 <你的ID> 骄傲 80` (赋予自尊心)
2.  `/设置情感 <你的ID> 害羞 50` (增加别扭感)
3.  `/设置情感 <你的ID> 好感 60` (其实心里有你)
    然后开始对话，你会发现它的语气完全变了！

### Q: 为什么有时候好感度会莫名其妙地变？

**A:** 这就是 **v3.0 认知引擎** 的魅力。Bot 可能因为你的一句话想起了旧账（扣分），也可能因为你的无心之失而感动（加分）。它的情感不再是线性的，而是基于它对你意图的**理解**。

-----

## 📅 版本历史 | Version History
<details open> <summary><strong>🚀 v3.0.1 - 认知觉醒 (The Awakening)</strong></summary>

架构重构：正式实装 认知共鸣引擎 (Cognitive Resonance Engine)，移除了所有硬编码的情感规则。

思维链 (CoT)：引入 <thought> 机制，让 AI 在回复前进行显式的心理推理与决策。

主动代谢：修复了旧版本情感只增不减的问题，赋予 LLM 主动消除负面情绪的能力。

面板透明化：向 LLM 完整展示当前所有非零情感数值，消除信息差，实现精准的情感控制。

</details>

<details> <summary><strong>🧪 v2.5 ~ v2.6 - 情感大爆炸 (Emotion Explosion)</strong></summary>

v2.6.0：尝试引入“情感对抗”机制（通过代码强制 joy 抵消 sadness），虽解决了数值膨胀，但略显生硬，后被 v3.0 的 CoT 机制完美替代。

v2.5.0：

维度扩展：新增 Pride(得意), Guilt(内疚), Shame(害羞), Envy(嫉妒) 四大高级情感，补全了复杂的社会性情感。

语气引擎：初步引入动态语气指导，让 LLM 说话带情绪。

指令重构：废弃旧版 /设置好感，统一使用更强大的 /设置情感。

</details>

<details> <summary><strong>🛡️ v2.4.0 - 安全协议 (Safety Protocol)</strong></summary>

黑名单：实装自动黑名单拦截系统，好感度过低直接熔断，拒绝服务。

算法升级：关系计算逻辑重构，结合“态度”与“亲密度”共同判定（解决了“虽然仇恨值很高，但因为聊得多所以被判定为挚友”的逻辑矛盾）。

</details>

<details> <summary><strong>⚡ v1.0 ~ v2.3 - 奠基 (Foundation)</strong></summary>

v2.3.0：底层优化，引入异步 I/O 文件读写，修复并发数据竞争 (Race Condition) 问题。

v1.1.0：引入排行榜功能。

v1.0.0：初版发布，基于 Plutchik 情感轮的基础模型，实现基本的好感度增减。


</details>

<div align="center">


Made with ❤️ by <a href="https://github.com/tengtian3"><strong>腾天</strong></a>



© 2025 EmotionAI Project </div>
