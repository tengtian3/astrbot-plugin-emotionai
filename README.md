<div align="center">

  <img src="./logo.png" width="500" height="500" alt="EmotionAI Logo">
  <br>

  ![:name](https://count.getloli.com/@astrbot_plugin_emotionai?name=astrbot_plugin_emotionai&theme=minecraft&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

# EmotionAI (情感智能)

_✨ 为 AstrBot 注入灵魂：12维度情感、动态语气引擎、沉浸式交互体验 ✨_

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AstrBot](https://img.shields.io/badge/AstrBot-Recommended-orange.svg)](https://github.com/Soulter/AstrBot)
[![Version](https://img.shields.io/badge/Version-v2.5.1-purple.svg)](https://github.com/tengtian3/astrbot-plugin-emotionai)

</div>

---

## 📖 简介 | Introduction

**你是否厌倦了 AI 永远千篇一律、毫无波澜的回复？**

**EmotionAI** 不仅仅是一个简单的“好感度计算器”。它是一个完整的**心理状态模拟引擎**。它让你的 AstrBot 拥有：
* **喜怒哀乐**：会开心、会生气、会害羞、甚至会吃醋（嫉妒）。
* **记忆与态度**：它记得你们的每一次互动，是从“陌生人”变成“挚友”，还是变成“死敌”。
* **真实的脾气**：当它生气时，LLM 的语气会变得冷漠；当它得意时，语气会变得傲娇。

让你的 Bot 不再是冷冰冰的机器，而是一个有血有肉的“数字生命”。

---

## 🌟 核心特性 | Features

### 🎭 12维度深度情感模型 (Deep Emotion Model)
v2.5 版本引入了心理学级别的精细情感建模，包含 **12 个独立维度**：

| 基础情感 (Plutchik) | 高级情感 (Advanced) | 状态指标 |
| :--- | :--- | :--- |
| 😄 **Joy** (喜悦) | 😤 **Pride** (得意/傲娇) |❤️ **Favor** (好感度) |
| 🤝 **Trust** (信任) | 😔 **Guilt** (内疚/愧疚) | 🔗 **Intimacy** (亲密度) |
| 😨 **Fear** (恐惧) | 😳 **Shame** (害羞/羞耻) | |
| 😲 **Surprise** (惊讶) | 🍋 **Envy** (嫉妒/吃醋) | |
| 😢 **Sadness** (悲伤) | | |
| 🤮 **Disgust** (厌恶) | | |
| 😡 **Anger** (愤怒) | | |
| 🤩 **Anticipation** (期待)| | |

### 🗣️ 动态语气引擎 (Dynamic Tone Engine)
这是本插件最强大的功能。系统不仅是记录数值，还会**强制干预 LLM 的回复风格**：
* 如果 **嫉妒(Envy)** 值高 $\rightarrow$ 系统注入提示：*"语气要酸溜溜的、不服气的。"*
* 如果 **害羞(Shame)** 值高 $\rightarrow$ 系统注入提示：*"说话要结巴、含糊其辞，想找地缝钻进去。"*
* 如果 **愤怒(Anger)** 值高 $\rightarrow$ 系统注入提示：*"使用简短有力的句子，表现出不耐烦和攻击性。"*

### 🛡️ 黑名单与惩罚机制 (Blacklist System)
* **自动拉黑**：如果用户长期辱骂或进行负面互动，导致 **好感度(Favor)** 降至最低（默认 -100），Bot 会心碎并自动拉黑该用户。
* **拒绝服务**：进入黑名单的用户，无论说什么，Bot 都会冷漠回复“您已加入黑名单”，不再消耗 Token 进行 LLM 思考。
* **救赎之道**：只有管理员可以通过指令重置或修改情感，将其从黑名单中“解救”出来。

### 📊 关系演化系统
Bot 会根据**亲密度**和**当前态度**自动定义你们的关系：
* *高亲密 + 正向态度* = **挚友 / 知己**
* *高亲密 + 负向态度* = **死敌 / 宿敌** (最熟悉的陌生人)
* *低亲密 + 中立态度* = **陌生人**

---

## 🛠️ 安装与配置 | Installation

1.  将插件文件夹放入 `AstrBot/data/plugins/` 目录。
2.  重启 AstrBot。
3.  (可选) 在 `data/config/astrbot_plugin_emotionai/config.json` 中微调配置（或通过 WebUI 配置）：

```json
{
  "session_based": false,      // 是否分群计算（False=全服共享好感，True=每个群独立）
  "favour_min": -100,          // 好感度下限（达到此值触发黑名单）
  "favour_max": 100,           // 好感度上限
  "change_min": -10,           // 单次对话情感扣分上限
  "change_max": 5,             // 单次对话情感加分上限
  "admin_qq_list": ["123456"], // 管理员QQ列表（必填，否则无法使用管理指令）
  "plugin_priority": 100000    // 优先级（建议保持很高，确保先于其他插件执行）
}
````

-----

## 💻 指令手册 | Commands

### 🙋‍♂️ 用户指令 (Everyone)

| 指令 | 示例 | 说明 |
| :--- | :--- | :--- |
| `/好感度` | `/好感度` | 查看自己当前所有的情感数值、关系等级和趋势。 |
| `/状态显示` | `/状态显示` | 开关。开启后，Bot 每次回复末尾都会带上情感变化小尾巴。 |
| `/好感排行` | `/好感排行 5` | 查看全服最受宠爱的用户 TOP N。 |
| `/负好感排行` | `/负好感排行 5` | 查看全服被讨厌的用户 TOP N (公开处刑)。 |
| `/黑名单统计` | `/黑名单统计` | 查看当前有多少人被关进了小黑屋。 |

### 👮 管理员指令 (Admin Only)

> 💡 **提示**：所有涉及 `<用户ID>` 的指令，如果是私聊或在群里 @对方，可以直接生效，或者手动输入纯数字 ID。

| 指令 | 格式 | 说明 |
| :--- | :--- | :--- |
| **/设置情感** | `/设置情感 <ID> <维度> <值>` | **上帝之手**。直接修改任意情感维度。支持中文名。<br>示例：`/设置情感 123456 嫉妒 90` (让某人瞬间让 Bot 吃醋) |
| **/重置好感** | `/重置好感 <ID>` | **一键重生**。清空该用户所有数据，并**移除黑名单**状态。 |
| **/查看好感** | `/查看好感 <ID>` | 偷窥他人的情感状态面板。 |
| **/备份数据** | `/备份数据` | 手动触发一次数据备份。 |

-----

## ❓ 常见问题 | FAQ

**Q: 为什么机器人突然只回我一句话，也不理我了？** A: 恭喜你，你可能把好感度刷到 -100 了，触发了**黑名单机制**。请联系管理员使用 `/重置好感` 或 `/设置情感` 把你的好感度拉回及格线以上。

**Q: 怎么让机器人表现出“傲娇”性格？** A: 你可以作为管理员，手动设置它的初始状态：
`/设置情感 <你的ID> 骄傲 80`
`/设置情感 <你的ID> 害羞 50`
`/设置情感 <你的ID> 好感 60`
然后和它对话，你会发现它的语气完全变了！

**Q: 为什么我看不到新增的“内疚、嫉妒”等情感？** A: 请确保插件已更新到 **v2.5.0+**。使用 `/好感度` 指令时，只有当这些数值不为 0 时，或者在详细面板中才会显示。

-----

## 📅 更新日志 | Changelog

### v2.5.1 (Current)

  * 🐛 **修复**：修复了 LLM 机械性复制提示词示例导致好感度误降的问题。
  * 🔧 **优化**：调整了黑名单拦截逻辑，管理员不再拥有全局豁免权（方便测试），但保留了使用救援指令的权限。

### v2.5.0 (Major Update)

  * ✨ **新增情感**：加入 **得意(Pride), 内疚(Guilt), 害羞(Shame), 嫉妒(Envy)** 四种高级情感。
  * 🧠 **语气引擎**：实装 `Tone Engine`，根据当前主导情感（Dominant Emotion）强制改写 LLM 的 system prompt，实现“听其言知其心”。
  * 🔄 **指令重构**：废弃旧版指令，统一使用 `/设置情感` 控制一切。

### v2.4.0

  * 🛡️ **黑名单**：实装自动黑名单拦截系统。
  * 📐 **算法升级**：关系计算逻辑重构，解决了“仇恨值高但因为互动多而被判定为挚友”的逻辑漏洞。

-----

<div align="center">
  <br>
  Made with ❤️ by <a href="https://github.com/tengtian3"><strong>腾天</strong></a>
  <br>
  © 2025 EmotionAI Project

</div>
