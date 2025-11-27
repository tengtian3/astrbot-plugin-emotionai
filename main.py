import json
import re
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from dataclasses import dataclass, asdict

# ==================== 核心导入 (保持 v3.3.1 稳定架构) ====================
# 1. 事件相关：使用别名 event_filter 避免冲突
from astrbot.api.event import filter as event_filter
from astrbot.api.event import AstrMessageEvent

# 2. 核心组件：从 api.all 导入
from astrbot.api.all import Context, Star, register, AstrBotConfig, logger

# 3. 辅助工具：从 api.star 导入
from astrbot.api.star import StarTools

# 4. LLM 相关
from astrbot.api.provider import LLMResponse, ProviderRequest


# ==================== 数据结构定义 ====================

@dataclass
class EmotionalState:
    """情感状态数据类"""
    joy: int = 0
    trust: int = 0
    fear: int = 0
    surprise: int = 0
    sadness: int = 0
    disgust: int = 0
    anger: int = 0
    anticipation: int = 0
    pride: int = 0
    guilt: int = 0
    shame: int = 0
    envy: int = 0
    
    favor: int = 0
    intimacy: int = 0
    relationship: str = "陌生人"
    attitude: str = "中立"
    is_blacklisted: bool = False
    
    interaction_count: int = 0
    last_interaction: float = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    
    show_status: bool = False
    show_thought: bool = False 
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalState':
        return cls(**data)


@dataclass
class RankingEntry:
    """排行榜条目"""
    rank: int
    user_key: str
    average_score: float
    favor: int
    intimacy: int
    display_name: str


# ==================== 数据迁移管理器 ====================

class DataMigrationManager:
    @staticmethod
    def migrate_user_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """迁移用户数据到最新版本"""
        converted = {}
        for key, value in data.items():
            if isinstance(value, dict) and "emotions" in value:
                state = EmotionalState()
                if "emotions" in value:
                    emotions = value["emotions"]
                    for k in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "pride", "guilt", "shame", "envy"]:
                        setattr(state, k, emotions.get(k, 0))
                
                if "states" in value:
                    states = value["states"]
                    state.favor = states.get("favor", 0)
                    state.intimacy = states.get("intimacy", 0)
                
                state.relationship = value.get("relationship", "陌生人")
                state.attitude = value.get("attitude", "中立")
                state.is_blacklisted = value.get("is_blacklisted", False)
                
                if "behavior" in value:
                    behavior = value["behavior"]
                    state.interaction_count = behavior.get("interaction_count", 0)
                    state.last_interaction = behavior.get("last_interaction", 0)
                    state.positive_interactions = behavior.get("positive_interactions", 0)
                    state.negative_interactions = behavior.get("negative_interactions", 0)
                
                if "settings" in value:
                    settings = value["settings"]
                    state.show_status = settings.get("show_status", False)
                    state.show_thought = settings.get("show_thought", False)
                
                converted[key] = state.to_dict()
            else:
                default_state = EmotionalState().to_dict()
                for k, v in default_state.items():
                    if k not in value:
                        value[k] = v
                converted[key] = value
        return converted

    @staticmethod
    def get_data_version(data: Dict[str, Any]) -> str:
        return "3.4"


# ==================== 内部管理器类 ====================

class TTLCache:
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.lock = asyncio.Lock()
        self.access_count = 0
        self.hit_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            self.access_count += 1
            if key in self.cache:
                value, expires_at = self.cache[key]
                if time.time() < expires_at:
                    self.hit_count += 1
                    self.cache[key] = (value, time.time() + self.default_ttl)
                    return value
                else:
                    del self.cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        async with self.lock:
            await self._cleanup_expired()
            if len(self.cache) >= self.max_size:
                del self.cache[min(self.cache.keys(), key=lambda k: self.cache[k][1])]
            ttl = ttl or self.default_ttl
            self.cache[key] = (value, time.time() + ttl)
    
    async def _cleanup_expired(self):
        current_time = time.time()
        for k in [k for k, (_, t) in self.cache.items() if current_time >= t]:
            del self.cache[k]
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self.lock:
            hit_rate = (self.hit_count / self.access_count * 100) if self.access_count > 0 else 0
            return {
                "total_entries": len(self.cache),
                "access_count": self.access_count,
                "hit_count": self.hit_count,
                "hit_rate": round(hit_rate, 2)
            }
    
    async def clear(self):
        async with self.lock:
            self.cache.clear()


class UserStateManager:
    def __init__(self, data_path: Path, cache: TTLCache = None):
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.user_data = self._load_data("user_emotion_data.json")
        self.dirty_keys = set()
        self.last_save_time = time.time()
        self.save_interval = 60
        self.lock = asyncio.Lock()
        self.cache = cache
        
    def _load_data(self, filename: str) -> Dict[str, Any]:
        path = self.data_path / filename
        if not path.exists(): return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return DataMigrationManager.migrate_user_data(data)
        except Exception as e:
            logger.warning(f"数据加载异常: {e}")
            return {}
    
    async def get_user_state(self, user_key: str) -> EmotionalState:
        async with self.lock:
            if user_key in self.user_data:
                return EmotionalState.from_dict(self.user_data[user_key])
            return EmotionalState()
    
    async def update_user_state(self, user_key: str, state: EmotionalState):
        async with self.lock:
            self.user_data[user_key] = state.to_dict()
            self.dirty_keys.add(user_key)
            
        if self.cache:
            await self.cache.set(f"state_{user_key}", state)
            
        await self._check_auto_save()
    
    async def _check_auto_save(self):
        current_time = time.time()
        if (current_time - self.last_save_time >= self.save_interval and self.dirty_keys):
            await self.force_save()
    
    async def force_save(self):
        async with self.lock:
            if self.dirty_keys:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._save_data, "user_emotion_data.json", self.user_data)
                self.dirty_keys.clear()
                self.last_save_time = time.time()
    
    def _save_data(self, filename: str, data: Dict[str, Any]):
        path = self.data_path / filename
        temp_path = path.with_suffix('.tmp')
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_path.replace(path)
        except Exception as e:
            logger.error(f"保存数据失败: {e}")


class RankingManager:
    def __init__(self, user_state_manager):
        self.user_state_manager = user_state_manager
        self.cache = TTLCache(default_ttl=60, max_size=10)
    
    async def get_average_ranking(self, limit: int = 10, reverse: bool = True) -> List[RankingEntry]:
        cache_key = f"ranking_{limit}_{reverse}"
        cached_result = await self.cache.get(cache_key)
        if cached_result: return cached_result
        
        averages = []
        async with self.user_state_manager.lock:
            for user_key, data in self.user_state_manager.user_data.items():
                state = EmotionalState.from_dict(data)
                avg = (state.favor + state.intimacy) / 2
                averages.append((user_key, avg, state.favor, state.intimacy))
        
        averages.sort(key=lambda x: x[1], reverse=reverse)
        entries = []
        for i, (user_key, avg, favor, intimacy) in enumerate(averages[:limit], 1):
            entries.append(RankingEntry(i, user_key, avg, favor, intimacy, self._format_user_display(user_key)))
        
        await self.cache.set(cache_key, entries)
        return entries
    
    def _format_user_display(self, user_key: str) -> str:
        if '_' in user_key:
            try: return f"用户{user_key.split('_', 1)[1]}"
            except ValueError: pass
        return f"用户{user_key}"
    
    async def get_global_stats(self) -> Dict[str, Any]:
        cache_key = "global_stats"
        cached_result = await self.cache.get(cache_key)
        if cached_result: return cached_result
        
        async with self.user_state_manager.lock:
            users = self.user_state_manager.user_data.values()
            total_users = len(users)
            if total_users == 0: return {"total_users": 0, "blacklisted_count": 0, "total_interactions": 0, "average_favor": 0, "average_intimacy": 0}
            
            total_interactions = sum(EmotionalState.from_dict(d).interaction_count for d in users)
            avg_favor = sum(EmotionalState.from_dict(d).favor for d in users) / total_users
            avg_intimacy = sum(EmotionalState.from_dict(d).intimacy for d in users) / total_users
            blacklisted_count = sum(1 for d in users if EmotionalState.from_dict(d).is_blacklisted)
        
        stats = {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "average_favor": round(avg_favor, 2),
            "average_intimacy": round(avg_intimacy, 2),
            "blacklisted_count": blacklisted_count
        }
        await self.cache.set(cache_key, stats, ttl=30)
        return stats


class EmotionAnalyzer:
    EMOTION_DISPLAY_NAMES = {
        "joy": "喜悦", "trust": "信任", "fear": "恐惧", "surprise": "惊讶",
        "sadness": "悲伤", "disgust": "厌恶", "anger": "愤怒", "anticipation": "期待",
        "pride": "得意", "guilt": "内疚", "shame": "害羞", "envy": "嫉妒"
    }
    
    TONE_INSTRUCTIONS = {
        "joy": "语气愉快、充满热情和活力。多使用积极词汇。",
        "trust": "语气平和、真诚且令人安心。展现可靠。",
        "fear": "语气紧张、谨慎或不安。表现出犹豫。",
        "surprise": "语气震惊、难以置信或充满好奇。",
        "sadness": "语气低落、消沉。句子简短，无力。",
        "disgust": "语气厌烦、抗拒甚至带有生理性不适。",
        "anger": "语气愤怒、急躁、有攻击性。句子简短有力。",
        "anticipation": "语气期待、急切。关注未来。",
        "pride": "语气自信、骄傲甚至有点自大。",
        "guilt": "语气歉疚、卑微。不断道歉或解释。",
        "shame": "语气害羞、尴尬。说话结巴或含糊。",
        "envy": "语气酸溜溜、不服气。表现出矛盾心理。"
    }
    
    @classmethod
    def get_dominant_emotions(cls, state: EmotionalState, count: int = 2) -> List[Tuple[str, int]]:
        """获取主导情感（返回前N个）"""
        emotions = {k: getattr(state, k) for k in cls.EMOTION_DISPLAY_NAMES.keys()}
        return sorted([(k, v) for k, v in emotions.items() if v > 0], key=lambda x: x[1], reverse=True)[:count]
    
    @classmethod
    def get_emotional_profile(cls, state: EmotionalState) -> Dict[str, Any]:
        """获取完整的情感档案"""
        top_emotions = cls.get_dominant_emotions(state, 2)
        
        dominant_emotion = "中立"
        dominant_key = None
        if top_emotions:
            dominant_key = top_emotions[0][0]
            dominant_emotion = cls.EMOTION_DISPLAY_NAMES.get(dominant_key, "中立")
            
        secondary_emotion = None
        secondary_key = None
        if len(top_emotions) > 1:
            secondary_key = top_emotions[1][0]
            if top_emotions[1][1] > top_emotions[0][1] * 0.3:
                secondary_emotion = cls.EMOTION_DISPLAY_NAMES.get(secondary_key, "")

        intensity = top_emotions[0][1] if top_emotions else 0
        
        all_vals = [getattr(state, k) for k in cls.EMOTION_DISPLAY_NAMES.keys()]
        total_intensity = min(100, sum(all_vals) // 2)
        
        if state.favor > state.intimacy: relationship_trend = "好感领先"
        elif state.intimacy > state.favor: relationship_trend = "亲密度领先"
        else: relationship_trend = "平衡发展"
            
        total_interactions = state.interaction_count
        positive_ratio = (state.positive_interactions / total_interactions * 100) if total_interactions > 0 else 0
            
        return {
            "dominant_emotion": dominant_emotion,
            "dominant_key": dominant_key,
            "secondary_emotion": secondary_emotion,
            "secondary_key": secondary_key,
            "emotion_intensity": intensity,
            "total_intensity": total_intensity,
            "relationship_trend": relationship_trend,
            "positive_ratio": positive_ratio
        }


# ==================== 命令处理器类 ====================

class UserCommandHandler:
    def __init__(self, plugin):
        self.plugin = plugin
    
    async def show_emotional_state(self, event: AstrMessageEvent):
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        if state.is_blacklisted:
             yield event.plain_result("【系统提示】您已被列入黑名单，无法查看详细状态。")
             event.stop_event()
             return
        yield event.plain_result(self.plugin._format_emotional_state(state))
        event.stop_event()
    
    async def toggle_status_display(self, event: AstrMessageEvent):
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.show_status = not state.show_status
        await self.plugin.user_manager.update_user_state(user_key, state)
        yield event.plain_result(f"【状态显示】已{'开启' if state.show_status else '关闭'}")
        event.stop_event()
    
    async def toggle_thought_display(self, event: AstrMessageEvent):
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.show_thought = not state.show_thought
        await self.plugin.user_manager.update_user_state(user_key, state)
        yield event.plain_result(f"【心理显示】已{'开启' if state.show_thought else '关闭'}（仅对自己生效）")
        event.stop_event()
    
    async def show_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        try: limit = max(1, min(int(num), 20))
        except ValueError: limit = 10
        rankings = await self.plugin.ranking_manager.get_average_ranking(limit, True)
        if not rankings:
            yield event.plain_result("【排行榜】暂无数据")
            event.stop_event()
            return
        lines = [f"【好感度 TOP {limit}】", "="*18]
        for e in rankings:
            trend = "↑" if e.average_score > 0 else "↓"
            lines.append(f"{e.rank}. {e.display_name}\n   均值: {e.average_score:.1f} {trend} (好感 {e.favor}|亲密 {e.intimacy})")
        stats = await self.plugin.ranking_manager.get_global_stats()
        lines.extend(["", "【全服统计】", f"用户: {stats['total_users']} | 黑名单: {stats['blacklisted_count']}", f"互动: {stats['total_interactions']}"])
        yield event.plain_result("\n".join(lines))
        event.stop_event()
    
    async def show_negative_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        try: limit = max(1, min(int(num), 20))
        except ValueError: limit = 10
        rankings = await self.plugin.ranking_manager.get_average_ranking(limit, False)
        lines = [f"【好感度 BOTTOM {limit}】", "="*18]
        for e in rankings:
            lines.append(f"{e.rank}. {e.display_name}\n   均值: {e.average_score:.1f} (好感 {e.favor}|亲密 {e.intimacy})")
        yield event.plain_result("\n".join(lines))
        event.stop_event()

    async def show_blacklist_stats(self, event: AstrMessageEvent):
        stats = await self.plugin.ranking_manager.get_global_stats()
        c, t = stats['blacklisted_count'], stats['total_users']
        r = (c / t * 100) if t > 0 else 0
        yield event.plain_result(f"【黑名单统计】\n人数: {c}/{t}\n占比: {r:.1f}%\n提示: 好感度过低自动触发")
        event.stop_event()
    
    async def show_cache_stats(self, event: AstrMessageEvent):
        s = await self.plugin.cache.get_stats()
        yield event.plain_result(f"【缓存统计】\n条目: {s['total_entries']}\n命中: {s['hit_count']}/{s['access_count']} ({s['hit_rate']}%)")
        event.stop_event()


class AdminCommandHandler:
    def __init__(self, plugin):
        self.plugin = plugin
    
    def _resolve_user_key(self, user_input: str) -> str:
        if self.plugin.session_based and '_' not in user_input:
            for k in self.plugin.user_manager.user_data:
                if k.endswith(f"_{user_input}"): return k
        return user_input
    
    async def set_emotion(self, event: AstrMessageEvent, user_input: str, dimension: str, value: str):
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
        try: val = int(value)
        except ValueError:
            yield event.plain_result("【错误】数值需为整数")
            event.stop_event()
            return
            
        target_key = dimension.lower()
        if target_key in self.plugin.CN_TO_EN_MAP: target_key = self.plugin.CN_TO_EN_MAP[target_key]
        
        if target_key not in asdict(EmotionalState()):
             yield event.plain_result(f"【错误】无效维度: {dimension}")
             event.stop_event()
             return

        if target_key == "favor":
            if not self.plugin.favour_min <= val <= self.plugin.favour_max:
                yield event.plain_result(f"【错误】好感度范围: {self.plugin.favour_min}~{self.plugin.favour_max}")
                event.stop_event()
                return
        elif not 0 <= val <= 100:
            yield event.plain_result(f"【错误】情感范围: 0~100")
            event.stop_event()
            return

        user_key = self._resolve_user_key(user_input)
        state = await self.plugin.user_manager.get_user_state(user_key)
        setattr(state, target_key, val)
        if target_key == "favor" and val > self.plugin.favour_min: state.is_blacklisted = False
            
        await self.plugin.user_manager.update_user_state(user_key, state)
        yield event.plain_result(f"【成功】{user_input} 的 [{dimension}] 已设为 {val}")
        event.stop_event()
    
    async def reset_favor(self, event: AstrMessageEvent, user_input: str):
        if not self.plugin._is_admin(event): return
        user_key = self._resolve_user_key(user_input)
        new_state = EmotionalState()
        new_state.is_blacklisted = False
        
        await self.plugin.user_manager.update_user_state(user_key, new_state)
        yield event.plain_result(f"【成功】{user_input} 情感已重置")
        event.stop_event()
    
    async def view_favor(self, event: AstrMessageEvent, user_input: str):
        if not self.plugin._is_admin(event): return
        user_key = self._resolve_user_key(user_input)
        state = await self.plugin.user_manager.get_user_state(user_key)
        yield event.plain_result(self.plugin._format_emotional_state(state))
        event.stop_event()
    
    async def backup_data(self, event: AstrMessageEvent):
        if not self.plugin._is_admin(event): return
        try:
            path = self.plugin._create_backup()
            yield event.plain_result(f"【成功】备份至: {path}")
        except Exception as e:
            yield event.plain_result(f"【错误】{str(e)}")
        event.stop_event()


# ==================== 主插件类 ====================

@register("EmotionAI", "腾天", "高级情感智能交互系统 v3.4", "3.4")
class EmotionAIPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._validate_and_init_config()
        
        data_dir = StarTools.get_data_dir() / "emotionai"
        self.cache = TTLCache(default_ttl=300, max_size=500)
        self.user_manager = UserStateManager(data_dir, self.cache)
        self.ranking_manager = RankingManager(self.user_manager)
        self.analyzer = EmotionAnalyzer()
        self.migration_manager = DataMigrationManager()
        
        self.CN_TO_EN_MAP = {v: k for k, v in EmotionAnalyzer.EMOTION_DISPLAY_NAMES.items()}
        self.CN_TO_EN_MAP.update({"好感": "favor", "好感度": "favor", "亲密": "intimacy", "亲密度": "intimacy", "骄傲": "pride", "愧疚": "guilt", "羞耻": "shame"})
        
        self.user_commands = UserCommandHandler(self)
        self.admin_commands = AdminCommandHandler(self)
        
        # 正则初始化
        self.emotion_pattern = re.compile(r"\[(?:\s*情感更新:)?\s*(.*?)\]", re.DOTALL)
        self.single_emotion_pattern = re.compile(r"(\w+|[\u4e00-\u9fa5]+)\s*[:：]\s*([+-]?\d+)")
        
        # [核心修复] 历史清洗正则：匹配 <thought> 块 (含 Markdown) 或 【当前情感状态】及其后的所有内容
        self.history_clean_pattern = re.compile(
            r"(?:```(?:xml|text)?\s*)?<(?:thought|thinking)>[\s\S]*?</(?:thought|thinking)>(?:\s*```)?|(?:\n*\s*【当前情感状态】[\s\S]*$)",
            re.IGNORECASE | re.MULTILINE
        )
        
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("EmotionAI v3.4 (Cognitive Resonance Engine) Loaded")
        
    def _validate_and_init_config(self):
        self.session_based = bool(self.config.get("session_based", False))
        self.favour_min = self.config.get("favour_min", -100)
        self.favour_max = self.config.get("favour_max", 100)
        if self.favour_max <= self.favour_min: self.favour_min, self.favour_max = -100, 100
        self.change_min = self.config.get("change_min", -10)
        self.change_max = self.config.get("change_max", 5)
        
        raw_list = self.config.get("admin_qq_list", [])
        self.admin_qq_list = [str(qq) for qq in raw_list if str(qq).isdigit()]
        self.plugin_priority = self.config.get("plugin_priority", 100000)
        
    async def _auto_save_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                await self.user_manager.force_save()
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"Auto-save failed: {e}")
                
    def _get_user_key(self, event: AstrMessageEvent) -> str:
        uid = event.get_sender_id()
        return f"{event.unified_msg_origin}_{uid}" if self.session_based else uid
    
    def _format_emotional_state(self, state: EmotionalState) -> str:
        p = self.analyzer.get_emotional_profile(state)
        return (f"【当前情感状态】\n==================\n"
                f"好感度：{state.favor} | 亲密度：{state.intimacy}\n"
                f"关系：{state.relationship} | 趋势：{p['relationship_trend']}\n"
                f"态度：{state.attitude} | 主导：{p['dominant_emotion']}\n"
                f"互动：{state.interaction_count}次 (正面 {p['positive_ratio']:.1f}%)\n\n"
                f"【情感维度详情】\n"
                f"  喜悦：{state.joy} | 信任：{state.trust} | 恐惧：{state.fear}\n"
                f"  惊讶：{state.surprise} | 悲伤：{state.sadness} | 厌恶：{state.disgust}\n"
                f"  愤怒：{state.anger} | 期待：{state.anticipation} | 得意：{state.pride}\n"
                f"  内疚：{state.guilt} | 害羞：{state.shame} | 嫉妒：{state.envy}")

    def _calculate_relationship_level(self, state: EmotionalState) -> str:
        score, att = state.intimacy, state.attitude
        if score < 20: return "陌生人"
        if att in ["溺爱", "喜爱", "友好"]:
            return "挚友" if score >= 80 else "好友" if score >= 60 else "朋友" if score >= 40 else "熟人"
        elif att in ["仇恨", "厌恶", "冷淡"]:
            return "死敌" if score >= 80 else "敌人" if score >= 60 else "交恶" if score >= 40 else "冷漠的熟人"
        return "老相识" if score >= 80 else "熟客" if score >= 60 else "熟人"
    
    def _calculate_attitude(self, state: EmotionalState) -> str:
        s, pos, neg = state.favor, max(1, self.favour_max), min(-1, self.favour_min)
        if s >= pos * 0.9: return "溺爱"
        if s >= pos * 0.6: return "喜爱"
        if s >= pos * 0.3: return "友好"
        if s <= neg * 0.9: return "仇恨"
        if s <= neg * 0.6: return "厌恶"
        if s <= neg * 0.3: return "冷淡"
        return "中立"
    
    def _get_interaction_frequency(self, state: EmotionalState) -> str:
        if state.interaction_count == 0: return "首次"
        days = (time.time() - state.last_interaction) / 86400
        return "频繁" if days < 1 else "经常" if days < 3 else "偶尔" if days < 7 else "稀少"

    # ==================== 核心逻辑 ====================
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """检查是否为管理员"""
        return event.role == "admin" or event.get_sender_id() in self.admin_qq_list

    def _create_backup(self) -> str:
        """创建数据备份"""
        data_dir = StarTools.get_data_dir() / "emotionai"
        backup_dir = data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        backup_name = f"emotionai_backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        backup_path.mkdir(exist_ok=True)
        
        for filename in ["user_emotion_data.json"]:
            src = data_dir / filename
            if src.exists():
                dst = backup_path / filename
                shutil.copy2(src, dst)
        
        return str(backup_path.relative_to(data_dir))

    @event_filter.event_message_type(event_filter.EventMessageType.ALL, priority=1000000)
    async def check_blacklist(self, event: AstrMessageEvent):
        if self._is_admin(event):
            msg = event.message_str.strip()
            if msg.startswith(("/重置好感", "/设置情感", "设置情感")): return

        user_key = self._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
        if state.is_blacklisted:
            yield event.plain_result("您已加入黑名单，请联系管理员移除")
            event.stop_event()

    @event_filter.on_llm_request(priority=100000)
    async def inject_emotional_context(self, event: AstrMessageEvent, req: ProviderRequest):
        user_key = self._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
        await self.cache.set(f"state_{user_key}", state)
        
        # 定义伪造样本的特征
        FAKE_USER_MSG = "（系统自检：请报告当前情感状态）"
        
        # [核武器 2.0] 伪装成底层协议握手 (Protocol Enforcement)
        FORCE_MSG = """
[PROTOCOL_ENFORCEMENT]
User input received.
INITIATING COGNITIVE PROCESS...
REQUIRED ACTION: Generate XML block.
FORMAT:
```xml
<thought>
[感知] ...
[评估] ...
[代谢] ...
[决策] ...
[更新] ...
</thought>
```

(System: Waiting for XML output...)
"""

        # 1. 历史清洗 (升级版：清洗 thought、伪造样本、以及旧的强制指令)
        # 【关键修复】这里必须缩进 8 个空格，确保在函数内部
        if hasattr(req, "contexts") and isinstance(req.contexts, list):
            cleaned_contexts = []
            skip_next = False
            
            for i, ctx in enumerate(req.contexts):
                if skip_next:
                    skip_next = False
                    continue

                if isinstance(ctx, dict) and "content" in ctx:
                    content_str = str(ctx["content"])
                    
                    # [新增] 清洗旧的强制指令，防止堆积导致模型麻木
                    if "[PROTOCOL_ENFORCEMENT]" in content_str:
                        continue # 直接丢弃旧指令
                        
                    # 清洗伪造样本
                    if FAKE_USER_MSG in content_str:
                        skip_next = True 
                        continue
                        
                    # 常规清洗 thought
                    clean_content = self.history_clean_pattern.sub("", content_str).strip()
                    if clean_content:
                        new_ctx = ctx.copy()
                        new_ctx["content"] = clean_content
                        cleaned_contexts.append(new_ctx)
                        
                elif isinstance(ctx, str):
                    # [新增] 字符串类型的清洗
                    if "[PROTOCOL_ENFORCEMENT]" in ctx:
                        continue
                        
                    if FAKE_USER_MSG in ctx:
                        skip_next = True
                        continue
                    clean_str = self.history_clean_pattern.sub("", ctx).strip()
                    if clean_str: cleaned_contexts.append(clean_str)
            
            req.contexts = cleaned_contexts

        # 2. 注入 System Prompt
        req.system_prompt += f"\n{self._build_cognitive_context(state)}"

        # 3. 【关键】注入带 <thought> 的 One-Shot 样本
        req.contexts.append({
            "role": "user",
            "content": FAKE_USER_MSG
        })
        
        req.contexts.append({
            "role": "assistant",
            # 已修正：闭合标签改为 </thought>，原代码中是 <thought>
            "content": """<thought>
[感知] 收到系统自检指令。
[评估] 当前运行平稳，心情平静。
[决策] 输出标准状态报告。
[更新] joy:0
</thought>
报告前辈，星绘的情感核心运转正常，随时准备为您服务。"""
        })

        # 4. 【核心修改】以 System 身份独立注入强制指令
        req.contexts.append({
            "role": "system", 
            "content": FORCE_MSG
        })

    def _build_cognitive_context(self, state: EmotionalState) -> str:
        # 1. 获取前三个主导情感 (Count=3)
        top_emotions = self.analyzer.get_dominant_emotions(state, count=3)
        
        # 2. 构建详细面板 (完整且有序)
        all_emotions_list = []
        for k, name in EmotionAnalyzer.EMOTION_DISPLAY_NAMES.items():
            val = getattr(state, k)
            all_emotions_list.append(f"[{name}:{val}]")
        emotion_status_str = " ".join(all_emotions_list)

        # 3. 构建三层混合语气要求
        tone_instruction = "保持正常对话语气。"
        if top_emotions:
            # --- 第一层：主导 ---
            k1, v1 = top_emotions[0]
            n1 = EmotionAnalyzer.EMOTION_DISPLAY_NAMES[k1]
            i1 = self.analyzer.TONE_INSTRUCTIONS.get(k1, "")
            
            status_desc = f"主导情感[{n1}](强度{v1}%)"
            guide_desc = f"主要表现出{i1}"
            
            if len(top_emotions) > 1:
                # --- 第二层：夹杂 ---
                k2, v2 = top_emotions[1]
                n2 = EmotionAnalyzer.EMOTION_DISPLAY_NAMES[k2]
                i2 = self.analyzer.TONE_INSTRUCTIONS.get(k2, "")
                
                status_desc += f"，夹杂[{n2}]"
                
                if len(top_emotions) > 2:
                    # --- 第三层：微带 ---
                    k3, v3 = top_emotions[2]
                    n3 = EmotionAnalyzer.EMOTION_DISPLAY_NAMES[k3]
                    i3 = self.analyzer.TONE_INSTRUCTIONS.get(k3, "")
                    
                    status_desc += f"，微带[{n3}]"
                    guide_desc += f"，同时{i2}，底层隐约透出{i3}"
                else:
                    # 只有两层的情况
                    guide_desc += f"，但隐约透出{i2}"
            
            tone_instruction = f"【混合语气要求】{status_desc}。请{guide_desc}。"

        return f"""
【当前情感与认知状态】
关系：{state.relationship} | 态度：{state.attitude}
好感度：{state.favor} | 亲密度：{state.intimacy}
**详细情感面板**：{emotion_status_str}

{tone_instruction}

【核心指令：认知共鸣引擎】
你拥有完整的情感自主权。在回复用户之前，你**必须**进行一次“内部心理活动”，且**必须**使用 <thought> 标签包裹。

1. **代谢机制（动态平衡系统）**：
   - **A. 负面消解**：若存在[愤怒/悲伤/厌恶/嫉妒]且当前互动良好，**必须**扣除数值（如 anger:-5）以模拟消气。
   - **B. 激情冷却（重要）**：[惊讶/害羞/恐惧/狂喜] 属于瞬时爆发情绪，不应长期维持高位。如果对话已经进入平稳阶段，**必须**让这些数值自然回落（如 surprise:-5, shame:-3），表现出情绪逐渐平复的过程。
   
2. **直接决定**：在 <thought> 中直接决定情感变化。

【输出格式强制协议】
1. **分离输出**：先完全输出 <thought>...</thought> 块，然后再开始输出对用户的回复。
2. **禁止省略**：无论对话多短（哪怕只是打招呼），都必须包含思维链。

【标准输出模板】
<thought>
1. 感知：用户在解释刚才的误会，原来他不是故意迟到的。
2. 评估：误会解除了，之前的[愤怒:20]显得没必要了；同时刚才看到他时的[惊讶:15]也随着对话进行而自然平复。
3. 代谢：(1)负面消解：因误会解除，大幅扣除愤怒。(2)激情冷却：惊讶感不再强烈，自然回落。
4. 决策：语气从之前的质问转为释然和关心。
5. 更新：anger:-10, surprise:-5, favor:2
</thought>
(你的回复内容...)

可用维度：joy, trust, fear, surprise, sadness, disgust, anger, anticipation, pride, guilt, shame, envy, favor, intimacy
情感变化范围：{self.change_min} ~ {self.change_max}
"""

    @event_filter.on_llm_response(priority=100000)
    async def process_emotional_update(self, event: AstrMessageEvent, resp: LLMResponse):
        user_key = self._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
        orig_text = resp.completion_text
        
        # [调试] 打印 LLM 的完整输出
        logger.info(f"[EmotionAI DEBUG] LLM 完整输出:\n{orig_text}")
        
        # 1. 提取思维链
        thought_pattern = re.compile(r"(?:```(?:xml|text)?\s*)?<(?:thought|thinking)>(.*?)</(?:thought|thinking)>(?:\s*```)?", re.DOTALL | re.IGNORECASE)
        thought_match = thought_pattern.search(orig_text)
        
        updates = {}
        
        if thought_match:
            thought_content = thought_match.group(1)
            logger.info(f"[EmotionAI DEBUG] 提取到的思维链内容:\n{thought_content}")
            
            # 扫描每一行
            lines = thought_content.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                
                # [关键逻辑] 只有包含 "更新" 或 "Update" 的行才会被处理
                if "更新" in line or "Update" in line:
                    logger.info(f"[EmotionAI DEBUG] >>> 命中更新行 [Line {i}]: {line}")
                    
                    # 在这一行里抓取数值
                    matches = self.single_emotion_pattern.findall(line)
                    logger.info(f"[EmotionAI DEBUG]     正则抓取结果: {matches}")
                    
                    for k, v in matches:
                        try:
                            k = k.lower()
                            if k in self.CN_TO_EN_MAP: k = self.CN_TO_EN_MAP[k]
                            updates[k] = int(v)
                        except ValueError: continue
                else:
                    # 打印被跳过的行，用于确认是否误判
                    # logger.debug(f"[EmotionAI DEBUG] 跳过非更新行: {line[:20]}...")
                    pass
        
        # 如果思维链里没找到更新（或者没有思维链），尝试从旧格式中查找
        if not updates:
            logger.warning("[EmotionAI DEBUG] 思维链中未找到更新，尝试全文扫描(旧兼容模式)...")
            matches = self.single_emotion_pattern.findall(orig_text)
            # 这里要注意：全文扫描极易误读（比如读到面板里的数值），所以仅作为最后的保底
            # 如果你不想让它读面板，可以把这块代码删掉，或者加上更严格的限制
            for k, v in matches:
                # 简单过滤：只接受带 + 或 - 的数值，或者在 "更新" 附近的数值
                # 但旧兼容模式很难完美，建议尽量依赖上面的思维链逻辑
                 try:
                    k = k.lower()
                    if k in self.CN_TO_EN_MAP: k = self.CN_TO_EN_MAP[k]
                    updates[k] = int(v)
                 except ValueError: continue

        # 移除思维链文本
        if not state.show_thought:
            orig_text = thought_pattern.sub("", orig_text)
        
        resp.completion_text = orig_text.strip()
        
        if updates:
            logger.info(f"[EmotionAI] 最终捕获的情感变更: {updates}")
            self._apply_emotion_updates(state, updates)
            self._update_interaction_stats(state, updates)
            await self.user_manager.update_user_state(user_key, state)
        else:
            logger.info("[EmotionAI] 本次无情感变更。")
        
        if state.show_status and updates:
            resp.completion_text += f"\n\n{self._format_emotional_state(state)}"

    def _apply_emotion_updates(self, state: EmotionalState, updates: Dict[str, int]):
        # 定义一个简单的截断函数，限制变化幅度
        def _clamp(delta: int) -> int:
            return max(self.change_min, min(self.change_max, delta))

        all_dims = list(EmotionAnalyzer.TONE_INSTRUCTIONS.keys())
        for dim in all_dims:
            if dim in updates:
                # 1. 获取 LLM 输出的变化值
                raw_delta = updates[dim]
                # 2. 强制截断 (例如: 输出 20, 强行变成 5; 输出 -50, 强行变成 -10)
                safe_delta = _clamp(raw_delta)
                
                # 3. 应用更新并限制最终结果在 0-100 之间
                val = getattr(state, dim) + safe_delta
                setattr(state, dim, max(0, min(100, val)))
        
        if "favor" in updates:
            # 好感度同样应用截断
            safe_delta = _clamp(updates["favor"])
            state.favor = max(self.favour_min, min(self.favour_max, state.favor + safe_delta))
            
        if "intimacy" in updates:
            # 亲密度同样应用截断
            safe_delta = _clamp(updates["intimacy"])
            state.intimacy = max(0, min(100, state.intimacy + safe_delta))
            
        # 检查黑名单逻辑保持不变
        if state.favor <= self.favour_min and not state.is_blacklisted:
            state.is_blacklisted = True
            logger.info(f"[EmotionAI] 用户 {state} 触发黑名单")

    def _update_interaction_stats(self, state: EmotionalState, updates: Dict[str, int]):
        state.interaction_count += 1
        state.last_interaction = time.time()
        
        pos_score = sum(updates.get(k, 0) for k in ["joy", "trust", "favor"] if updates.get(k,0)>0)
        neg_score = sum(updates.get(k, 0) for k in ["anger", "disgust", "sadness"] if updates.get(k,0)>0)
        
        if pos_score > neg_score: state.positive_interactions += 1
        elif neg_score > pos_score: state.negative_interactions += 1
        
        state.attitude = self._calculate_attitude(state)
        state.relationship = self._calculate_relationship_level(state)

    # ==================== 注册命令 ====================
    
    @event_filter.command("好感度", priority=5)
    @event_filter.regex(r"^好感度$")
    async def cmd_show_state(self, event: AstrMessageEvent):
        async for r in self.user_commands.show_emotional_state(event): yield r

    @event_filter.command("状态显示", priority=5)
    async def cmd_toggle_status(self, event: AstrMessageEvent):
        async for r in self.user_commands.toggle_status_display(event): yield r
        
    @event_filter.command("心理显示", priority=5) 
    async def cmd_toggle_thought(self, event: AstrMessageEvent):
        async for r in self.user_commands.toggle_thought_display(event): yield r

    @event_filter.command("好感排行", priority=5)
    async def cmd_rank(self, event: AstrMessageEvent, num: str = "10"):
        async for r in self.user_commands.show_favor_ranking(event, num): yield r

    @event_filter.command("负好感排行", priority=5)
    async def cmd_bad_rank(self, event: AstrMessageEvent, num: str = "10"):
        async for r in self.user_commands.show_negative_favor_ranking(event, num): yield r
        
    @event_filter.command("黑名单统计", priority=5)
    async def cmd_black_stats(self, event: AstrMessageEvent):
        async for r in self.user_commands.show_blacklist_stats(event): yield r

    @event_filter.command("缓存统计", priority=5)
    async def cmd_cache(self, event: AstrMessageEvent):
        async for r in self.user_commands.show_cache_stats(event): yield r

    @event_filter.command("设置情感", priority=5)
    async def cmd_set_emotion(self, event: AstrMessageEvent, user: str, dim: str, val: str):
        async for r in self.admin_commands.set_emotion(event, user, dim, val): yield r

    @event_filter.command("重置好感", priority=5)
    async def cmd_reset_favor(self, event: AstrMessageEvent, user: str):
        async for r in self.admin_commands.reset_favor(event, user): yield r

    @event_filter.command("查看好感", priority=5)
    async def cmd_view_favor(self, event: AstrMessageEvent, user: str):
        async for r in self.admin_commands.view_favor(event, user): yield r

    @event_filter.command("备份数据", priority=5)
    async def cmd_backup(self, event: AstrMessageEvent):
        async for r in self.admin_commands.backup_data(event): yield r

    async def terminate(self):
        if hasattr(self, 'auto_save_task'): self.auto_save_task.cancel()
        await self.user_manager.force_save()
        logger.info("EmotionAI 插件已安全关闭")