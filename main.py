import json
import re
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from dataclasses import dataclass, asdict

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import AstrBotConfig, logger


# ==================== 数据结构定义 ====================

@dataclass
class EmotionalState:
    """情感状态数据类"""
    # 基础情感维度
    joy: int = 0
    trust: int = 0
    fear: int = 0
    surprise: int = 0
    sadness: int = 0
    disgust: int = 0
    anger: int = 0
    anticipation: int = 0
    
    # 高级情感维度
    pride: int = 0
    guilt: int = 0
    shame: int = 0
    envy: int = 0
    
    # 复合状态
    favor: int = 0
    intimacy: int = 0
    
    # 关系状态
    relationship: str = "陌生人"
    attitude: str = "中立"
    
    # 黑名单状态
    is_blacklisted: bool = False
    
    # 行为统计
    interaction_count: int = 0
    last_interaction: float = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    
    # 用户设置
    show_status: bool = False
    
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
        converted = {}
        for key, value in data.items():
            if isinstance(value, dict) and "emotions" in value:
                state = EmotionalState()
                if "emotions" in value:
                    emotions = value["emotions"]
                    state.joy = emotions.get("joy", 0)
                    state.trust = emotions.get("trust", 0)
                    state.fear = emotions.get("fear", 0)
                    state.surprise = emotions.get("surprise", 0)
                    state.sadness = emotions.get("sadness", 0)
                    state.disgust = emotions.get("disgust", 0)
                    state.anger = emotions.get("anger", 0)
                    state.anticipation = emotions.get("anticipation", 0)
                    state.pride = emotions.get("pride", 0)
                    state.guilt = emotions.get("guilt", 0)
                    state.shame = emotions.get("shame", 0)
                    state.envy = emotions.get("envy", 0)
                
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
        return "3.0.1"


# ==================== 内部管理器类 ====================

class UserStateManager:
    """用户状态管理器"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.user_data = self._load_data("user_emotion_data.json")
        self.dirty_keys = set()
        self.last_save_time = time.time()
        self.save_interval = 60
        self.lock = asyncio.Lock()
        
    def _load_data(self, filename: str) -> Dict[str, Any]:
        path = self.data_path / filename
        if not path.exists(): return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return DataMigrationManager.migrate_user_data(data)
        except (json.JSONDecodeError, TypeError) as e:
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
            return {"total_entries": len(self.cache), "access_count": self.access_count, "hit_count": self.hit_count, "hit_rate": round(hit_rate, 2)}
    
    async def clear(self):
        async with self.lock:
            self.cache.clear()


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
        
        stats = {"total_users": total_users, "total_interactions": total_interactions, "average_favor": round(avg_favor, 2), "average_intimacy": round(avg_intimacy, 2), "blacklisted_count": blacklisted_count}
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
        emotions = {k: getattr(state, k) for k in cls.EMOTION_DISPLAY_NAMES.keys()}
        return sorted([(k, v) for k, v in emotions.items() if v > 0], key=lambda x: x[1], reverse=True)[:count]
    
    @classmethod
    def get_emotional_profile(cls, state: EmotionalState) -> Dict[str, Any]:
        top_emotions = cls.get_dominant_emotions(state, 2)
        dominant_emotion = cls.EMOTION_DISPLAY_NAMES.get(top_emotions[0][0], "中立") if top_emotions else "中立"
        intensity = top_emotions[0][1] if top_emotions else 0
        
        # 2. 次要情感（用于混合情感分析）
        secondary_emotion = None
        secondary_key = None
        if len(top_emotions) > 1:
            secondary_key = top_emotions[1][0]
            if top_emotions[1][1] > top_emotions[0][1] * 0.3:
                secondary_emotion = cls.EMOTION_DISPLAY_NAMES.get(secondary_key, "")

        all_vals = [getattr(state, k) for k in cls.EMOTION_DISPLAY_NAMES.keys()]
        total_intensity = min(100, sum(all_vals) // 2)
        
        if state.favor > state.intimacy: relationship_trend = "好感领先"
        elif state.intimacy > state.favor: relationship_trend = "亲密度领先"
        else: relationship_trend = "平衡发展"
            
        total_interactions = state.interaction_count
        positive_ratio = (state.positive_interactions / total_interactions * 100) if total_interactions > 0 else 0
            
        return {
            "dominant_emotion": dominant_emotion,
            "dominant_key": top_emotions[0][0] if top_emotions else None,
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
        await self.plugin.cache.set(f"state_{user_key}", state)
        yield event.plain_result(f"【成功】{user_input} 的 [{dimension}] 已设为 {val}")
        event.stop_event()
    
    async def reset_favor(self, event: AstrMessageEvent, user_input: str):
        if not self.plugin._is_admin(event): return
        user_key = self._resolve_user_key(user_input)
        new_state = EmotionalState()
        await self.plugin.user_manager.update_user_state(user_key, new_state)
        await self.plugin.cache.set(f"state_{user_key}", new_state)
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

@register("EmotionAI", "腾天", "高级情感智能交互系统 v3.0", "3.0.1")
class EmotionAIPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._validate_and_init_config()
        
        data_dir = StarTools.get_data_dir() / "emotionai"
        self.user_manager = UserStateManager(data_dir)
        self.ranking_manager = RankingManager(self.user_manager)
        self.analyzer = EmotionAnalyzer()
        self.migration_manager = DataMigrationManager()
        
        self.CN_TO_EN_MAP = {v: k for k, v in EmotionAnalyzer.EMOTION_DISPLAY_NAMES.items()}
        self.CN_TO_EN_MAP.update({"好感": "favor", "好感度": "favor", "亲密": "intimacy", "亲密度": "intimacy", "骄傲": "pride", "愧疚": "guilt", "羞耻": "shame"})
        
        self.cache = TTLCache(default_ttl=300, max_size=500)
        self.user_commands = UserCommandHandler(self)
        self.admin_commands = AdminCommandHandler(self)
        
        self.emotion_pattern = re.compile(r"\[情感更新:\s*(.*?)\]", re.DOTALL)
        self.single_emotion_pattern = re.compile(r"(\w+|[\u4e00-\u9fa5]+):\s*([+-]?\d+)")
        self.thought_pattern = re.compile(r"<thought>.*?</thought>", re.DOTALL)
        
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("EmotionAI v3.0.1 (Cognitive Resonance Engine) Loaded")
        
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
                f"  喜悦：{state.joy} | 信任：{state.trust} | 恐惧：{state.fear} | 惊讶：{state.surprise}\n"
                f"  悲伤：{state.sadness} | 厌恶：{state.disgust} | 愤怒：{state.anger} | 期待：{state.anticipation}\n"
                f"  得意：{state.pride} | 内疚：{state.guilt} | 害羞：{state.shame} | 嫉妒：{state.envy}")

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

    # ==================== 核心逻辑 (V3.0.1 修复版) ====================
    
    @filter.event_message_type(filter.EventMessageType.ALL, priority=1000000)
    async def check_blacklist(self, event: AstrMessageEvent):
        if self._is_admin(event):
            msg = event.message_str.strip()
            if msg.startswith(("/重置好感", "/设置情感", "设置情感")): return

        user_key = self._get_user_key(event)
        state = await self.user_manager.get_user_state(user_key)
        if state.is_blacklisted:
            yield event.plain_result("您已加入黑名单，请联系管理员移除")
            event.stop_event()

    @filter.on_llm_request(priority=100000)
    async def inject_emotional_context(self, event: AstrMessageEvent, req: ProviderRequest):
        user_key = self._get_user_key(event)
        state = await self.cache.get(f"state_{user_key}") or await self.user_manager.get_user_state(user_key)
        await self.cache.set(f"state_{user_key}", state)
        req.system_prompt += f"\n{self._build_cognitive_context(state)}"

    def _build_cognitive_context(self, state: EmotionalState) -> str:
        profile = self.analyzer.get_emotional_profile(state)
        
        tone_instruction = "保持正常对话语气。"
        if profile['dominant_key']:
            primary_inst = self.analyzer.TONE_INSTRUCTIONS.get(profile['dominant_key'], "")
            tone_msg = f"主导情感[{profile['dominant_emotion']}](强度{profile['emotion_intensity']}%)。"
            
            if profile['secondary_key']:
                sec_inst = self.analyzer.TONE_INSTRUCTIONS.get(profile['secondary_key'], "")
                tone_msg += f" 但同时也夹杂着[{profile['secondary_emotion']}]。"
                tone_instruction = f"【混合语气要求】{tone_msg} 请主要表现出{primary_inst}，但隐约透出{sec_inst}。"
            else:
                tone_instruction = f"【语气要求】{tone_msg} {primary_inst}"

        # [核心修改 v3.0.1] 
        # 1. 显式列出所有非零情感数值，让 LLM 看到"已有的愤怒"。
        # 2. 明确指示 LLM：如果现在的互动（正向）抵消了旧情绪（负向），必须输出负值。
        
        active_emotions = []
        for k in EmotionAnalyzer.EMOTION_DISPLAY_NAMES.keys():
            val = getattr(state, k)
            if val > 0:
                name = EmotionAnalyzer.EMOTION_DISPLAY_NAMES[k]
                active_emotions.append(f"[{name}:{val}]")
        
        emotion_status_str = " ".join(active_emotions) if active_emotions else "无明显情感波动"

        return f"""
【情感与认知状态】
当前关系：{state.relationship} | 态度：{state.attitude}
综合好感度：{state.favor} | 亲密度：{state.intimacy}
主导情感：{profile['dominant_emotion']} (强度{profile['emotion_intensity']}%)
**当前详细面板**：{emotion_status_str}

{tone_instruction}

【核心指令：认知共鸣引擎】
你拥有完整的情感自主权。在回复用户之前，你必须进行一次“内部心理活动”：
1. **感知 (Perception)**: 用户的意图是什么？
2. **评估 (Appraisal)**: 这句话如何影响你当前的情感？
3. **代谢 (Metabolism)**: **重要！** 检查你的“当前详细面板”。例如，如果你看到有 [愤怒:20]，但用户现在让你很开心，说明愤怒被抵消了。你**必须**输出负值（如 anger:-5）来减少它。如果不输出负值，愤怒会永远停留在 20！
4. **决策 (Decision)**: 决定你的回复风格。

【输出格式要求】
请务必严格按照以下 XML 格式输出你的思考过程，然后再输出回复：


你的回复内容... [情感更新: anger:-10, trust:2]

可用维度：joy, trust, fear, surprise, sadness, disgust, anger, anticipation, pride, guilt, shame, envy, favor, intimacy
范围：{self.change_min} ~ {self.change_max}
"""

    @filter.on_llm_response(priority=100000)
    async def process_emotional_update(self, event: AstrMessageEvent, resp: LLMResponse):
        user_key = self._get_user_key(event)
        orig_text = resp.completion_text
        
        thought_match = self.thought_pattern.search(orig_text)
        if thought_match:
            thought_content = thought_match.group(0)
            logger.debug(f"[EmotionAI] 🧠 思维链: {thought_content}")
            orig_text = orig_text.replace(thought_content, "").strip()
            
        updates = self._parse_emotion_updates(orig_text)
        if updates:
            tag_match = self.emotion_pattern.search(orig_text)
            if tag_match:
                orig_text = orig_text.replace(tag_match.group(0), "").strip()
        
        resp.completion_text = orig_text
        
        state = await self.user_manager.get_user_state(user_key)
        self._apply_emotion_updates(state, updates)
        self._update_interaction_stats(state, updates)
        
        await self.user_manager.update_user_state(user_key, state)
        await self.cache.set(f"state_{user_key}", state)
        
        if state.show_status and updates:
            resp.completion_text += f"\n\n{self._format_emotional_state(state)}"

    def _parse_emotion_updates(self, text: str) -> Dict[str, int]:
        updates = {}
        match = self.emotion_pattern.search(text)
        if match:
            for k, v in self.single_emotion_pattern.findall(match.group(1)):
                try:
                    k = k.lower()
                    if k in self.CN_TO_EN_MAP: k = self.CN_TO_EN_MAP[k]
                    updates[k] = int(v)
                except ValueError: continue
        return updates

    def _apply_emotion_updates(self, state: EmotionalState, updates: Dict[str, int]):
        all_dims = list(EmotionAnalyzer.TONE_INSTRUCTIONS.keys())
        for dim in all_dims:
            if dim in updates:
                val = getattr(state, dim) + updates[dim]
                setattr(state, dim, max(0, min(100, val)))
        
        if "favor" in updates:
            state.favor = max(self.favour_min, min(self.favour_max, state.favor + updates["favor"]))
        if "intimacy" in updates:
            state.intimacy = max(0, min(100, state.intimacy + updates["intimacy"]))
            
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