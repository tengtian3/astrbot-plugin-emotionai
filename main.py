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
    
    # 复合状态
    favor: int = 0
    intimacy: int = 0
    
    # 关系状态
    relationship: str = "陌生人"
    attitude: str = "中立"
    
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
    """数据迁移管理器"""
    
    @staticmethod
    def migrate_user_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """迁移用户数据到最新版本"""
        converted = {}
        for key, value in data.items():
            if isinstance(value, dict) and "emotions" in value:
                # 旧格式转换
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
                
                if "states" in value:
                    states = value["states"]
                    state.favor = states.get("favor", 0)
                    state.intimacy = states.get("intimacy", 0)
                
                state.relationship = value.get("relationship", "陌生人")
                state.attitude = value.get("attitude", "中立")
                
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
                converted[key] = value
        return converted
    
    @staticmethod
    def get_data_version(data: Dict[str, Any]) -> str:
        """获取数据版本"""
        # 检查是否有版本标记
        if "_version" in data:
            return data["_version"]
        
        # 通过数据结构判断版本
        if data and isinstance(next(iter(data.values())), dict) and "emotions" in next(iter(data.values())):
            return "1.0"  # 旧版本格式
        else:
            return "2.0"  # 新版本格式


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
        """加载数据文件"""
        path = self.data_path / filename
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return DataMigrationManager.migrate_user_data(data)
        except (json.JSONDecodeError, TypeError) as e:
            if isinstance(e, json.JSONDecodeError):
                logger.warning(f"JSON解析错误，可能文件格式损坏: {e}")
            else:
                logger.warning(f"加载数据时遇到TypeError，可能文件格式陈旧或损坏: {e}")
            return {}
    
    async def get_user_state(self, user_key: str) -> EmotionalState:
        """获取用户情感状态"""
        async with self.lock:
            if user_key in self.user_data:
                return EmotionalState.from_dict(self.user_data[user_key])
            return EmotionalState()
    
    async def update_user_state(self, user_key: str, state: EmotionalState):
        """更新用户状态"""
        async with self.lock:
            self.user_data[user_key] = state.to_dict()
            self.dirty_keys.add(user_key)
        await self._check_auto_save()
    
    async def _check_auto_save(self):
        """检查是否需要自动保存"""
        current_time = time.time()
        if (current_time - self.last_save_time >= self.save_interval and 
            self.dirty_keys):
            await self.force_save()
    
    async def force_save(self):
        """强制保存所有脏数据"""
        async with self.lock:
            if self.dirty_keys:
                self._save_data("user_emotion_data.json", self.user_data)
                self.dirty_keys.clear()
                self.last_save_time = time.time()
    
    def _save_data(self, filename: str, data: Dict[str, Any]):
        """保存数据到文件"""
        path = self.data_path / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def clear_all_data(self):
        """清空所有用户数据"""
        async with self.lock:
            self.user_data.clear()
            self.dirty_keys.clear()
            await self.force_save()


class TTLCache:
    """带过期时间的缓存"""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.lock = asyncio.Lock()
        self.access_count = 0
        self.hit_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self.lock:
            self.access_count += 1
            if key in self.cache:
                value, expires_at = self.cache[key]
                if time.time() < expires_at:
                    self.hit_count += 1
                    return value
                else:
                    del self.cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        async with self.lock:
            # 清理过期缓存
            await self._cleanup_expired()
            
            # 如果超过最大大小，删除最旧的
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            self.cache[key] = (value, expires_at)
    
    async def _cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expires_at) in self.cache.items()
            if current_time >= expires_at
        ]
        for key in expired_keys:
            del self.cache[key]
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        async with self.lock:
            hit_rate = (self.hit_count / self.access_count * 100) if self.access_count > 0 else 0
            return {
                "total_entries": len(self.cache),
                "access_count": self.access_count,
                "hit_count": self.hit_count,
                "hit_rate": round(hit_rate, 2)
            }
    
    async def clear(self):
        """清空缓存"""
        async with self.lock:
            self.cache.clear()


class RankingManager:
    """排行榜管理器"""
    
    def __init__(self, user_state_manager):
        self.user_state_manager = user_state_manager
        self.cache = TTLCache(default_ttl=60, max_size=10)
    
    async def get_average_ranking(self, limit: int = 10, reverse: bool = True) -> List[RankingEntry]:
        """获取好感度和亲密度的平均值排行榜"""
        cache_key = f"ranking_{limit}_{reverse}"
        
        # 尝试从缓存获取
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # 计算排行榜
        averages = []
        
        async with self.user_state_manager.lock:
            for user_key, data in self.user_state_manager.user_data.items():
                state = EmotionalState.from_dict(data)
                avg = (state.favor + state.intimacy) / 2
                averages.append((user_key, avg, state.favor, state.intimacy))
        
        # 排序
        averages.sort(key=lambda x: x[1], reverse=reverse)
        
        # 转换为 RankingEntry 对象
        entries = []
        for i, (user_key, avg, favor, intimacy) in enumerate(averages[:limit], 1):
            display_name = self._format_user_display(user_key)
            entries.append(RankingEntry(
                rank=i,
                user_key=user_key,
                average_score=avg,
                favor=favor,
                intimacy=intimacy,
                display_name=display_name
            ))
        
        # 存入缓存
        await self.cache.set(cache_key, entries)
        
        return entries
    
    def _format_user_display(self, user_key: str) -> str:
        """格式化用户显示名称"""
        if '_' in user_key:
            try:
                session_id, user_id = user_key.split('_', 1)
                return f"用户{user_id}"
            except ValueError:
                return f"用户{user_key}"
        return f"用户{user_key}"
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        cache_key = "global_stats"
        
        # 尝试从缓存获取
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        async with self.user_state_manager.lock:
            total_users = len(self.user_state_manager.user_data)
            total_interactions = 0
            avg_favor = 0
            avg_intimacy = 0
            
            for user_key, data in self.user_state_manager.user_data.items():
                state = EmotionalState.from_dict(data)
                total_interactions += state.interaction_count
                avg_favor += state.favor
                avg_intimacy += state.intimacy
            
            if total_users > 0:
                avg_favor /= total_users
                avg_intimacy /= total_users
        
        stats = {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "average_favor": round(avg_favor, 2),
            "average_intimacy": round(avg_intimacy, 2)
        }
        
        # 存入缓存
        await self.cache.set(cache_key, stats, ttl=30)
        
        return stats


class EmotionAnalyzer:
    """情感分析器"""
    
    EMOTION_DISPLAY_NAMES = {
        "joy": "喜悦",
        "trust": "信任", 
        "fear": "恐惧",
        "surprise": "惊讶",
        "sadness": "悲伤",
        "disgust": "厌恶",
        "anger": "愤怒",
        "anticipation": "期待"
    }
    
    @classmethod
    def get_dominant_emotion(cls, state: EmotionalState) -> str:
        """获取主导情感"""
        emotions = {
            "喜悦": state.joy,
            "信任": state.trust,
            "恐惧": state.fear,
            "惊讶": state.surprise,
            "悲伤": state.sadness,
            "厌恶": state.disgust,
            "愤怒": state.anger,
            "期待": state.anticipation
        }
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant[0] if dominant[1] > 0 else "中立"
    
    @classmethod
    def get_emotional_profile(cls, state: EmotionalState) -> Dict[str, Any]:
        """获取完整的情感档案"""
        dominant_emotion = cls.get_dominant_emotion(state)
        
        # 计算情感强度
        total_emotion = sum([
            state.joy, state.trust, state.fear, state.surprise,
            state.sadness, state.disgust, state.anger, state.anticipation
        ])
        emotion_intensity = min(100, total_emotion // 2)
        
        # 判断关系趋势
        if state.favor > state.intimacy:
            relationship_trend = "好感领先"
        elif state.intimacy > state.favor:
            relationship_trend = "亲密度领先"
        else:
            relationship_trend = "平衡发展"
            
        # 计算互动质量
        total_interactions = state.interaction_count
        if total_interactions > 0:
            positive_ratio = (state.positive_interactions / total_interactions) * 100
        else:
            positive_ratio = 0
            
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_intensity,
            "relationship_trend": relationship_trend,
            "positive_ratio": positive_ratio
        }


# ==================== 命令处理器类 ====================

class UserCommandHandler:
    """用户命令处理器"""
    
    def __init__(self, plugin):
        self.plugin = plugin
    
    async def show_emotional_state(self, event: AstrMessageEvent):
        """显示情感状态"""
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        response_text = self.plugin._format_emotional_state(state)
        yield event.plain_result(response_text)
        event.stop_event()
    
    async def toggle_status_display(self, event: AstrMessageEvent):
        """切换状态显示开关"""
        user_key = self.plugin._get_user_key(event)
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.show_status = not state.show_status
        await self.plugin.user_manager.update_user_state(user_key, state)
        
        status_text = "开启" if state.show_status else "关闭"
        yield event.plain_result(f"【状态显示】已{status_text}状态显示")
        event.stop_event()
    
    async def show_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示好感度排行榜"""
        try:
            limit = min(int(num), 20)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("【错误】排行数量必须是一个正整数（最大20）。")
            event.stop_event()
            return

        rankings = await self.plugin.ranking_manager.get_average_ranking(limit, True)
        
        if not rankings:
            yield event.plain_result("【排行榜】当前没有任何用户数据。")
            event.stop_event()
            return

        response_lines = [f"【好感度平均值 TOP {limit} 排行榜】", "=================="]
        for entry in rankings:
            trend = "↑" if entry.average_score > 0 else "↓"
            line = (
                f"{entry.rank}. {entry.display_name}\n"
                f"   平均值: {entry.average_score:.1f} {trend} (好感 {entry.favor} | 亲密 {entry.intimacy})"
            )
            response_lines.append(line)
        
        # 添加全局统计
        stats = await self.plugin.ranking_manager.get_global_stats()
        response_lines.extend([
            "",
            "【全局统计】",
            f"   总用户数: {stats['total_users']} | 总互动: {stats['total_interactions']}",
            f"   平均好感: {stats['average_favor']} | 平均亲密: {stats['average_intimacy']}"
        ])
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()
    
    async def show_negative_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示负好感排行榜"""
        try:
            limit = min(int(num), 20)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("【错误】排行数量必须是一个正整数（最大20）。")
            event.stop_event()
            return

        rankings = await self.plugin.ranking_manager.get_average_ranking(limit, False)
        
        if not rankings:
            yield event.plain_result("【排行榜】当前没有任何用户数据。")
            event.stop_event()
            return

        response_lines = [f"【好感度平均值 BOTTOM {limit} 排行榜】", "=================="]
        for entry in rankings:
            line = (
                f"{entry.rank}. {entry.display_name}\n"
                f"   平均值: {entry.average_score:.1f} (好感 {entry.favor} | 亲密 {entry.intimacy})"
            )
            response_lines.append(line)
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()


class AdminCommandHandler:
    """管理员命令处理器"""
    
    def __init__(self, plugin):
        self.plugin = plugin
    
    def _resolve_user_key(self, user_input: str) -> str:
        """解析用户输入的用户标识符"""
        if self.plugin.session_based:
            if '_' in user_input:
                return user_input
            else:
                for user_key in self.plugin.user_manager.user_data:
                    if user_key.endswith(f"_{user_input}"):
                        return user_key
                return user_input
        else:
            return user_input
    
    async def set_favor(self, event: AstrMessageEvent, user_input: str, value: str):
        """设置好感度"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        try:
            favor_value = int(value)
            if not self.plugin.favour_min <= favor_value <= self.plugin.favour_max:
                yield event.plain_result(f"【错误】好感度值必须在 {self.plugin.favour_min} 到 {self.plugin.favour_max} 之间。")
                event.stop_event()
                return
        except ValueError:
            yield event.plain_result("【错误】好感度值必须是数字")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        
        state = await self.plugin.user_manager.get_user_state(user_key)
        state.favor = favor_value
        
        await self.plugin.user_manager.update_user_state(user_key, state)
        
        # 更新缓存
        await self.plugin.cache.set(f"state_{user_key}", state)
        
        mode_info = "（会话模式）" if self.plugin.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的好感度已设置为 {favor_value}")
        event.stop_event()
    
    async def reset_favor(self, event: AstrMessageEvent, user_input: str):
        """重置用户好感度状态"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
            
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        new_state = EmotionalState()
        
        await self.plugin.user_manager.update_user_state(user_key, new_state)
        
        # 更新缓存
        await self.plugin.cache.set(f"state_{user_key}", new_state)
        
        mode_info = "（会话模式）" if self.plugin.session_based else ""
        yield event.plain_result(f"【成功】用户 {user_input}{mode_info} 的好感度状态已完全重置")
        event.stop_event()
    
    async def reset_plugin(self, event: AstrMessageEvent):
        """重置插件所有数据"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        # 重置所有数据
        await self.plugin.user_manager.clear_all_data()
        await self.plugin.cache.clear()
        
        logger.info("管理员执行了插件重置操作")
        
        yield event.plain_result("【成功】插件所有数据已重置")
        event.stop_event()
    
    async def view_favor(self, event: AstrMessageEvent, user_input: str):
        """管理员查看指定用户的好感状态"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        if not user_input or not user_input.strip():
            yield event.plain_result("【错误】用户标识符不能为空")
            event.stop_event()
            return
        
        # 解析用户标识符
        user_key = self._resolve_user_key(user_input)
        state = await self.plugin.user_manager.get_user_state(user_key)
        
        response_lines = [
            f"【用户 {user_input} 情感状态】",
            "==================",
            f"用户标识: {user_key}",
            f"好感度: {state.favor} | 亲密度: {state.intimacy}",
            f"关系: {state.relationship} | 态度: {state.attitude}",
            f"互动次数: {state.interaction_count}",
            f"正面互动: {state.positive_interactions} | 负面互动: {state.negative_interactions}",
            f"最后互动: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.last_interaction)) if state.last_interaction > 0 else '从未互动'}",
            f"状态显示: {'开启' if state.show_status else '关闭'}",
            "",
            "【情感维度详情】",
            f"  喜悦: {state.joy} | 信任: {state.trust} | 恐惧: {state.fear} | 惊讶: {state.surprise}",
            f"  悲伤: {state.sadness} | 厌恶: {state.disgust} | 愤怒: {state.anger} | 期待: {state.anticipation}"
        ]
        
        yield event.plain_result("\n".join(response_lines))
        event.stop_event()
    
    async def backup_data(self, event: AstrMessageEvent):
        """备份插件数据"""
        if not self.plugin._is_admin(event):
            yield event.plain_result("【错误】需要管理员权限")
            event.stop_event()
            return
            
        try:
            backup_path = self.plugin._create_backup()
            yield event.plain_result(f"【成功】数据备份成功: {backup_path}")
            event.stop_event()
        except Exception as e:
            yield event.plain_result(f"【错误】备份失败: {str(e)}")
            event.stop_event()


# ==================== 主插件类 ====================

@register("EmotionAI", "腾天", "高级情感智能交互系统 v2.2", "2.2.0")
class EmotionAIPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # 配置验证和初始化
        self._validate_and_init_config()
        
        # 获取规范的数据目录
        data_dir = StarTools.get_data_dir() / "emotionai"
        
        # 初始化各个管理器
        self.user_manager = UserStateManager(data_dir)
        self.ranking_manager = RankingManager(self.user_manager)
        self.analyzer = EmotionAnalyzer()
        self.migration_manager = DataMigrationManager()
        
        # 缓存系统
        self.cache = TTLCache(default_ttl=300, max_size=500)
        
        # 命令处理器
        self.user_commands = UserCommandHandler(self)
        self.admin_commands = AdminCommandHandler(self)
        
        # 情感更新模式
        self.emotion_pattern = re.compile(r"\[情感更新:\s*(.*?)\]", re.DOTALL)
        self.single_emotion_pattern = re.compile(r"(\w+):\s*([+-]?\d+)")
        
        # 自动保存任务
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        
        logger.info(f"EmotionAI 插件初始化完成 - 优先级版本 (优先级: {self.plugin_priority})")
        
    def _validate_and_init_config(self):
        """验证配置并初始化配置参数"""
        errors = []
        
        # 获取配置值
        self.session_based = bool(self.config.get("session_based", False))
        
        # 验证好感度范围
        self.favour_min = self.config.get("favour_min", -100)
        self.favour_max = self.config.get("favour_max", 100)
        if self.favour_max <= self.favour_min:
            errors.append(f"favour_max({self.favour_max}) 必须大于 favour_min({self.favour_min})")
            self.favour_min, self.favour_max = -100, 100
        
        # 验证变化范围
        self.change_min = self.config.get("change_min", -10)
        self.change_max = self.config.get("change_max", 5)
        if self.change_max <= self.change_min:
            errors.append(f"change_max({self.change_max}) 必须大于 change_min({self.change_min})")
            self.change_min, self.change_max = -10, 5
        
        # 验证数值范围合理性
        if abs(self.favour_min) > 1000 or abs(self.favour_max) > 1000:
            errors.append("好感度范围过大，建议在 -1000 到 1000 之间")
            self.favour_min, self.favour_max = -100, 100
        
        if abs(self.change_min) > 100 or abs(self.change_max) > 100:
            errors.append("变化范围过大，建议在 -100 到 100 之间")
            self.change_min, self.change_max = -10, 5
        
        # 验证管理员列表
        self.admin_qq_list = self.config.get("admin_qq_list", [])
        for i, qq in enumerate(self.admin_qq_list):
            if not isinstance(qq, str) or not qq.isdigit():
                errors.append(f"管理员QQ号格式错误: {qq}")
                self.admin_qq_list[i] = None
        
        # 清理无效的管理员QQ号
        self.admin_qq_list = [qq for qq in self.admin_qq_list if qq is not None]
        
        # 验证插件优先级
        self.plugin_priority = self.config.get("plugin_priority", 100000)
        if not isinstance(self.plugin_priority, int) or self.plugin_priority < 0:
            errors.append("插件优先级必须是一个正整数")
            self.plugin_priority = 100000
        
        # 记录配置错误
        if errors:
            error_msg = "配置验证警告:\n" + "\n".join(f"• {error}" for error in errors)
            logger.warning(error_msg)
        
        logger.info(f"配置加载完成: session_based={self.session_based}, "
                   f"favour_range=[{self.favour_min}, {self.favour_max}], "
                   f"change_range=[{self.change_min}, {self.change_max}], "
                   f"admin_count={len(self.admin_qq_list)}, "
                   f"priority={self.plugin_priority}")
        
    async def _auto_save_loop(self):
        """自动保存循环"""
        while True:
            try:
                await asyncio.sleep(30)
                await self.user_manager.force_save()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动保存失败: {e}")
                
    def _get_user_key(self, event: AstrMessageEvent) -> str:
        """获取用户键"""
        user_id = event.get_sender_id()
        if self.session_based:
            session_id = event.unified_msg_origin
            return f"{session_id}_{user_id}"
        return user_id
        
    def _get_session_id(self, event: AstrMessageEvent) -> Optional[str]:
        """获取会话ID"""
        return event.unified_msg_origin if self.session_based else None
        
    def _format_emotional_state(self, state: EmotionalState) -> str:
        """格式化情感状态显示"""
        # 计算关系等级
        relationship_level = self._calculate_relationship_level(state)
        
        # 获取情感档案
        profile = self.analyzer.get_emotional_profile(state)
        
        # 计算互动频率
        frequency = self._get_interaction_frequency(state)
        
        # 构建状态显示
        lines = [
            "【当前情感状态】",
            "==================",
            f"好感度：{state.favor} | 亲密度：{state.intimacy}",
            f"关系：{relationship_level} | 趋势：{profile['relationship_trend']}",
            f"态度：{state.attitude} | 主导情感：{profile['dominant_emotion']}",
            f"互动：{state.interaction_count}次 ({frequency})",
            f"正面互动：{profile['positive_ratio']:.1f}%",
            "",
            "【情感维度详情】",
            f"  喜悦：{state.joy} | 信任：{state.trust} | 恐惧：{state.fear} | 惊讶：{state.surprise}",
            f"  悲伤：{state.sadness} | 厌恶：{state.disgust} | 愤怒：{state.anger} | 期待：{state.anticipation}"
        ]
        
        return "\n".join(lines)
        
    def _calculate_relationship_level(self, state: EmotionalState) -> str:
        """计算关系等级"""
        intimacy_score = state.intimacy
        
        if intimacy_score >= 80:
            return "亲密关系"
        elif intimacy_score >= 60:
            return "挚友"
        elif intimacy_score >= 40:
            return "朋友"
        elif intimacy_score >= 20:
            return "熟人"
        else:
            return "陌生人"
            
    def _get_interaction_frequency(self, state: EmotionalState) -> str:
        """获取互动频率描述"""
        if state.interaction_count == 0:
            return "首次互动"
            
        days_since_last = (time.time() - state.last_interaction) / (24 * 3600)
        if days_since_last < 1:
            return "频繁互动"
        elif days_since_last < 3:
            return "经常互动"
        elif days_since_last < 7:
            return "偶尔互动"
        else:
            return "稀少互动"
    
    # ==================== LLM集成 ====================
    
    @filter.on_llm_request(priority=100000)
    async def inject_emotional_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """注入情感上下文 - 高优先级确保先执行"""
        logger.debug(f"[EmotionAI] 开始处理LLM请求 - 优先级: {self.plugin_priority}")
        
        user_key = self._get_user_key(event)
        logger.debug(f"[EmotionAI] 用户标识: {user_key}")
        
        # 从缓存获取状态或从管理器获取
        state = await self.cache.get(f"state_{user_key}")
        if state is None:
            logger.debug(f"[EmotionAI] 缓存未命中，从管理器获取状态")
            state = await self.user_manager.get_user_state(user_key)
            await self.cache.set(f"state_{user_key}", state)
        else:
            logger.debug(f"[EmotionAI] 缓存命中，使用缓存状态")
        
        # 构建增强的情感上下文
        emotional_context = self._build_emotional_context(state)
        req.system_prompt += f"\n{emotional_context}"
        
        logger.debug(f"[EmotionAI] 已注入情感上下文到LLM请求")
        
    def _build_emotional_context(self, state: EmotionalState) -> str:
        """构建情感上下文 - 增强版提示词"""
        profile = self.analyzer.get_emotional_profile(state)
        
        return f"""
【情感状态上下文】
当前关系：{state.relationship} | 态度：{state.attitude}
综合好感度：{state.favor}/100 | 亲密度：{state.intimacy}/100
互动统计：{state.interaction_count}次 | 正面比例：{profile['positive_ratio']:.1f}%
主导情感：{profile['dominant_emotion']} (强度：{profile['emotion_intensity']}%)

【情感响应指导】
请根据对话内容和当前情感状态调整你的回应风格和语气。如果对话让你感到：
- 开心/愉快：适当增加 joy (1-3)
- 信任/安心：适当增加 trust (1-2)  
- 惊讶：适当增加 surprise (1-2)
- 期待：适当增加 anticipation (1-2)
- 不开心：适当增加 sadness (1-2)
- 生气：适当增加 anger (1-2)
- 害怕：适当增加 fear (1-2)
- 厌恶：适当增加 disgust (1-2)

【情感更新机制】
在回复的末尾使用以下格式更新情感状态（只在情感确实变化时使用）：
[情感更新: joy:1, trust:1, favor:2]

可用维度：
- 基础情感: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
- 关系状态: favor, intimacy

变化范围：{self.change_min} 到 {self.change_max}
示例：如果对话让你感到开心和信任，可以写：[情感更新: joy:2, trust:1]

【重要提示】
请务必在每次回复时考虑是否需要更新情感状态，即使是很小的变化也值得记录。
这有助于建立更真实、更连贯的情感连接。情感更新应该反映你对当前对话的真实感受。
"""
    
    @filter.on_llm_response(priority=100000)
    async def process_emotional_update(self, event: AstrMessageEvent, resp: LLMResponse):
        """处理情感更新 - 高优先级确保先执行"""
        logger.debug(f"[EmotionAI] 开始处理LLM响应 - 优先级: {self.plugin_priority}")
        
        user_key = self._get_user_key(event)
        logger.debug(f"[EmotionAI] 用户标识: {user_key}")
            
        original_text = resp.completion_text
        
        # 详细调试日志：记录原始响应
        logger.debug(f"[EmotionAI] 用户 {user_key} 的LLM原始响应: '{original_text}'")
        
        emotion_updates = self._parse_emotion_updates(original_text)
        
        # 详细调试日志：记录解析过程
        logger.debug(f"[EmotionAI] 开始解析情感更新，原始文本: '{original_text}'")
        logger.debug(f"[EmotionAI] 用户 {user_key} 的情感更新解析结果: {emotion_updates}")
        
        # 清理回复文本
        if emotion_updates:
            emotion_block = self.emotion_pattern.search(original_text)
            if emotion_block:
                resp.completion_text = original_text.replace(emotion_block.group(0), '').strip()
                logger.debug(f"[EmotionAI] 用户 {user_key} 已清理情感更新标记: {emotion_block.group(0)}")
                logger.debug(f"[EmotionAI] 清理后的响应文本: '{resp.completion_text}'")
        
        # 更新情感状态
        state = await self.user_manager.get_user_state(user_key)
        logger.debug(f"[EmotionAI] 用户 {user_key} 的当前状态: 好感度={state.favor}, 亲密度={state.intimacy}")
        
        self._apply_emotion_updates(state, emotion_updates)
        self._update_interaction_stats(state, emotion_updates)
        
        # 保存状态
        await self.user_manager.update_user_state(user_key, state)
        
        # 更新缓存
        await self.cache.set(f"state_{user_key}", state)
        
        # 详细调试日志：记录状态变化
        if emotion_updates:
            logger.info(f"[EmotionAI] 用户 {user_key} 情感状态已更新: {emotion_updates}")
            logger.debug(f"[EmotionAI] 用户 {user_key} 更新后状态: 好感度={state.favor}, 亲密度={state.intimacy}")
        else:
            logger.debug(f"[EmotionAI] 用户 {user_key} 本次对话无情感更新")
        
        # 添加状态显示（如果启用）
        if state.show_status and emotion_updates:
            status_text = f"\n\n{self._format_emotional_state(state)}"
            resp.completion_text += status_text
            logger.debug(f"[EmotionAI] 用户 {user_key} 已添加状态显示")
            
        logger.debug(f"[EmotionAI] LLM响应处理完成")
            
    def _parse_emotion_updates(self, text: str) -> Dict[str, int]:
        """解析情感更新"""
        emotion_updates = {}
        emotion_match = self.emotion_pattern.search(text)
        
        if emotion_match:
            logger.debug(f"[EmotionAI] 找到情感更新标记: {emotion_match.group(0)}")
            emotion_content = emotion_match.group(1)
            single_matches = self.single_emotion_pattern.findall(emotion_content)
            logger.debug(f"[EmotionAI] 情感更新内容匹配结果: {single_matches}")
            
            for emotion, value in single_matches:
                try:
                    change_value = int(value)
                    # 对好感度变化进行限制
                    if emotion.lower() == 'favor':
                        change_value = max(
                            self.change_min, 
                            min(self.change_max, change_value)
                        )
                    emotion_updates[emotion.lower()] = change_value
                    logger.debug(f"[EmotionAI] 解析到情感更新: {emotion}={change_value}")
                except ValueError:
                    logger.debug(f"[EmotionAI] 情感值解析失败: {emotion}={value}")
                    continue
        else:
            logger.debug(f"[EmotionAI] 未找到情感更新标记")
                    
        return emotion_updates
        
    def _apply_emotion_updates(self, state: EmotionalState, updates: Dict[str, int]):
        """应用情感更新"""
        emotion_attrs = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
        state_attrs = ['favor', 'intimacy']
        
        for attr in emotion_attrs:
            if attr in updates:
                new_value = getattr(state, attr) + updates[attr]
                setattr(state, attr, max(0, min(100, new_value)))
                logger.debug(f"[EmotionAI] 更新基础情感 {attr}: {getattr(state, attr) - updates[attr]} -> {getattr(state, attr)}")
        
        for attr in state_attrs:
            if attr in updates:
                new_value = getattr(state, attr) + updates[attr]
                if attr == 'favor':
                    setattr(state, attr, max(
                        self.favour_min, 
                        min(self.favour_max, new_value)
                    ))
                else:
                    setattr(state, attr, max(0, min(100, new_value)))
                logger.debug(f"[EmotionAI] 更新状态 {attr}: {getattr(state, attr) - updates[attr]} -> {getattr(state, attr)}")
        
    def _update_interaction_stats(self, state: EmotionalState, updates: Dict[str, int]):
        """更新互动统计"""
        state.interaction_count += 1
        state.last_interaction = time.time()
        logger.debug(f"[EmotionAI] 更新互动统计: 次数={state.interaction_count}")
        
        # 判断互动性质
        if updates:
            positive_emotions = sum([
                updates.get('joy', 0),
                updates.get('trust', 0), 
                updates.get('surprise', 0),
                updates.get('anticipation', 0)
            ])
            negative_emotions = sum([
                updates.get('fear', 0),
                updates.get('sadness', 0),
                updates.get('disgust', 0),
                updates.get('anger', 0)
            ])
            
            if positive_emotions > negative_emotions:
                state.positive_interactions += 1
                logger.debug(f"[EmotionAI] 记录正面互动")
            elif negative_emotions > positive_emotions:
                state.negative_interactions += 1
                logger.debug(f"[EmotionAI] 记录负面互动")
        else:
            # 如果没有情感更新，认为是中性互动
            logger.debug(f"[EmotionAI] 记录中性互动")
            pass
        
        # 更新关系状态
        old_relationship = state.relationship
        state.relationship = self._calculate_relationship_level(state)
        if old_relationship != state.relationship:
            logger.debug(f"[EmotionAI] 关系状态变化: {old_relationship} -> {state.relationship}")
    
    # ==================== 用户命令 ====================
    
    @filter.command("好感度", priority=5)
    async def show_emotional_state(self, event: AstrMessageEvent):
        """显示情感状态"""
        async for result in self.user_commands.show_emotional_state(event):
            yield result
        
    @filter.command("状态显示", priority=5)
    async def toggle_status_display(self, event: AstrMessageEvent):
        """切换状态显示开关"""
        async for result in self.user_commands.toggle_status_display(event):
            yield result
        
    # ==================== 排行榜命令 ====================
    
    @filter.command("好感排行", priority=5)
    async def show_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示好感度排行榜"""
        async for result in self.user_commands.show_favor_ranking(event, num):
            yield result
        
    @filter.command("负好感排行", priority=5)
    async def show_negative_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示负好感排行榜"""
        async for result in self.user_commands.show_negative_favor_ranking(event, num):
            yield result
        
    # ==================== 缓存统计命令 ====================
    
    @filter.command("缓存统计", priority=5)
    async def show_cache_stats(self, event: AstrMessageEvent):
        """显示缓存统计信息"""
        stats = await self.cache.get_stats()
        
        response = [
            "【缓存统计信息】",
            "==================",
            f"缓存条目: {stats['total_entries']}",
            f"访问次数: {stats['access_count']}",
            f"命中次数: {stats['hit_count']}",
            f"命中率: {stats['hit_rate']}%",
            f"",
            f"提示: 缓存用于提高情感状态读取性能"
        ]
        
        yield event.plain_result("\n".join(response))
        event.stop_event()
        
    # ==================== 管理员命令 ====================
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """检查管理员权限"""
        return event.role == "admin" or event.get_sender_id() in self.admin_qq_list
        
    @filter.command("设置好感", priority=5)
    async def admin_set_favor(self, event: AstrMessageEvent, user_input: str, value: str):
        """设置好感度"""
        async for result in self.admin_commands.set_favor(event, user_input, value):
            yield result
        
    @filter.command("重置好感", priority=5)
    async def admin_reset_favor(self, event: AstrMessageEvent, user_input: str):
        """重置用户好感度状态"""
        async for result in self.admin_commands.reset_favor(event, user_input):
            yield result
        
    @filter.command("重置插件", priority=5)
    async def admin_reset_plugin(self, event: AstrMessageEvent):
        """重置插件所有数据"""
        async for result in self.admin_commands.reset_plugin(event):
            yield result
    
    @filter.command("查看好感", priority=5)
    async def admin_view_favor(self, event: AstrMessageEvent, user_input: str):
        """管理员查看指定用户的好感状态"""
        async for result in self.admin_commands.view_favor(event, user_input):
            yield result
        
    @filter.command("备份数据", priority=5)
    async def admin_backup_data(self, event: AstrMessageEvent):
        """备份插件数据"""
        async for result in self.admin_commands.backup_data(event):
            yield result
            
    def _create_backup(self) -> str:
        """创建数据备份"""
        data_dir = StarTools.get_data_dir() / "emotionai"
        backup_dir = data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        backup_name = f"emotionai_backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        # 创建备份目录
        backup_path.mkdir(exist_ok=True)
        
        # 复制数据文件
        for filename in ["user_emotion_data.json"]:
            src = data_dir / filename
            if src.exists():
                dst = backup_path / filename
                shutil.copy2(src, dst)
        
        return str(backup_path.relative_to(data_dir))
        
    async def terminate(self):
        """插件终止时清理资源"""
        if hasattr(self, 'auto_save_task'):
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
                
        # 强制保存所有数据
        await self.user_manager.force_save()
        logger.info("EmotionAI 插件已安全关闭")