"""购车智能体状态定义。

使用 LangGraph 的 StateGraph 机制，定义多轮对话中需要跨节点传递和
累积的所有状态字段。
"""

from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CarAdvisorState(TypedDict):
    """购车智能体全局状态。

    各节点通过读写此 State 来协作完成"需求收集 → 车型搜索 →
    对比分析 → 最终推荐"的完整流程。

    Attributes:
        messages:
            对话消息列表（含 system / user / assistant）。
            使用 Annotated[list, add_messages] 语义：
            - 每次节点返回 {"messages": [...]} 时，新消息**追加**到历史中
            - 不会覆盖已有的 messages

        user_profile:
            从对话中提取的结构化用户需求。包含以下可选字段：
            - budget:          预算区间，如 "15-20万"
            - car_type:        车型偏好：轿车 / SUV / MPV / 不确定
            - fuel_type:       能源类型：燃油 / 混动 / 纯电 / 不确定
            - usage:           主要用途：通勤 / 家庭 / 商务 / 综合
            - concerns:        核心关注点列表，如 ["安全", "油耗", "空间"]
            - annual_mileage:  年行驶里程（公里）
            - has_charger:     是否有充电桩：是 / 否 / 不确定
            - passengers:      常载人数
            - hold_years:      计划持有年限
            初始为 None，由"需求分析"节点填充和更新。

        candidates:
            候选车型列表，由"车型搜索"节点根据 user_profile 从
            car_data.json 中筛选得出。每项为 car_data.json 中的
            完整车型字典。初始为 None。

        final_recommendation:
            最终推荐结果，由"推荐生成"节点综合候选车型和对话上下文
            后输出。包含：
            - understanding:      对用户需求的理解摘要
            - recommended_models: 推荐车型列表（含推荐理由）
            - follow_up_question: 追问问题（信息不足时）
            初始为 None。

        search_count:
            搜索次数计数器。每次"车型搜索"节点执行后 +1。
            用于防止 Agent 无限循环搜索（结合 conditional edge 判断）。
            初始值为 0。

        exclusions:
            用户明确排除的项。例如 ["日系", "纯电", "CVT变速箱"]。
            由 intent="preference" 的 chatbot 节点更新。

        preferences:
            用户表达的偏好。例如 ["空间大", "省油", "安全性好"]。
            由 intent="preference" 的 chatbot 节点更新。

        car_opinions:
            用户对具体车型的评价。例如 {"本田CR-V": "隔音差", "比亚迪宋PLUS": "性价比高"}。
            由 intent="opinion" 的 chatbot 节点记录。
    """

    messages: Annotated[list, add_messages]
    user_profile: Optional[dict[str, Any]]
    candidates: Optional[list[dict[str, Any]]]
    final_recommendation: Optional[dict[str, Any]]
    search_count: int
    # 多意图支持
    exclusions: list[str]            # 排除项，如 ["日系", "纯电"]
    preferences: list[str]           # 偏好项，如 ["空间大", "省油"]
    car_opinions: dict[str, Any]     # 对具体车型的评价，如 {"CR-V": "隔音差"}
    # 上下文压缩
    history_summary: Optional[str]   # 历史摘要，超过 5 轮时自动生成
