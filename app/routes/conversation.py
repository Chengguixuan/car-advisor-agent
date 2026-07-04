"""会话管理 API — 删除会话。"""

import logging

from fastapi import APIRouter, HTTPException

from car_advisor.src.session_store import delete_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.delete("/{thread_id}")
async def delete_conversation(thread_id: str):
    """删除指定会话及其所有消息。"""
    try:
        delete_session(thread_id)
        logger.info("DELETE /conversation/%s", thread_id)
        return {"status": "deleted", "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
