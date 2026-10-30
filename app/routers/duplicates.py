"""Duplicates router."""
from fastapi import APIRouter, Body
from app.dependencies import DbSession
from app.services.duplicate_service import list_duplicates, make_decision

router = APIRouter(tags=["Duplicates"])


@router.get(
    "/api/duplicates",
    response_model=list[dict],
    summary="List all candidate duplicate pairs with similarity scores",
)
def get_duplicates(db: DbSession):
    return list_duplicates(db)


@router.post(
    "/api/duplicates/{link_id}/decision",
    response_model=dict,
    summary="Confirm or split a duplicate pair (DD-3)",
)
def decide_duplicate(
    link_id: str,
    db: DbSession,
    payload: dict = Body(..., example={"decision": "Confirmed", "actor": "Coordinator"}),
):
    decision = payload.get("decision", "")
    actor = payload.get("actor", "Operator")
    return make_decision(db, link_id, decision, actor)
