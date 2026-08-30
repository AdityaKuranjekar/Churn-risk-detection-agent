from fastapi import APIRouter, Depends, HTTPException
from app.deps import get_db, require_auth
from app.services import orchestrator
from app.services.signals import CustomerNotFound
from app import schemas

router = APIRouter(prefix="/api", tags=["analysis"], dependencies=[Depends(require_auth)])

@router.post("/customers/{cid}/analyze", response_model=schemas.AnalysisResult)
def analyze(cid: int, db=Depends(get_db)):
    try:
        return orchestrator.analyze_customer(db, cid, force=True)
    except CustomerNotFound:
        raise HTTPException(404, f"customer {cid} not found")

@router.post("/customers/{cid}/approve", response_model=schemas.ApproveResponse)
def approve(cid: int, body: schemas.ApproveRequest, db=Depends(get_db)):
    aid = body.analysis_id
    if aid is None:
        last = orchestrator.get_last_analysis(db, cid)
        if not last: 
            raise HTTPException(404, "no analysis to approve")
        aid = last["analysis_id"]
    try:
        res = orchestrator.approve_analysis(db, aid, body.message, body.status)
        if not res:
            raise HTTPException(404, f"analysis {aid} not found")
        return res
    except KeyError:
        raise HTTPException(404, f"analysis {aid} not found")
