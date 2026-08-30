from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Optional

class Contributor(BaseModel):
    factor: str
    label: str
    impact: float
    detail: str = ""
    severity: str = "low"

class RiskBreakdown(BaseModel):
    health_score: int
    risk_band: str
    contributors: list[Contributor] = []
    positives: list[Contributor] = []
    summary_line: str = ""
    as_of: Optional[str] = None

class MLBlock(BaseModel):
    churn_probability: float
    band: str
    top_features: list[dict[str, Any]] = []
    model_version: str = "unknown"

class PlaybookBlock(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    score: Optional[float] = None
    method: Optional[str] = None
    low_confidence: Optional[bool] = None

class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    analysis_id: Optional[int] = None
    created_at: Optional[str] = None
    # 7 LLM-contract keys
    summary: str
    top_reasons: list[str] = []
    recommended_action: str
    priority: str
    draft_message: str = ""
    playbook_citation: str = ""
    generated_by: str = "unknown"
    # context blocks
    customer: dict[str, Any]
    ml: MLBlock
    risk_breakdown: RiskBreakdown
    base_action: dict[str, Any]
    playbook: PlaybookBlock
    escalate: bool = False
    method_agreement: dict[str, Any] = {}
    signals_snapshot: dict[str, Any] = {}
    status: str = "new"

class DashboardRow(BaseModel):
    customer_id: int
    name: str
    plan_tier: str
    arr: float
    churn_probability: float
    health_score: int
    risk_band: str
    priority: str
    top_reason: str = ""
    days_to_renewal: Optional[int] = None
    renewal_date: Optional[str] = None
    last_analysis_at: Optional[str] = None
    escalate: bool = False
    recommended_action: Optional[str] = None

class CustomerDetail(BaseModel):
    customer: dict[str, Any]          # from get_signals identity fields
    signals: dict[str, Any]           # the scored signal values
    usage_series: list[dict[str, Any]] = []
    feedback_list: list[dict[str, Any]] = []
    last_analysis: Optional[AnalysisResult] = None

class ApproveRequest(BaseModel):
    analysis_id: Optional[int] = None      # default: latest for customer
    status: str = Field(pattern="^(approved|edited|dismissed)$")
    message: str = ""

class ApproveResponse(BaseModel):
    analysis_id: int
    status: str
    approved_message: str

class HealthResponse(BaseModel):
    status: str                        # "ok" | "degraded"
    components: dict[str, Any]
