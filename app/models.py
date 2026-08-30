from datetime import datetime
from sqlalchemy import Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.orm import mapped_column

from app.db import Base, engine

class Customer(Base):
    __tablename__ = "customers"
    
    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String)
    plan_tier = mapped_column(String)
    plan_tier_ord = mapped_column(Integer)
    tenure_days = mapped_column(Integer)
    arr = mapped_column(Float)
    monthly_charges = mapped_column(Float)
    renewal_date = mapped_column(String)
    num_devices = mapped_column(Integer)
    age = mapped_column(Integer)
    usage_level = mapped_column(Float)
    last_login_days = mapped_column(Integer)
    payment_failures = mapped_column(Integer)
    support_contacts = mapped_column(Integer)
    engagement_score = mapped_column(Float)
    is_demo = mapped_column(Integer)

class UsageDaily(Base):
    __tablename__ = "usage_daily"
    
    id = mapped_column(Integer, primary_key=True)
    customer_id = mapped_column(Integer, index=True)
    date = mapped_column(String)
    active_minutes = mapped_column(Float)
    logins = mapped_column(Integer)
    sessions = mapped_column(Integer)
    feature_events = mapped_column(Integer)

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = mapped_column(Integer, primary_key=True)
    customer_id = mapped_column(Integer, index=True)
    date = mapped_column(String)
    channel = mapped_column(String)
    text = mapped_column(Text)
    sentiment = mapped_column(Float)
    is_complaint = mapped_column(Integer)

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = mapped_column(Integer, primary_key=True)
    customer_id = mapped_column(Integer, index=True)
    created_at = mapped_column(DateTime, default=datetime.utcnow)
    churn_probability = mapped_column(Float)
    health_score = mapped_column(Integer)
    risk_band = mapped_column(String)
    priority = mapped_column(String)
    recommended_action = mapped_column(String)
    escalate = mapped_column(Boolean, default=False)
    playbook_id = mapped_column(String, nullable=True)
    generated_by = mapped_column(String)  # gemini:*, fallback, low_risk_shortcut
    result_json = mapped_column(Text)     # full assembled dict
    status = mapped_column(String, default="new")  # new | approved | edited | dismissed
    approved_message = mapped_column(Text, nullable=True)
    approved_at = mapped_column(DateTime, nullable=True)

class Meta(Base):
    __tablename__ = "meta"
    
    key = mapped_column(String, primary_key=True)
    value = mapped_column(String)

def init_db():
    Base.metadata.create_all(engine)
