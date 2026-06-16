"""Pydantic request models — FastAPI validates incoming JSON against these."""
from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class QuestionCreate(BaseModel):
    question: str


class AnswerCreate(BaseModel):
    answer: str


class FeedbackCreate(BaseModel):
    category: str = "General"
    message: str


class FeedbackResponse(BaseModel):
    response: str


class AnnounceCreate(BaseModel):
    message: str


class RoleUpdate(BaseModel):
    role: str  # investor | expert | admin
