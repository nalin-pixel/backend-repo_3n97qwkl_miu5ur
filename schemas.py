"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any

# Example schemas (replace with your own):

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    region: Optional[str] = Field(None, description="Region or curriculum system")
    language: str = Field("en", description="Preferred language code: en, fa, de, fr")

class Quiz(BaseModel):
    """
    Quiz session document
    Collection name: "quiz"
    """
    id: str = Field(..., description="Quiz session id")
    user: str = Field(..., description="User identifier (name or email)")
    topic: str = Field(..., description="Topic key")
    difficulty: int = Field(1, ge=1, le=5)
    count: int = Field(5, ge=1, le=20)
    lang: str = Field("en", description="Language for questions")
    questions: List[dict] = Field(default_factory=list, description="List of {id, question, answer}")

class QuizResult(BaseModel):
    """
    Quiz result per session
    Collection name: "quiz_result"
    """
    quiz_id: str
    user: str
    topic: str
    lang: str
    total: int
    correct: int
    score: float
    points: int
    details: List[dict]

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Add your own schemas here:
# --------------------------------------------------

# Note: The Flames database viewer will automatically:
# 1. Read these schemas from GET /schema endpoint
# 2. Use them for document validation when creating/editing
# 3. Handle all database operations (CRUD) directly
# 4. You don't need to create any database endpoints!
