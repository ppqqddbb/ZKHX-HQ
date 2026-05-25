from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pydantic import BaseModel
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from config.settings import *

from code.sqlite import (
    SessionLocal,
    engine,
    Base,
    User
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/swagger",  # 前端访问路径
    StaticFiles(directory=StaticFiles_path),  # 本地 Swagger UI 静态文件目录
    name="swagger-ui"
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",  # 仍指向这个接口
        title=app.title + " - Swagger UI",
        swagger_js_url="/swagger/swagger-ui-bundle.js",
        swagger_css_url="/swagger/swagger-ui.css",
        swagger_favicon_url="/swagger/favicon-32x32.png",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_custom_openapi_json():
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
    )

# 创建表（如果不存在）
Base.metadata.create_all(bind=engine)

# =========================
# 使用 Argon2 加密
# =========================
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)


# =========================
# Pydantic 请求 / 响应模型
# =========================

class UserRegisterRequest(BaseModel):
    username: str
    password: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


class MessageResponse(BaseModel):
    message: str


# =========================
# 数据库依赖
# =========================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# 注册接口
# =========================

@app.post("/register", response_model=MessageResponse)
def register(req: UserRegisterRequest, db: Session = Depends(get_db)):
    # 判断是否已存在
    user = db.query(User).filter(User.username == req.username).first()
    if user:
        raise HTTPException(status_code=400, detail="用户名已存在")

    # 直接使用 Argon2 加密原始密码
    hashed_password = pwd_context.hash(req.password)

    new_user = User(
        username=req.username,
        password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "注册成功"}


# =========================
# 登录接口
# =========================

@app.post("/login", response_model=MessageResponse)
def login(req: UserLoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()

    if not user:
        raise HTTPException(status_code=400, detail="用户不存在")

    # 校验密码
    if not pwd_context.verify(req.password, user.password):
        raise HTTPException(status_code=400, detail="密码错误")

    return {"message": "登录成功"}


# =========================
# 启动服务
# =========================

if __name__ == "__main__":
    uvicorn.run("api.api_database:app", host="0.0.0.0", port=5001, reload=True)
