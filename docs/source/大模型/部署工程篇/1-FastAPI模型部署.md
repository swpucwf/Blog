# FastAPI模型部署

## 1. 为什么选 FastAPI

把模型封装成 HTTP API 是部署的常见做法。FastAPI 基于 Starlette 与 Pydantic：Starlette 带来足以媲美 NodeJS/Go 框架的高性能；Pydantic 借 Python 类型提示自动完成数据校验与转换，省掉大量参数验证代码。它还有「杀手级」特性——根据代码自动生成交互式 API 文档（OpenAPI + Swagger UI），可直接在页面上调试；并充分利用 `async/await`。安装一条命令：

```bash
pip install "fastapi[all]"    # 含 uvicorn（高性能 ASGI 服务器）
```

> **ASGI** 是现代 Python Web 框架与服务器通信的标准接口；`uvicorn` 就是在生产环境运行 FastAPI 的 ASGI 服务器。

## 2. 主要用法

### 2.1 最小应用

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}
```

`app = FastAPI()` 是应用主入口；`@app.get("/")` 把下面的异步函数绑定到根路径的 GET 请求，返回值自动转成 JSON。启动：

```bash
uvicorn 01_test:app --reload
# INFO: Uvicorn running on http://127.0.0.1:8000
```

`01_test` 是文件名、`app` 是 FastAPI 实例名、`--reload` 让代码变更后自动重启（开发用）。

### 2.2 自动交互式文档

访问 `http://127.0.0.1:8000/docs`，得到 Swagger UI 生成的交互式文档：可查看所有端点、参数与返回结构，点「Try it out」→「Execute」立即调用：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_1_1.png" width="80%" alt="FastAPI 自动生成的交互式 API 文档" />
  <p>图 1：FastAPI 自动生成的交互式 API 文档</p>
</div>

返回结果里甚至带可直接使用的 `curl` 命令、请求 URL、响应内容与响应头——所见即所得，调试效率大幅提升。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_1_2.png" width="80%" alt="在 FastAPI 文档中执行 API" />
  <p>图 2：在文档页执行 API 并查看响应</p>
</div>

另有 ReDoc 风格文档 `http://127.0.0.1:8000/redoc`，视图更紧凑、适合通读整体结构；顶部「Download」可直接下载 `openapi.json`——这份规范文件可用于生成多语言客户端代码（TypeScript/Java 等），也可导入 Postman 一键建测试集合。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_1_3.png" width="80%" alt="由 ReDoc 生成的 API 文档" />
  <p>图 3：ReDoc 生成的 API 文档</p>
</div>

两套文档都源于 FastAPI 严格遵循 **OpenAPI 规范**，按代码自动生成 `openapi.json`，Swagger UI 与 ReDoc 只是它的可视化前端。

### 2.3 路由与参数：请求体、查询参数、路径参数

**请求体**用 Pydantic 模型定义结构：

```python
from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str
    # a: int
    # c: bool | None = None   # 可选字段

@app.post("/predict/")
def predict(request: PredictRequest):
    return {"input_text": request.text, "prediction": "some_result"}
```

把参数类型声明为 `PredictRequest` 后，FastAPI 自动做四件事：读请求体 JSON → 按模型校验（字段是否存在、是否字符串）→ 通过则转成对象赋值 → 失败自动返回 **HTTP 422** 并附详细错误。**代码即文档且自带校验**：不再需要 `if 'text' not in data`、`isinstance(...)` 这类防御代码。

**查询参数**（`?key=value`）直接在函数签名里声明，可给默认值变可选：

```python
@app.get("/search/")
async def search(query: str):            # /search/?query=你好
    return {"query": query}

@app.get("/users/")
async def read_users(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

**路径参数**用与格式化字符串相同的语法，自动按类型校验：

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):       # /items/123，item_id 自动为 int
    return {"item_id": item_id}
```

三类参数可同时使用，FastAPI 自动区分。

### 2.4 模型部署 API 模板

把上述要点整合成一个工程化模板：日志配置、请求体模型、预测路由（业务校验 + `HTTPException` + 全局异常捕获 + 结构化 JSON 返回）、根路径与 `/health` 健康检查（含可选 `verbose` 查询参数）。

```python
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI 模型API", description="演示如何用 FastAPI 部署模型", version="1.0.0")

class PredictRequest(BaseModel):
    text: str
    model_name: str | None = None

@app.post("/predict/")
async def predict(request: PredictRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="输入文本不能为空")
        prediction = f"模型对'{text}'的预测结果"     # 实际此处调用 model.predict
        return {"code": 0, "message": "成功",
                "data": {"input_text": text, "prediction": prediction}}
    except Exception as e:
        logger.error(f"服务器内部错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

@app.get("/health")
async def health_check(verbose: bool = False):
    if verbose:
        return {"status": "ok", "details": "All systems operational."}
    return {"status": "ok"}
```

要点：`title`/`description`/`version` 显示在自动文档里；业务校验失败用 `HTTPException` 中断并返回标准 HTTP 错误（比直接 `return` 错误信息规范）；`try...except` 兜住未预料异常，防止服务崩溃并记录详细日志；`/health` 是监控服务的工程惯例。

## 3. 部署 NER 模型

### 3.1 项目结构

新建 `ner_deployment`，从 C8 项目复制必要文件：

```text
ner_deployment/
├── checkpoints/        # best_model.pth + config.json
├── data/               # vocabulary.json + categories.json
├── src/                # models/ tokenizer/ utils/（仅保留推理必需模块）
├── main.py             # FastAPI 应用
└── predict.py          # NerPredictor（由 C8/06_predict.py 复制重命名）
```

### 3.2 服务代码：模型只加载一次

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import NerPredictor

MODEL_DIR = "./checkpoints"
app = FastAPI(title="命名实体识别 API", version="1.0.0")

class NerRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    app.state.predictor = NerPredictor(model_dir=MODEL_DIR)   # 启动时加载一次

@app.post("/predict/ner")
async def predict_ner(request: NerRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="输入文本不能为空")
        entities = app.state.predictor.predict(text)
        return {"code": 0, "message": "成功", "data": {"text": text, "entities": entities}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")
```

两个关键点：

- **`@app.on_event("startup")`**：应用启动时执行且仅执行一次，正适合加载模型权重这类耗时操作，避免每次请求重复加载；
- **`app.state.predictor`**：FastAPI 的应用级共享状态容器，模型实例存进去后，所有路由函数都能取用，实现高效复用。

> 新版 FastAPI 更推荐 `lifespan` 上下文管理器替代 `on_event`，语义相同、资源释放更清晰。

### 3.3 启动与测试

```bash
uvicorn main:app --reload
curl -X POST "http://127.0.0.1:8000/predict/ner" -H "Content-Type: application/json" \
  -d "{\"text\":\"患者自述发热、咳嗽，伴有轻微头痛。\"}"
```

返回结构化实体列表（`发热`→sym、`咳嗽`→sym、`头`→bod…），一个健壮、高效、自带交互文档的 API 服务就封装完成。

> **Windows 提示**：PowerShell 中 `curl` 是 `Invoke-WebRequest` 的别名，参数格式不同会报错；建议在 `cmd` 或 Git Bash 中执行。

> **补记**：这一篇的核心不是 FastAPI 语法，而是三个部署习惯——**启动时加载模型**（别放请求里）、**应用级共享状态**（`app.state`）、**标准错误码与健康检查**（400/422/500 + `/health`）。这三点决定了服务是「能跑」还是「能上生产」。

## 写在最后

本地跑通只是第一步，服务要真正对外可用，还得放到云服务器上并保证持久运行——下一篇[云服务器部署与Docker Compose](2-云服务器部署与Docker)从 ECS 安全组、uv 依赖管理讲到 systemd 与容器化两套部署方案。
