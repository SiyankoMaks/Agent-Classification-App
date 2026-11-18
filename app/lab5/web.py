import html
import os
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Security, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator

from .analytics import Analysis
from .intake import MAX_BYTES, Intake
from .output import add_comment, export, graph, table_rows
from .store import Denied, Invalid, Store


app = FastAPI(title='Лабораторная работа № 5 — обеспечивающие подсистемы')
bearer = HTTPBearer(auto_error=False)
Code = Annotated[str, Field(min_length=1, max_length=160)]


class Split(BaseModel):
    seed: int = Field(42, ge=0, le=2**32-1)
    test_size: float = Field(0.25, ge=0.2, le=0.4)


class Territory(Split):
    dataset: int
    age_band: Code
    level: Code
    start: int = Field(ge=1900, le=2100)
    stop: int = Field(ge=1900, le=2100)
    territories: list[Code] = Field(default_factory=list, max_length=250)
    target: Code
    factors: list[Code] = Field(min_length=1, max_length=12)

    @model_validator(mode='after')
    def distinct(self):
        if self.start > self.stop or len(set([self.target, *self.factors])) != len(self.factors)+1:
            raise ValueError('Проверьте границы периода и уникальность показателей')
        return self


class Training(Split):
    label_dataset: int
    method: Literal['logistic', 'forest']


class Profile(BaseModel):
    dataset: int
    pseudonym: str
    model: int
    external: bool = False


class Note(BaseModel):
    body: str


@lru_cache(maxsize=1)
def store():
    return Store(os.environ['LAB5_DATABASE_URL'])


def operation(name, authorization, action):
    token = authorization.credentials if authorization else ''
    active = store()
    actor = active.authorize(token, name)
    try:
        result = action(active, actor)
    except Exception:
        active.audit(actor['id'], name, 'failed')
        raise
    failed = isinstance(result, dict) and result.get('status') in ('rejected', 'failed')
    active.audit(actor['id'], name, 'failed' if failed else 'ok')
    return result


@app.exception_handler(Invalid)
def invalid(request, error):
    return JSONResponse({'detail': str(error)}, status_code=422)


@app.exception_handler(Denied)
def denied(request, error):
    return JSONResponse({'detail': str(error)}, status_code=403)


@app.get('/', response_class=HTMLResponse)
def home():
    return '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Учебная ИС</title>
        <main style="font:18px system-ui;max-width:700px;margin:50px auto"><h1>Учебная ИС анализа данных</h1>
        <p>Импортируйте таблицу, запустите анализ и откройте сохранённый результат через
        <a href="/docs">интерактивный интерфейс API</a>. Для закрытых операций нужен ключ доступа.</p>
        <p>Профили и метки — только синтетические. Результат не является оценкой реального человека.</p>
        <form id="lookup"><label>Ключ <input name="token" type="password" required></label>
        <label>Номер запуска <input name="run" type="number" min="1" required></label>
        <button>Показать</button></form><p id="message" role="status"></p>
        <iframe id="result" title="Результат анализа" sandbox hidden style="width:100%;height:720px;border:1px solid #ddd"></iframe>
        <script>
          const form = document.getElementById('lookup');
          const message = document.getElementById('message');
          const result = document.getElementById('result');
          form.addEventListener('submit', async (event) => {
            event.preventDefault();
            result.hidden = true;
            message.textContent = 'Загрузка…';
            const fields = new FormData(form);
            try {
              const response = await fetch('/runs/' + encodeURIComponent(fields.get('run')) + '/view', {
                headers: { Authorization: 'Bearer ' + fields.get('token') }, cache: 'no-store'
              });
              if (!response.ok) {
                const body = await response.json();
                throw new Error(typeof body.detail === 'string' ? body.detail : 'Проверьте номер запуска');
              }
              result.srcdoc = await response.text();
              result.hidden = false;
              message.textContent = '';
            } catch (error) { message.textContent = error.message; }
          });
        </script></main></html>'''


@app.post('/datasets')
def upload(kind: Literal['IN01', 'IN02', 'IN03'] = Form(), file: UploadFile = File(), source: str = Form(),
           description: str = Form(), updated: str = Form(), synthetic: bool = Form(),
           profile_dataset: int | None = Form(None), authorization: HTTPAuthorizationCredentials = Security(bearer)):
    return operation('import', authorization, lambda active, actor: Intake(active).import_file(
        kind, file.filename or '', file.file.read(MAX_BYTES + 1), source, description, updated, synthetic, profile_dataset))


@app.post('/territory')
def territory(request: Territory, authorization: HTTPAuthorizationCredentials = Security(bearer)):
    return operation('territory.request', authorization, lambda active, actor: Analysis(active).territory(
        actor, request.dataset, request.model_dump(exclude={'dataset'})))


@app.post('/models')
def train(request: Training, authorization: HTTPAuthorizationCredentials = Security(bearer)):
    return operation('train', authorization, lambda active, actor: Analysis(active).train(
        actor, request.label_dataset, request.method, request.seed, request.test_size))


@app.post('/classify')
def classify(request: Profile, authorization: HTTPAuthorizationCredentials = Security(bearer)):
    return operation('classify.request', authorization, lambda active, actor: Analysis(active).classify(
        actor, request.dataset, request.pseudonym, request.model, request.external))


@app.get('/runs')
def history(authorization: HTTPAuthorizationCredentials = Security(bearer)):
    return operation('history', authorization, lambda active, actor: active.history(actor))


@app.get('/runs/{run_id}')
def run(run_id: int, authorization: HTTPAuthorizationCredentials = Security(bearer)):
    return operation('read', authorization, lambda active, actor: active.read_run(actor, run_id))


@app.get('/runs/{run_id}/view', response_class=HTMLResponse)
def view(run_id: int, authorization: HTTPAuthorizationCredentials = Security(bearer)):
    snapshot = operation('view', authorization, lambda active, actor: active.read_run(actor, run_id))
    rows = [row for row in table_rows(snapshot) if row[0] != 'прогноз']
    rows = ['<tr>' + ''.join(f'<td>{html.escape(display(value))}</td>' for value in row) + '</tr>' for row in rows]
    notes = [entry['body'] for entry in snapshot['explanations'] if entry['body']]
    notes.append(snapshot['result']['limitation'])
    notes += [f'{entry["author"]}: {entry["body"]}' for entry in snapshot['comments']]
    return HTMLResponse('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Результат анализа</title>'
            '<style>main{font:16px system-ui;max-width:880px;margin:30px auto}svg{width:480px;max-width:100%}'
            'td{padding:4px 8px;border-bottom:1px solid #ddd}</style><main>'
            f'<h1>Запуск № {run_id}</h1>{graph(snapshot)}<table>{"".join(rows)}</table>'
            + ''.join(f'<p>{html.escape(note)}</p>' for note in notes) + '</main></html>',
            headers={'Cache-Control': 'no-store'})


def display(value):
    if value is None:
        return '—'
    return f'{value:.6g}' if isinstance(value, float) else str(value)


@app.post('/runs/{run_id}/comments')
def comment(run_id: int, note: Note, authorization: HTTPAuthorizationCredentials = Security(bearer)):
    return operation('comment', authorization, lambda active, actor: {'id': add_comment(active, actor, run_id, note.body)})


@app.get('/runs/{run_id}/export/{format_name}')
def download(run_id: int, format_name: Literal['csv', 'xlsx', 'pdf'], authorization: HTTPAuthorizationCredentials = Security(bearer)):
    body, media_type = operation('export', authorization, lambda active, actor: export(active.read_run(actor, run_id), format_name))
    return Response(body, media_type=media_type, headers={'Content-Disposition': f'attachment; filename=run-{run_id}.{format_name}'})
