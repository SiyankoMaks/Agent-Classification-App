import hashlib
import json
import os
import secrets
from pathlib import Path

from sqlalchemy import create_engine, text


class Invalid(ValueError):
    pass


class Denied(PermissionError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def document(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def execute(connection, sql, **values):
    return connection.execute(text(sql), values)


class Store:
    def __init__(self, url=None):
        self.engine = create_engine(url or os.environ['LAB5_DATABASE_URL'], pool_pre_ping=True)

    def migrate(self):
        schema = Path(__file__).with_name('001_schema.sql').read_text(encoding='utf-8')
        with self.engine.begin() as connection:
            connection.exec_driver_sql('SELECT pg_advisory_xact_lock(5200501)')
            if connection.exec_driver_sql("SELECT to_regclass('lab_actor')").scalar() is None:
                connection.exec_driver_sql(schema.replace('%', '%%'))

    def provision(self, name, role):
        token = secrets.token_urlsafe(32)
        with self.engine.begin() as connection:
            execute(connection, 'INSERT INTO lab_actor (name,role,token_hash) VALUES (:name,:role,:token)',
                    name=name, role=role, token=digest(token.encode()))
        return token

    def audit(self, actor, operation, status, run=None):
        with self.engine.begin() as connection:
            execute(connection, 'INSERT INTO lab_audit (actor_id,run_id,operation,status) VALUES (:actor,:run,:op,:status)',
                    actor=actor, run=run, op=operation, status=status)

    def authorize(self, token, operation, admin=False):
        with self.engine.connect() as connection:
            actor = execute(connection, 'SELECT id,role FROM lab_actor WHERE token_hash=:token',
                            token=digest(token.encode())).mappings().one_or_none() if token else None
        allowed = actor is not None and (not admin or actor['role'] == 'admin')
        self.audit(actor['id'] if actor else None, operation, 'allowed' if allowed else 'denied')
        if not allowed:
            raise Denied('Операция не разрешена')
        return actor

    def history(self, actor):
        with self.engine.connect() as connection:
            return [dict(row) for row in execute(connection, '''
                SELECT id,kind,status,created_at,finished_at FROM lab_run
                WHERE actor_id=:actor OR :admin ORDER BY id DESC LIMIT 100
                ''', actor=actor['id'], admin=actor['role'] == 'admin').mappings()]

    def read_run(self, actor, run_id):
        with self.engine.connect() as connection:
            result = execute(connection, '''
                SELECT r.*, ds.id AS dataset_version, ds.source_updated, src.code AS source,
                       u.code AS training_source, training.id AS training_profiles, m.label_dataset_id, ls.code AS label_source
                FROM lab_run r LEFT JOIN lab_profile p ON p.id=r.profile_id
                JOIN lab_dataset ds ON ds.id=coalesce(r.dataset_id,p.dataset_id)
                JOIN lab_source src ON src.id=ds.source_id
                LEFT JOIN lab_model m ON m.id=r.model_id
                LEFT JOIN lab_dataset labels ON labels.id=m.label_dataset_id
                LEFT JOIN lab_source ls ON ls.id=labels.source_id
                LEFT JOIN lab_dataset training ON training.id=labels.profile_dataset_id
                LEFT JOIN lab_source u ON u.id=training.source_id
                WHERE r.id=:run AND (r.actor_id=:actor OR :admin)
                ''', run=run_id, actor=actor['id'], admin=actor['role'] == 'admin').mappings().one_or_none()
            if result is None:
                raise Denied('Запуск недоступен')
            snapshot = dict(result)
            snapshot['explanations'] = [dict(row) for row in execute(connection, '''
                SELECT kind,status,body FROM lab_explanation WHERE run_id=:run ORDER BY kind
                ''', run=run_id).mappings()]
            snapshot['comments'] = [dict(row) for row in execute(connection, '''
                SELECT a.name AS author,c.body,c.created_at FROM lab_comment c
                JOIN lab_actor a ON a.id=c.actor_id WHERE c.run_id=:run ORDER BY c.id
                ''', run=run_id).mappings()]
        return snapshot
