CREATE TABLE lab_actor (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE,
    role text NOT NULL CHECK (role IN ('analyst', 'admin')),
    token_hash char(64) NOT NULL UNIQUE
);
CREATE TABLE lab_source (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL UNIQUE,
    description text NOT NULL,
    synthetic boolean NOT NULL
);
CREATE TABLE lab_dataset (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id bigint NOT NULL REFERENCES lab_source (id),
    profile_dataset_id bigint REFERENCES lab_dataset (id),
    kind text NOT NULL CHECK (kind IN ('IN01', 'IN02', 'IN03')),
    filename text NOT NULL,
    file_format text NOT NULL CHECK (file_format IN ('.csv', '.xlsx')),
    digest char(64) NOT NULL,
    original bytea NOT NULL CHECK (octet_length(original) BETWEEN 1 AND 10485760),
    source_updated date NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now(),
    status text NOT NULL DEFAULT 'registered' CHECK (status IN ('registered', 'rejected', 'ready')),
    protocol jsonb NOT NULL DEFAULT '{}'::jsonb,
    CHECK ((kind = 'IN03') = (profile_dataset_id IS NOT NULL)),
    UNIQUE NULLS NOT DISTINCT (source_id, kind, file_format, digest, source_updated, profile_dataset_id)
);
CREATE TABLE lab_territory (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL,
    level text NOT NULL,
    UNIQUE (code, level)
);
CREATE TABLE lab_age_band (
    code text PRIMARY KEY
);
CREATE TABLE lab_unit (
    code text PRIMARY KEY
);
CREATE TABLE lab_indicator (
    code text PRIMARY KEY,
    unit text NOT NULL REFERENCES lab_unit (code)
);
CREATE TABLE lab_observation (
    dataset_id bigint NOT NULL REFERENCES lab_dataset (id),
    territory_id bigint NOT NULL REFERENCES lab_territory (id),
    age_band text NOT NULL REFERENCES lab_age_band (code),
    indicator text NOT NULL REFERENCES lab_indicator (code),
    year smallint NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    value numeric,
    PRIMARY KEY (dataset_id, territory_id, age_band, indicator, year)
);
CREATE TABLE lab_profile (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id bigint NOT NULL REFERENCES lab_dataset (id),
    pseudonym text NOT NULL CHECK (pseudonym ~ '^[A-Za-z0-9_-]{1,40}$'),
    features jsonb NOT NULL CHECK (jsonb_typeof(features) = 'object'),
    UNIQUE (dataset_id, pseudonym)
);
CREATE TABLE lab_risk_class (
    code text PRIMARY KEY
);
CREATE TABLE lab_label (
    dataset_id bigint NOT NULL REFERENCES lab_dataset (id),
    profile_id bigint NOT NULL REFERENCES lab_profile (id),
    class_code text NOT NULL REFERENCES lab_risk_class (code),
    PRIMARY KEY (dataset_id, profile_id)
);
CREATE TABLE lab_model (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    label_dataset_id bigint NOT NULL REFERENCES lab_dataset (id),
    actor_id bigint NOT NULL REFERENCES lab_actor (id),
    method text NOT NULL CHECK (method IN ('logistic', 'forest')),
    feature_schema jsonb NOT NULL,
    parameters jsonb NOT NULL,
    metrics jsonb NOT NULL,
    artifact bytea NOT NULL,
    digest char(64) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE lab_run (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    actor_id bigint NOT NULL REFERENCES lab_actor (id),
    dataset_id bigint REFERENCES lab_dataset (id),
    profile_id bigint REFERENCES lab_profile (id),
    model_id bigint REFERENCES lab_model (id),
    kind text NOT NULL CHECK (kind IN ('territory', 'classify')),
    parameters jsonb NOT NULL,
    status text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'failed', 'completed')),
    result jsonb,
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    CHECK ((kind = 'territory' AND dataset_id IS NOT NULL AND profile_id IS NULL AND model_id IS NULL)
        OR (kind = 'classify' AND dataset_id IS NULL AND profile_id IS NOT NULL AND model_id IS NOT NULL)),
    CHECK ((status = 'completed') = (result IS NOT NULL)),
    CHECK ((status = 'failed') = (error IS NOT NULL)),
    CHECK ((status = 'running') = (finished_at IS NULL))
);
CREATE TABLE lab_explanation (
    run_id bigint NOT NULL REFERENCES lab_run (id),
    kind text NOT NULL CHECK (kind IN ('local', 'llm')),
    status text NOT NULL CHECK (status IN ('ready', 'disabled', 'unavailable')),
    body text,
    PRIMARY KEY (run_id, kind),
    CHECK ((status = 'ready') = (body IS NOT NULL))
);
CREATE TABLE lab_comment (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES lab_run (id),
    actor_id bigint NOT NULL REFERENCES lab_actor (id),
    body text NOT NULL CHECK (char_length(body) BETWEEN 1 AND 4000),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE lab_audit (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    actor_id bigint REFERENCES lab_actor (id),
    run_id bigint REFERENCES lab_run (id),
    operation text NOT NULL,
    status text NOT NULL CHECK (status IN ('allowed', 'denied', 'ok', 'failed')),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX lab_run_history ON lab_run (actor_id, id DESC);
CREATE INDEX lab_comment_run ON lab_comment (run_id, id);
CREATE INDEX lab_audit_run ON lab_audit (run_id, id);
INSERT INTO lab_risk_class (code) VALUES ('Низкий'), ('Средний'), ('Высокий');
