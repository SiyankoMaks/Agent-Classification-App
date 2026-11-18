import io
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

from .intake import CLASSES
from .store import Invalid, digest, document, execute


def as_frame(rows):
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    for column in NUMERIC_FEATURES:
        frame[column] = frame[column].astype(float)
    return frame


def score_regression(expected, predicted):
    return {'r2': float(r2_score(expected, predicted)), 'mae': float(mean_absolute_error(expected, predicted)),
            'rmse': float(root_mean_squared_error(expected, predicted))}


def score_classification(expected, predicted):
    return {'accuracy': float(accuracy_score(expected, predicted)),
            'macro_f1': float(f1_score(expected, predicted, labels=CLASSES, average='macro', zero_division=0)),
            'confusion': confusion_matrix(expected, predicted, labels=CLASSES).tolist()}


def territory_result(rows, parameters):
    selected = [parameters['target'], *parameters['factors']]
    frame = pd.DataFrame(rows, columns=['code', 'year', 'indicator', 'unit', 'value'])
    if frame.empty or not set(selected) <= set(frame['indicator']):
        raise Invalid('Нет совместимой выборки для всех выбранных показателей')
    units = dict(zip(frame['indicator'], frame['unit']))
    frame['value'] = frame['value'].map(lambda value: float(value) if value is not None else np.nan)
    wide = frame.pivot(index=['code', 'year'], columns='indicator', values='value')[selected].sort_index()
    complete = wide.dropna()
    if len(complete) < 10:
        raise Invalid('Нужно не менее 10 полных пар территория–год')
    stats = {column: {'n': len(complete), 'mean': float(complete[column].mean()),
                      'std': float(complete[column].std(ddof=1)), 'median': float(complete[column].median()),
                      'min': float(complete[column].min()), 'max': float(complete[column].max()), 'unit': units[column]}
             for column in selected}
    corr = complete.corr()
    correlation = [{'left': left, 'right': right, 'r': None if pd.isna(corr.loc[left, right]) else float(corr.loc[left, right])}
                   for left in selected for right in selected]
    factors, target = parameters['factors'], parameters['target']
    train, test = train_test_split(complete, test_size=parameters['test_size'], random_state=parameters['seed'])
    if len(test) < 2 or test[target].nunique() < 2 or train[target].nunique() < 2:
        raise Invalid('Цель должна варьировать в обеих частях выборки; в контрольной нужно не менее двух строк')
    candidates = {'baseline': DummyRegressor(strategy='mean'),
                  'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                  'forest': RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=parameters['seed'])}
    metrics, predictions = {}, []
    for name, model in candidates.items():
        model.fit(train[factors], train[target])
        predicted = model.predict(test[factors])
        metrics[name] = score_regression(test[target], predicted)
        for (code, year), actual, estimate in zip(test.index, test[target], predicted, strict=True):
            predictions.append({'territory': code, 'year': int(year), 'method': name,
                                'actual': float(actual), 'estimate': float(estimate), 'unit': units[target]})
    ridge = candidates['ridge']
    scaler, model = ridge.named_steps['standardscaler'], ridge.named_steps['ridge']
    coefficients = model.coef_ / scaler.scale_
    return {'n': len(complete), 'excluded': len(wide) - len(complete), 'train_n': len(train), 'test_n': len(test),
            'stats': stats, 'correlation': correlation, 'metrics': metrics, 'predictions': predictions,
            'ridge_coefficients': dict(zip(factors, coefficients.tolist())),
            'ridge_intercept': float(model.intercept_ - coefficients @ scaler.mean_),
            'limitation': 'Сравнение на фиксированной контрольной части; статистическая связь не доказывает причинность.'}


def classifier_bundle(rows, method, seed, test_size):
    if not rows:
        raise Invalid('Нет размеченных профилей в версии IN03')
    labels = pd.Series([row['class_code'] for row in rows])
    if set(labels) != set(CLASSES) or labels.value_counts().min() < 2:
        raise Invalid('Нужно не менее двух примеров каждого из трёх классов')
    frame = as_frame([row['features'] for row in rows])
    try:
        train_x, test_x, train_y, test_y = train_test_split(frame, labels, stratify=labels, test_size=test_size,
                                                          random_state=seed)
    except ValueError as error:
        raise Invalid('Выбранная доля не позволяет разделить выборку по классам') from error
    if set(train_y) != set(CLASSES) or set(test_y) != set(CLASSES):
        raise Invalid('Все три класса требуются в обучающей и контрольной частях')
    prep = ColumnTransformer([('category', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
                              ('number', StandardScaler(), NUMERIC_FEATURES)])
    if method == 'logistic':
        estimator = LogisticRegression(max_iter=1000, random_state=seed)
    elif method == 'forest':
        estimator = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise Invalid('Неизвестный метод классификации')
    pipeline = make_pipeline(prep, estimator).fit(train_x, train_y)
    baseline = DummyClassifier(strategy='most_frequent').fit(train_x, train_y)
    weights = (np.mean(np.abs(estimator.coef_), axis=0) if method == 'logistic' else estimator.feature_importances_)
    totals = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    encoder = prep.named_transformers_['category']
    grouped = [feature for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True)
               for _ in categories] + NUMERIC_FEATURES
    for feature, weight in zip(grouped, weights, strict=True):
        totals[feature] += float(weight)
    top = sorted(totals, key=totals.get, reverse=True)[:3]
    return {'pipeline': pipeline, 'top': top}, {
        'model': score_classification(test_y, pipeline.predict(test_x)),
        'baseline': score_classification(test_y, baseline.predict(test_x)),
        'classes': list(CLASSES), 'train_n': len(train_x), 'test_n': len(test_x)}


class Analysis:
    def __init__(self, store):
        self.store = store

    def territory(self, actor, dataset, parameters):
        with self.store.engine.begin() as connection:
            if not execute(connection, "SELECT id FROM lab_dataset WHERE id=:id AND kind='IN01' AND status='ready'",
                           id=dataset).scalar():
                raise Invalid('Нет допустимой версии IN01')
        def calculate():
            with self.store.engine.begin() as connection:
                connection.exec_driver_sql('SELECT pg_advisory_xact_lock(5200503)')
                rows = execute(connection, '''
                    SELECT t.code,o.year,o.indicator,i.unit,o.value FROM lab_observation o
                    JOIN lab_territory t ON t.id=o.territory_id JOIN lab_indicator i ON i.code=o.indicator
                    WHERE o.dataset_id=:dataset AND t.level=:level AND o.age_band=:band
                      AND o.year BETWEEN :start AND :stop AND o.indicator=ANY(:selected)
                      AND (cardinality(CAST(:territories AS text[]))=0 OR t.code=ANY(CAST(:territories AS text[])))
                    ''', dataset=dataset, level=parameters['level'], band=parameters['age_band'],
                    start=parameters['start'], stop=parameters['stop'], selected=[parameters['target'], *parameters['factors']],
                    territories=parameters['territories']).all()
                return territory_result(rows, parameters), []
        return self.run(actor, 'territory', dataset, None, None, parameters, calculate)

    def train(self, actor, label_dataset, method, seed, test_size):
        with self.store.engine.begin() as connection:
            connection.exec_driver_sql('SELECT pg_advisory_xact_lock(5200503)')
            rows = execute(connection, '''
                SELECT p.features,l.class_code FROM lab_label l JOIN lab_profile p ON p.id=l.profile_id
                JOIN lab_dataset d ON d.id=l.dataset_id AND d.profile_dataset_id=p.dataset_id
                WHERE d.id=:id AND d.kind='IN03' AND d.status='ready' ORDER BY p.id
                ''', id=label_dataset).mappings().all()
            bundle, metrics = classifier_bundle(rows, method, seed, test_size)
            buffer = io.BytesIO()
            joblib.dump(bundle, buffer)
            artifact = buffer.getvalue()
            version = execute(connection, '''INSERT INTO lab_model
                (label_dataset_id,actor_id,method,feature_schema,parameters,metrics,artifact,digest)
                VALUES (:dataset,:actor,:method,CAST(:schema AS jsonb),CAST(:params AS jsonb),CAST(:metrics AS jsonb),:artifact,:digest)
                RETURNING id''', dataset=label_dataset, actor=actor['id'], method=method, schema=document(FEATURE_COLUMNS),
                params=document({'seed': seed, 'test_size': test_size}), metrics=document(metrics), artifact=artifact,
                digest=digest(artifact)).scalar_one()
        return {'version': version, 'metrics': metrics}

    def classify(self, actor, dataset, pseudonym, model_id, external=False):
        with self.store.engine.connect() as connection:
            profile = execute(connection, '''
                SELECT p.id,p.features FROM lab_profile p JOIN lab_dataset d ON d.id=p.dataset_id
                WHERE p.dataset_id=:dataset AND p.pseudonym=:name AND d.status='ready' AND d.kind='IN02'
                ''', dataset=dataset, name=pseudonym).mappings().one_or_none()
            model = execute(connection, 'SELECT * FROM lab_model WHERE id=:id', id=model_id).mappings().one_or_none()
        if profile is None or model is None:
            raise Invalid('Версия модели или профиль не найдены')
        def calculate():
            features = profile['features']
            if model['feature_schema'] != FEATURE_COLUMNS or set(features) != set(FEATURE_COLUMNS):
                raise Invalid('Схема профиля не соответствует версии модели')
            artifact = model['artifact']
            if digest(artifact) != model['digest']:
                raise Invalid('Контрольная сумма модели не совпадает')
            bundle = joblib.load(io.BytesIO(artifact))
            pipeline = bundle['pipeline']
            probabilities = dict(zip(pipeline.classes_.tolist(), pipeline.predict_proba(as_frame([features]))[0].tolist()))
            chosen = max(CLASSES, key=probabilities.__getitem__)
            result = {'class': chosen, 'probabilities': probabilities, 'method': model['method'], 'top': bundle['top'],
                      'limitation': 'Учебная оценка синтетических данных; не заключение о человеке.'}
            local = (f'Модель выбрала класс «{chosen}» с оценкой {probabilities[chosen]:.2%}. '
                     f'Общая важность признаков модели: {", ".join(bundle["top"])}. '
                     'Это не объясняет индивидуальную причинность. ' + result['limitation'])
            return result, [('local', 'ready', local), ('llm', *external_explanation(features, result, external))]
        return self.run(actor, 'classify', None, profile['id'], model_id, {'external': external}, calculate)

    def run(self, actor, kind, dataset, profile, model, parameters, calculate):
        with self.store.engine.begin() as connection:
            run_id = execute(connection, '''INSERT INTO lab_run (actor_id,dataset_id,profile_id,model_id,kind,parameters)
                VALUES (:actor,:dataset,:profile,:model,:kind,CAST(:parameters AS jsonb)) RETURNING id''',
                actor=actor['id'], dataset=dataset, profile=profile, model=model, kind=kind,
                parameters=document(parameters)).scalar_one()
        try:
            result, explanations = calculate()
            with self.store.engine.begin() as connection:
                for source, status, body in explanations:
                    execute(connection, '''INSERT INTO lab_explanation (run_id,kind,status,body)
                        VALUES (:run,:kind,:status,:body)''', run=run_id, kind=source, status=status, body=body)
                execute(connection, '''UPDATE lab_run SET status='completed',result=CAST(:result AS jsonb),
                    finished_at=now() WHERE id=:run''', result=document(result), run=run_id)
        except Exception as error:
            with self.store.engine.begin() as connection:
                execute(connection, "UPDATE lab_run SET status='failed',error=:error,finished_at=now() WHERE id=:run",
                        run=run_id, error=str(error) if isinstance(error, Invalid) else 'Внутренняя ошибка расчёта')
            self.store.audit(actor['id'], kind, 'failed', run_id)
            if not isinstance(error, Invalid):
                raise
        else:
            self.store.audit(actor['id'], kind, 'ok', run_id)
        return self.store.read_run(actor, run_id)


def external_explanation(features, result, enabled):
    if not enabled:
        return 'disabled', None
    if not os.environ.get('POLZA_API_KEY') or not os.environ.get('POLZA_MODEL'):
        return 'unavailable', None
    from openai import OpenAI
    try:
        with OpenAI(api_key=os.environ['POLZA_API_KEY'], base_url=os.getenv('POLZA_BASE_URL', 'https://polza.ai/api/v1'),
                    timeout=5, max_retries=0) as client:
            response = client.chat.completions.create(model=os.environ['POLZA_MODEL'], max_tokens=500, messages=[
                {'role': 'system', 'content': 'Учебные синтетические данные. Ответь только JSON: class — один из классов '
                 'Низкий, Средний, Высокий; confidence — число 0–100; explanation — краткий русский текст. '
                 'Не делай выводов о реальных людях или причинности.'},
                {'role': 'user', 'content': document({'features': features, 'model': result})}])
        payload = json.loads(response.choices[0].message.content)
        verdict, confidence, explanation = payload['class'], payload['confidence'], payload['explanation']
        if (verdict not in CLASSES or type(confidence) not in (int, float) or not 0 <= confidence <= 100
                or not isinstance(explanation, str) or not 1 <= len(explanation.strip()) <= 1500):
            raise Invalid('Неверный формат пояснения')
        comparison = 'совпадает' if verdict == result['class'] else 'не совпадает'
        return 'ready', f'LLM: {verdict} ({confidence} %); {comparison} с классом модели. {explanation}'
    except Exception:
        return 'unavailable', None
