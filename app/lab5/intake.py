import csv
import io
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

from app.ml.features import FEATURE_COLUMNS, FORM_FIELDS

from .store import Invalid, digest, document, execute


CLASSES = ('Низкий', 'Средний', 'Высокий')
MAX_BYTES = 10 * 1024 * 1024
HEADERS = {
    'IN01': ('territory', 'level', 'year', 'age_band', 'indicator', 'unit', 'value'),
    'IN02': ('pseudonym', *FEATURE_COLUMNS),
    'IN03': ('pseudonym', 'class_code'),
}


class FieldError(Invalid):
    def __init__(self, field, reason):
        super().__init__(reason)
        self.field = field


def required(row, name):
    value = row[name]
    if value is None or not str(value).strip() or len(str(value).strip()) > 160:
        raise FieldError(name, 'Обязательное значение длиной до 160 символов')
    return str(value).strip()


def numeric(value, field, minimum=None, maximum=None, integer=False):
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation:
        raise FieldError(field, 'Требуется число') from None
    if (not parsed.is_finite() or (minimum is not None and parsed < minimum)
            or (maximum is not None and parsed > maximum) or abs(parsed.adjusted()) > 100):
        raise FieldError(field, 'Число вне допустимого диапазона')
    if integer and parsed != parsed.to_integral_value():
        raise FieldError(field, 'Требуется целое число')
    return parsed


def profile_features(row):
    features = {}
    for field in FORM_FIELDS:
        name = field['name']
        value = required(row, name)
        if field['type'] == 'select':
            if value not in field['options']:
                raise FieldError(name, 'Значение отсутствует в списке допустимых')
            features[name] = int(value) if set(field['options']) == {'0', '1'} else value
        else:
            low, high = (12, 17) if name == 'Возраст' else (Decimal(field['min']), Decimal(field.get('max', '1e100')))
            integer = field['step'] == '1' and not name.endswith(', %')
            parsed = numeric(value, name, low, high, integer)
            features[name] = int(parsed) if integer else str(parsed)
    return features


def read_rows(raw, suffix):
    if suffix == '.csv':
        yield from csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
    elif suffix == '.xlsx':
        with ZipFile(io.BytesIO(raw)) as archive:
            if sum(member.file_size for member in archive.infolist()) > 64 * 1024 * 1024:
                raise Invalid('Содержимое XLSX превышает допустимый размер')
        workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=False, keep_links=False)
        try:
            if len(workbook.worksheets) != 1 or workbook.active.max_column is None or workbook.active.max_column > 64:
                raise Invalid('Требуется один лист и не более 64 столбцов')
            for cells in workbook.active.iter_rows():
                if any(cell.data_type == 'f' for cell in cells):
                    raise Invalid('Формулы в импортируемых ячейках не допускаются')
                yield [cell.value for cell in cells]
        finally:
            workbook.close()
    else:
        raise Invalid('Поддерживаются CSV UTF-8 и XLSX')


def check_rows(raw, suffix, kind, known_units, known_profiles):
    rows, errors = [], []
    keys = set()
    units = dict(known_units)
    try:
        with_rows = read_rows(raw, suffix)
        headers = tuple(str(cell).strip() for cell in next(with_rows, ()))
        if len(headers) != len(set(headers)) or set(headers) != set(HEADERS[kind]):
            raise Invalid('Заголовки не совпадают со схемой ' + kind)
        for line, values in enumerate(with_rows, 2):
            if line > 10_001:
                raise Invalid('Допускается не более 10 000 строк данных')
            try:
                if len(values) != len(headers):
                    raise FieldError('*', 'Неверное количество столбцов')
                row = dict(zip(headers, values, strict=True))
                if kind == 'IN01':
                    record = {key: required(row, key) for key in ('territory', 'level', 'age_band', 'indicator', 'unit')}
                    record['year'] = int(numeric(row['year'], 'year', 1900, 2100, True))
                    unit = record['unit']
                    upper = 100 if unit == '%' else None
                    record['value'] = (None if row['value'] is None or str(row['value']).strip() == ''
                                       else numeric(row['value'], 'value', 0 if upper else None, upper))
                    if units.setdefault(record['indicator'], unit) != unit:
                        raise FieldError('unit', 'У показателя уже задана другая единица')
                    key = tuple(record[name] for name in ('territory', 'level', 'year', 'age_band', 'indicator'))
                else:
                    key = required(row, 'pseudonym')
                    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', key):
                        raise FieldError('pseudonym', 'Недопустимый псевдоним')
                    record = {'pseudonym': key}
                    if kind == 'IN02':
                        record['features'] = profile_features(row)
                    else:
                        class_code = required(row, 'class_code')
                        if class_code not in CLASSES:
                            raise FieldError('class_code', 'Неизвестный класс')
                        if key not in known_profiles:
                            raise FieldError('pseudonym', 'Профиль отсутствует в указанной версии IN02')
                        record.update(profile=known_profiles[key], class_code=class_code)
                if key in keys:
                    raise FieldError('*', 'Повтор ключа в наборе')
                keys.add(key)
                rows.append(record)
            except FieldError as error:
                errors.append({'row': line, 'field': error.field, 'reason': str(error)})
                if len(errors) >= 200:
                    break
        if not rows and not errors:
            raise Invalid('Пустой набор')
    except (Invalid, OSError, ValueError, csv.Error, KeyError, RuntimeError) as error:
        errors.append({'row': None, 'field': '*', 'reason': str(error) if isinstance(error, Invalid) else 'Некорректный файл'})
    except Exception as error:
        if suffix != '.xlsx':
            raise
        errors.append({'row': None, 'field': '*', 'reason': 'Некорректный XLSX'})
    return rows, errors


class Intake:
    def __init__(self, store):
        self.store = store

    def import_file(self, kind, filename, raw, source, description, updated, synthetic, profile_dataset=None):
        if kind not in HEADERS or not 0 < len(raw) <= MAX_BYTES:
            raise Invalid('Недопустимый тип или размер файла')
        suffix = Path(filename).suffix.lower()
        if suffix not in {'.csv', '.xlsx'}:
            raise Invalid('Поддерживаются CSV UTF-8 и XLSX')
        if kind != 'IN01' and not synthetic:
            raise Invalid('Профили и метки допускаются только как синтетические данные')
        if (kind == 'IN03') != (profile_dataset is not None):
            raise Invalid('Для IN03 обязательна версия IN02; для остальных типов она не задаётся')
        if not source.strip() or not description.strip():
            raise Invalid('Укажите код и описание источника')
        try:
            updated = date.fromisoformat(updated)
        except ValueError:
            raise Invalid('Дата обновления указывается в формате ГГГГ-ММ-ДД') from None
        with self.store.engine.begin() as connection:
            connection.exec_driver_sql('SELECT pg_advisory_xact_lock(5200502)')
            if profile_dataset is not None and not execute(connection, '''
                SELECT id FROM lab_dataset WHERE id=:id AND kind='IN02' AND status='ready'
                ''', id=profile_dataset).scalar():
                raise Invalid('Не найдена подготовленная версия IN02')
            execute(connection, '''INSERT INTO lab_source (code,description,synthetic)
                               VALUES (:code,:description,:synthetic) ON CONFLICT (code) DO NOTHING''',
                    code=source, description=description, synthetic=synthetic)
            entry = execute(connection, 'SELECT * FROM lab_source WHERE code=:code', code=source).mappings().one()
            if entry['synthetic'] != synthetic or entry['description'] != description:
                raise Invalid('Изменились метаданные источника; используйте новый код')
            attributes = dict(src=entry['id'], kind=kind, file_format=suffix, digest=digest(raw),
                              updated=updated, profiles=profile_dataset)
            version = execute(connection, '''
                SELECT id FROM lab_dataset WHERE source_id=:src AND kind=:kind AND file_format=:file_format AND digest=:digest
                AND source_updated=:updated AND profile_dataset_id IS NOT DISTINCT FROM CAST(:profiles AS bigint)
                ''', **attributes).scalar()
            duplicate = version is not None
            if not duplicate:
                version = execute(connection, '''
                    INSERT INTO lab_dataset (source_id,kind,file_format,digest,source_updated,profile_dataset_id,filename,original)
                    VALUES (:src,:kind,:file_format,:digest,:updated,:profiles,:filename,:raw) RETURNING id
                    ''', **attributes, filename=Path(filename).name, raw=raw).scalar_one()
        with self.store.engine.begin() as connection:
            connection.exec_driver_sql('SELECT pg_advisory_xact_lock(5200502)')
            current = execute(connection, 'SELECT status,protocol FROM lab_dataset WHERE id=:id', id=version).mappings().one()
            if current['status'] != 'registered':
                return {'version': version, 'duplicate': duplicate, **current}
            units = dict(execute(connection, 'SELECT code,unit FROM lab_indicator').tuples().all())
            profiles = dict(execute(connection, 'SELECT pseudonym,id FROM lab_profile WHERE dataset_id=:id',
                                    id=profile_dataset).tuples().all()) if profile_dataset is not None else {}
            rows, errors = check_rows(raw, suffix, kind, units, profiles)
            status = 'rejected' if errors else 'ready'
            if not errors:
                for row in rows:
                    self.save_row(connection, version, kind, row)
            protocol = {'rows': 0 if errors else len(rows), 'errors': errors,
                        'transform': 'trim; typed values; preserve missing and source precision'}
            execute(connection, 'UPDATE lab_dataset SET status=:status,protocol=CAST(:protocol AS jsonb) WHERE id=:id',
                    status=status, protocol=document(protocol), id=version)
        return {'version': version, 'duplicate': duplicate, 'status': status, 'protocol': protocol}

    @staticmethod
    def save_row(connection, version, kind, row):
        if kind == 'IN01':
            execute(connection, '''INSERT INTO lab_territory (code,level) VALUES (:territory,:level)
                       ON CONFLICT (code,level) DO NOTHING''', **row)
            execute(connection, 'INSERT INTO lab_age_band VALUES (:age_band) ON CONFLICT DO NOTHING', **row)
            execute(connection, 'INSERT INTO lab_unit VALUES (:unit) ON CONFLICT DO NOTHING', **row)
            execute(connection, 'INSERT INTO lab_indicator VALUES (:indicator,:unit) ON CONFLICT DO NOTHING', **row)
            execute(connection, '''INSERT INTO lab_observation (dataset_id,territory_id,age_band,indicator,year,value)
                SELECT :version,id,:age_band,:indicator,:year,:value FROM lab_territory WHERE code=:territory AND level=:level
                ''', version=version, **row)
        elif kind == 'IN02':
            execute(connection, 'INSERT INTO lab_profile (dataset_id,pseudonym,features) VALUES (:id,:name,CAST(:data AS jsonb))',
                    id=version, name=row['pseudonym'], data=document(row['features']))
        else:
            execute(connection, 'INSERT INTO lab_label VALUES (:version,:profile,:class_code)', version=version, **row)
