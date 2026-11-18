import csv
import html
import io
import os
from pathlib import Path
from xml.sax.saxutils import escape

from openpyxl import Workbook
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

from .store import Invalid, document, execute


def completed(snapshot):
    if snapshot['status'] != 'completed':
        raise Invalid('Выгрузка доступна только для завершённого запуска')
    return snapshot['result']


def table_rows(snapshot):
    result = completed(snapshot)
    yield ('Раздел', 'Ключ', 'Поле', 'Значение', 'Единица')
    for name in ('id', 'source', 'source_updated', 'dataset_version', 'model_id', 'training_source',
                 'training_profiles', 'label_dataset_id', 'label_source', 'parameters'):
        if snapshot[name] is not None:
            value = snapshot[name]
            yield ('контекст', '', name, document(value) if isinstance(value, dict) else value, '')
    if snapshot['kind'] == 'classify':
        yield ('оценка', '', 'класс', result['class'], '')
        for name, value in result['probabilities'].items():
            yield ('оценка', name, 'вероятность', value, 'доля')
    else:
        for name in ('n', 'excluded', 'train_n', 'test_n'):
            yield ('выборка', '', name, result[name], 'пар')
        for indicator, stats in result['stats'].items():
            for metric, value in stats.items():
                if metric != 'unit':
                    yield ('статистика', indicator, metric, value, stats['unit'] if metric != 'n' else 'пар')
        for pair in result['correlation']:
            yield ('корреляция', pair['left'], pair['right'], pair['r'], '')
        unit = result['stats'][snapshot['parameters']['target']]['unit']
        for method, scores in result['metrics'].items():
            for metric, value in scores.items():
                yield ('регрессия', method, metric, value, unit if metric != 'r2' else '')
        yield ('ridge', '', 'intercept', result['ridge_intercept'], unit)
        for name, value in result['ridge_coefficients'].items():
            yield ('ridge', name, 'коэффициент', value, '')
        for entry in result['predictions']:
            key = f'{entry["territory"]} / {entry["year"]} / {entry["method"]}'
            for field in ('actual', 'estimate'):
                yield ('прогноз', key, field, entry[field], entry['unit'])


def export(snapshot, format_name):
    if format_name == 'csv':
        stream = io.StringIO(newline='')
        csv.writer(stream).writerows([["'" + cell if isinstance(cell, str) and cell.lstrip().startswith(('=', '+', '-', '@'))
                                      else cell for cell in row] for row in table_rows(snapshot)])
        return stream.getvalue().encode('utf-8'), 'text/csv; charset=utf-8'
    if format_name == 'xlsx':
        workbook = Workbook()
        for row in table_rows(snapshot):
            workbook.active.append([None if cell is None else str(cell) for cell in row])
            for cell in workbook.active[workbook.active.max_row]:
                cell.data_type = 's'
        stream = io.BytesIO()
        workbook.save(stream)
        return stream.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    if format_name == 'pdf':
        return pdf_report(snapshot), 'application/pdf'
    raise Invalid('Допустимые форматы: CSV, XLSX и PDF')


def pdf_report(snapshot):
    result = completed(snapshot)
    fonts = [os.getenv('LAB5_PDF_FONT'), '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
             str(Path(os.getenv('WINDIR', 'C:/Windows')) / 'Fonts' / 'arial.ttf')]
    if 'Lab5Cyrillic' not in pdfmetrics.getRegisteredFontNames():
        font = next((path for path in fonts if path and Path(path).is_file()), None)
        if font is None:
            raise Invalid('Укажите LAB5_PDF_FONT: путь к TTF со знаками кириллицы')
        pdfmetrics.registerFont(TTFont('Lab5Cyrillic', font))
    stream = io.BytesIO()
    style = ParagraphStyle('lab5', fontName='Lab5Cyrillic', fontSize=10, leading=14, spaceAfter=4)
    title = ParagraphStyle('lab5title', parent=style, fontSize=15, leading=20, spaceAfter=10)
    story = [Paragraph(f'Результат запуска № {snapshot["id"]}', title)]
    lines = [result['limitation']]
    lines.extend(' · '.join(str(value) for value in row if value is not None and value != '')
                 for row in table_rows(snapshot) if row[0] in {'контекст', 'выборка', 'оценка', 'регрессия', 'ridge'}
                 and row[2] != 'parameters')
    parameters = dict(snapshot['parameters'])
    territories = parameters.pop('territories', [])
    lines.append('Параметры: ' + document(parameters))
    if territories:
        lines.append(f'Число выбранных территорий: {len(territories)}. Полный список сохранён в параметрах запуска.')
    lines.extend(item['body'] for item in snapshot['explanations'] if item['body'])
    if snapshot['comments']:
        latest = snapshot['comments'][-1]
        lines.append(f'Последний комментарий ({latest["author"]}): {latest["body"]}')
    if snapshot['kind'] == 'territory':
        lines.append('Полные описательные статистики, корреляции и прогнозы: CSV или XLSX этого запуска.')
    story.extend(Paragraph(escape(line), style) for line in lines)
    SimpleDocTemplate(stream, title=f'Запуск {snapshot["id"]}').build(story)
    return stream.getvalue()


def graph(snapshot):
    result = completed(snapshot)
    if snapshot['kind'] == 'classify':
        elements = []
        for index, (name, value) in enumerate(result['probabilities'].items()):
            height = 18 + 30 * index
            elements.append(f'<text x="0" y="{height+14}">{html.escape(name)}</text>'
                            f'<rect x="105" y="{height}" width="{value*340:.3f}" height="17" fill="#166a83"/>'
                            f'<text x="455" y="{height+14}">{value:.2%}</text>')
        return '<svg viewBox="0 0 520 120" aria-label="Вероятности классов">' + ''.join(elements) + '</svg>'
    values = [row for row in result['predictions'] if row['method'] == 'ridge']
    limits = [row[field] for row in values for field in ('actual', 'estimate')]
    low, high = min(limits), max(limits)
    span = high - low or 1
    points = ''.join(f'<circle cx="{40+240*(row["actual"]-low)/span:.3f}" '
                     f'cy="{260-240*(row["estimate"]-low)/span:.3f}" r="3" fill="#166a83"/>' for row in values)
    return ('<svg viewBox="0 0 345 300" aria-label="Факт и прогноз Ridge">'
            '<path d="M40 20 V260 H280 M40 260 L280 20" fill="none" stroke="#717780"/>'
            '<text x="145" y="290">Факт</text><text x="44" y="17">Прогноз Ridge</text>' + points + '</svg>')


def add_comment(store, actor, run_id, body):
    completed(store.read_run(actor, run_id))
    if not 1 <= len(body.strip()) <= 4000:
        raise Invalid('Комментарий должен содержать от 1 до 4000 знаков')
    with store.engine.begin() as connection:
        return execute(connection, '''INSERT INTO lab_comment (run_id,actor_id,body) VALUES (:run,:actor,:body)
                       RETURNING id''', run=run_id, actor=actor['id'], body=body.strip()).scalar_one()
