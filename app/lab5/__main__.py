import argparse
import os
import subprocess
from pathlib import Path

from sqlalchemy.engine import make_url

from .store import Denied, Invalid, Store, digest


def pg_env(database_url):
    parsed = make_url(database_url)
    values = {'PGHOST': parsed.host, 'PGPORT': str(parsed.port) if parsed.port else None,
              'PGUSER': parsed.username, 'PGPASSWORD': parsed.password, 'PGDATABASE': parsed.database,
              'PGSSLMODE': parsed.query.get('sslmode')}
    return {**{key: value for key, value in os.environ.items() if key not in values},
            **{key: value for key, value in values.items() if value is not None}}


def executable(name):
    directory = os.getenv('LAB5_PG_BIN')
    return str(Path(directory) / name) if directory else name


def maintenance(store, action, file):
    actor = store.authorize(os.getenv('LAB5_TOKEN'), action, admin=True)
    source = os.environ['LAB5_DATABASE_URL']
    checksum = file.with_suffix(file.suffix + '.sha256')
    try:
        if action == 'backup':
            if file.exists() or checksum.exists():
                raise Invalid('Укажите новое имя файла копии')
            with file.open('xb') as output:
                subprocess.run([executable('pg_dump'), '--no-password', '--format=custom'], env=pg_env(source),
                               stdout=output, check=True)
            checksum.write_text(digest(file.read_bytes()) + '\n', encoding='ascii')
        else:
            target = os.environ['LAB5_RESTORE_DATABASE_URL']
            destination = Store(target)
            try:
                with destination.engine.connect() as connection:
                    occupied = connection.exec_driver_sql('''
                        SELECT count(*) FROM information_schema.tables
                        WHERE table_schema NOT IN ('pg_catalog','information_schema')
                        ''').scalar_one()
                if occupied or pg_env(source).get('PGDATABASE') == pg_env(target).get('PGDATABASE'):
                    raise Invalid('Восстановление допускается только в отдельную пустую БД с другим именем')
                if digest(file.read_bytes()) != checksum.read_text(encoding='ascii').strip():
                    raise Invalid('Контрольная сумма архива не совпадает')
                subprocess.run([executable('pg_restore'), '--no-password', '--single-transaction', '--no-owner',
                                '--no-privileges', '--dbname=' + pg_env(target)['PGDATABASE'], str(file)],
                               env=pg_env(target), check=True)
                with destination.engine.connect() as connection:
                    connection.exec_driver_sql('SELECT count(*) FROM lab_dataset').scalar_one()
                    connection.exec_driver_sql('SELECT count(*) FROM lab_model').scalar_one()
            finally:
                destination.engine.dispose()
    except Exception:
        store.audit(actor['id'], action, 'failed')
        raise
    store.audit(actor['id'], action, 'ok')


def main():
    parser = argparse.ArgumentParser(description='Администрирование учебной ИС')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init')
    provision = commands.add_parser('user')
    provision.add_argument('name')
    provision.add_argument('role', choices=['analyst', 'admin'])
    for command in ('backup', 'restore'):
        commands.add_parser(command).add_argument('file', type=Path)
    args = parser.parse_args()
    store = Store()
    try:
        if args.command == 'init':
            store.migrate()
        elif args.command == 'user':
            print(store.provision(args.name, args.role))
        else:
            maintenance(store, args.command, args.file)
    except (Invalid, Denied) as error:
        parser.exit(1, f'{error}\n')
    finally:
        store.engine.dispose()


if __name__ == '__main__':
    main()
