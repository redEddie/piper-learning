"""Queue a campaign after a preserved comparison run, without GPU overlap."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('campaign command is required after --')
    if args.state.exists():
        raise FileExistsError(args.state)
    state = dict(status='waiting_for_comparison_run', wait_for=str(args.wait_for.resolve()),
        command=command, started_utc=datetime.now(timezone.utc).isoformat(), child_pid=None)

    def save():
        temporary=args.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(args.state)

    try:
        save()
        while True:
            prerequisite=json.loads(args.wait_for.read_text())
            state['prerequisite_status']=prerequisite['status']
            state['last_checked_utc']=datetime.now(timezone.utc).isoformat()
            save()
            if prerequisite['status']=='completed':
                break
            if prerequisite['status'] in ('failed','cancelled','interrupted'):
                raise RuntimeError(f"Comparison run did not complete: {prerequisite['status']}")
            time.sleep(5)
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL)
        state.update(status='running',child_pid=process.pid)
        save()
        code=process.wait()
        if code:
            raise RuntimeError(f'Queued campaign exited {code}')
        state.update(status='completed',child_pid=None,ended_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        state.update(status='failed',error=str(error),ended_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == '__main__':
    main()
