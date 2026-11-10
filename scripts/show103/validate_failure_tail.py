"""Compatibility entry point for task-failure/contact/reset validation.

The previous controller-failure injection was superseded when failure was
corrected to mean grasp timeout or a cube dropped outside the bowl.
"""
from pathlib import Path
import runpy

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).with_name('validate_task_failure.py')),run_name='__main__')
