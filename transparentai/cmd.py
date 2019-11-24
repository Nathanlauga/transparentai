import sys
import os


def main():
    path = os.path.dirname(os.path.abspath(__file__))
    os.system('gunicorn app:app --chdir '+path)


if __name__ == '__main__':
    sys.exit(main())
