import yaml
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../transparentai')))
import utils


def read_config(fpath):
    """
    """
    with utils.OpenFile(fpath) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    return conf


def generate_notebook_file(params):
    """
    """
    fpath = os.path.dirname(os.path.realpath(__file__)) + \
        '/templates/notebook_template_analyse_bias_dataset.ipynb'
    with utils.OpenFile(fpath) as file:
        notebook_str = file.read()
    file.close()

    if 'read_csv_params' in params:
        notebook_str = notebook_str.replace(
            '\'$data_path\'', f"'$data_path', {params['read_csv_params']}")

    if 'target' in params:
        notebook_str = notebook_str.replace('$target', f"{params['target']}")
    
    notebook_str = notebook_str.replace('$env', f"{params['env']}")
    notebook_str = notebook_str.replace('$data_path', f"{params['path']}")
    notebook_str = notebook_str.replace('$privileged_values', f"{params['privileged_values']}")

    fpath = params['output'] + f"Bias analyse of {params['name']} dataset.ipynb"
    utils.str_to_file(notebook_str, fpath)


def execute_notebook(params):
    """
    """
    fname = params['output'] + f"Bias analyse of {params['name']} dataset.ipynb"
    os.system(
        f'jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute "{fname}"')


def generate_notebook(params, output_format):
    """
    """
    fname = params['output'] + \
        f"Bias analyse of {params['name']} dataset.nbconvert.ipynb"
    os.system(f'jupyter nbconvert "{fname}" --to {output_format}')


def main():
    """
    """
    fpath = '/home/lauga/Documents/workspace/transparentai/scripts/config.yml'
    output_format = 'pdf'
    # select a valid env on `jupyter kernelspec list` with transparentai installed

    conf = read_config(fpath=fpath)

    generate_notebook_file(params=conf)
    execute_notebook(params=conf)
    generate_notebook(params=conf, output_format=output_format)


if __name__ == '__main__':
    main()
