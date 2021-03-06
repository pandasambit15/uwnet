import xarray as xr
import json

import click

from sam.case import InitialConditionCase, get_ngqaua_ic, default_parameters

NGAQUA_ROOT = "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"



@click.command()
@click.argument('path')
@click.option(
    '-nn',
    '--neural-network',
    type=click.Path(),
    help='use the neural network in this pickled model file.')
@click.option('-ic', '--initial-condition', type=click.Path(), default=None)
@click.option('-n', '--ngaqua-root', type=click.Path(), default=NGAQUA_ROOT)
@click.option('-t', type=int, default=0)
@click.option('-p', '--parameters', type=click.Path(), default=None)
@click.option('-d', '--debug',  is_flag=True)
def main(path,
         neural_network,
         initial_condition,
         ngaqua_root,
         t,
         model_run_path='model.pkl',
         parameters=None,
         debug=False):
    """Create SAM case directory for an NGAqua initial value problem and optionally
    run the model with docker.

    """
    if parameters:
        parameters = json.load(open(parameters))
    else:
        parameters = default_parameters()

    if initial_condition is None:
        initial_condition = get_ngqaua_ic(ngaqua_root, t)
    else:
        initial_condition = xr.open_dataset(initial_condition)

    case = InitialConditionCase(path=path, ic=initial_condition,
                                sam_src="/opt/sam",
                                prm=parameters)

    # configure neural network run
    if neural_network:
        case.prm['python']['dopython'] = False

        # setup the neural network
        case.prm['python'].update(
            dict(
                dopython=True,
                usepython=True,
                function_name='call_neural_network',
                module_name='uwnet.sam_interface'))

        case.mkdir()

        print(f"Copying neural networks to model directory")
        case.add(neural_network, model_run_path)
        case.env.update(dict(UWNET_MODEL=model_run_path))

    if 'nudging' in parameters:
        config = parameters['nudging']
        case.env['UWNET_NUDGE_TIME_SCALE'] = config['time_scale']

    if debug:
        case.prm['parameters'].update({
            'nsave3d': 20,
            'nsave2d': 20,
            'nstat': 20,
            'nstop': 120,
        })

    case.save()

if __name__ == '__main__':
    main()
