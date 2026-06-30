# Python 2.7 REST API wrapping nesmdb.convert.main
import os
import sys
import traceback

from flask import Flask, request, jsonify

# ensure you are loadin where `vgm2wav` is acessible
# /usr/local/lib/python2.7/site-packages/nesmdb/vgm2wav
from nesmdb.convert import main

app = Flask(__name__)

DATA_DIR = '/data'

# Same (in_type, out_type) table used inside nesmdb.convert.main().
# Replicated here so we can compute the deterministic output path.
CONVERSION_TO_TYPES = {
    # VGM simplifiers
    'vgm_simplify': ('.vgm', '.simp.vgm'),
    'vgm_shorten': ('.vgm', '.short.vgm'),
    # NES disassembly raw
    'vgm_to_ndr': ('.vgm', '.ndr.pkl'),
    'ndr_to_txt': ('.ndr.pkl', '.ndr.txt'),
    'txt_to_ndr': ('.ndr.txt', '.ndr.pkl'),
    'ndr_to_vgm': ('.ndr.pkl', '.ndr.vgm'),
    # NES disassembly functional
    'vgm_to_ndf': ('.vgm', '.ndf.pkl'),
    'ndf_to_txt': ('.ndf.pkl', '.ndf.txt'),
    'txt_to_ndf': ('.ndf.txt', '.ndf.pkl'),
    'ndf_to_vgm': ('.ndf.pkl', '.ndf.vgm'),
    # NES language modeling format
    'vgm_to_nlm': ('.vgm', '.nlm.pkl'),
    'nlm_to_vgm': ('.nlm.pkl', '.nlm.vgm'),
    # NES-MDB score formats
    'ndf_to_exprsco': ('.ndf.pkl', '.exprsco.pkl'),
    'ndf_to_midi': ('.ndf.pkl', '.mid'),
    'exprsco_to_seprsco': ('.exprsco.pkl', '.seprsco.pkl'),
    'exprsco_to_blndsco': ('.exprsco.pkl', '.blndsco.pkl'),
    # WAV converters
    'vgm_to_wav': ('.vgm', '.wav'),
    'ndr_to_wav': ('.ndr.pkl', '.wav'),
    'ndf_to_wav': ('.ndf.pkl', '.wav'),
    'nlm_to_wav': ('.nlm.pkl', '.wav'),
    'midi_to_wav': ('.mid', '.mid.wav'),
    'exprsco_to_wav': ('.exprsco.pkl', '.exprsco.wav'),
    'seprsco_to_wav': ('.seprsco.pkl', '.seprsco.wav'),
    'blndsco_to_wav': ('.blndsco.pkl', '.blndsco.wav'),
}

VALID_KEYS = set(CONVERSION_TO_TYPES.keys())

def proc():
    """Run the conversion. Returns 'Success' on normal completion.

    Raises if main() raises (bad args, unreadable input, etc.).
    Note: nesmdb's main() swallows conversion-step errors internally,
    so a returned 'Success' is validated afterwards via os.path.exists.
    """
    main()
    return "Success"


def _is_under(path, base):
    """True if `path` resolves inside `base` (prevents path escape)."""
    abs_base = os.path.abspath(base)
    abs_path = os.path.abspath(path)
    return abs_path == abs_base or abs_path.startswith(abs_base + os.sep)


def _expected_output_path(convertionkey, path_to_input, output_dir):
    """Replicates the output-path logic from nesmdb.convert.main()."""
    in_type, out_type = CONVERSION_TO_TYPES[convertionkey]
    out_fn = os.path.basename(path_to_input).replace(in_type, out_type)
    return os.path.join(output_dir, out_fn)


class ApiError(Exception):
    def __init__(self, message, status_code=400):
        Exception.__init__(self, message)
        self.message = message
        self.status_code = status_code


@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify({'error': err.message}), err.status_code

@app.route('/data-mount-point', methods=['GET'])
def data_mount_point():
    value = os.environ.get('DATA_MOUNT_POINT')
    if value is None:
        raise ApiError('DATA_MOUNT_POINT environment variable is not set.', 404)
    return jsonify({'DATA_MOUNT_POINT': value}), 200


@app.route('/convert', methods=['POST'])
def convert():


    payload = request.get_json(silent=True)
    if payload is None:
        raise ApiError('Request body must be valid JSON.', 400)

    convertionkey = payload.get('convertionkey').encode('utf-8')
    path_to_input = payload.get('path_to_input').encode('utf-8')
    output_dir = payload.get('output_dir').encode('utf-8')

    # --- Validation ---
    if not convertionkey or not path_to_input or not output_dir:
        raise ApiError(
            'Missing required field(s): convertionkey, path_to_input, output_dir.',
            400)

    if convertionkey not in VALID_KEYS:
        raise ApiError(
            'Invalid convertionkey "{}". Must be one of: {}.'.format(
                convertionkey, ', '.join(sorted(VALID_KEYS))),
            400)

    if not _is_under(path_to_input, DATA_DIR):
        raise ApiError('path_to_input must be located under {}.'.format(DATA_DIR), 400)

    if not _is_under(output_dir, DATA_DIR):
        raise ApiError('output_dir must be located under {}.'.format(DATA_DIR), 400)

    if not os.path.isfile(path_to_input):
        raise ApiError('Input file not found: {}.'.format(path_to_input), 400)

    # --- Run conversion ---
    out_fp = _expected_output_path(convertionkey, path_to_input, output_dir)

    sys.argv = ["convert.py", convertionkey, path_to_input, "--out_dir", output_dir]
    print(sys.argv)
    # sys.argv = ["convert.py", "midi_to_wav", "/data/348_TheLegendofZelda_01_02OverworldBGM_destokenized_v3.mid", "--out_dir", "/data/test_api/"]
    try:
        result = proc()
    except SystemExit as e:
        # argparse calls sys.exit() on bad arguments.
        raise ApiError('Conversion failed (argument error): {}.'.format(e), 500)
    except Exception:
        raise ApiError('Conversion failed:\n{}'.format(traceback.format_exc()), 500)

    # --- Validate result ---
    if result != "Success":
        raise ApiError('Conversion did not report success.', 500)

    # Backstop for nesmdb's silently-swallowed conversion errors.
    if not os.path.isfile(out_fp):
        raise ApiError(
            'Conversion reported success but output file was not created: {}.'.format(out_fp),
            500)

    return jsonify({'file_path': out_fp}), 200


if __name__ == '__main__':
    print("PWD: ", os.getcwd())
    import nesmdb
    import inspect
    print("NESMDB: ", inspect.getfile(nesmdb))
    # Single request at a time: threaded=False, single process.
    app.run(host='0.0.0.0', port=5555, threaded=False, processes=1)