import os
import sys
import configparser
import time
from typing import Optional
from vimba import *

def print_camera(cam: Camera):
    print('/// Camera Name   : {}'.format(cam.get_name()))
    print('/// Model Name    : {}'.format(cam.get_model()))
    print('/// Camera ID     : {}'.format(cam.get_id()))
    print('/// Serial Number : {}'.format(cam.get_serial()))
    print('/// Interface ID  : {}\n'.format(cam.get_interface_id()))

def print_preamble():
    print('////////////////////////////////////////////')
    print('/// Vimba API Load Save Settings Example ///')
    print('////////////////////////////////////////////\n')

def print_usage():
    print('Usage:')
    print('    python load_save_settings.py [camera_id]')
    print('    python load_save_settings.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)

def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]

def setFeatures(cam, settings_file):
    settings = {}
    config = configparser.ConfigParser()
    config.optionxform=str
    try:
        with open(settings_file) as f:
            config.read_file(f)
        if config.has_section('Camera'):
            settings = {}
            for name, value in config.items('Camera'):
                settings[name] = value
        with cam:
            for name in settings.keys():
                try:
                    if name == 'BalanceRatioAbs':
                        ratio_value = settings[name].split(',')
                        selector = cam.get_feature_by_name('BalanceRatioSelector')
                        ratio = cam.get_feature_by_name('BalanceRatioAbs')
                        selector.set('Red')
                        ratio.set(ratio_value[0])
                        selector.set('Blue')
                        ratio.set(ratio_value[1])
                    else:
                        feature = cam.get_feature_by_name(name)
                        feature.set(settings[name])
                except (AttributeError, VimbaFeatureError):
                    print("Feature \'{}\' not found.".format(name))
                    pass
    except IOError:
        print("{} is not exist.".format(settings_file))
        sys.exit(0)
    print("--> Feature values have been loaded from given file '{}'".format(settings_file))

def main():
    print_preamble()
    camera_id = parse_args()

    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()

        print('Cameras found: {}'.format(len(cams)))
        for cam in cams:
            print_camera(cam)

        print("Start load...")

        if camera_id:
            try:
                cam = vimba.get_camera_by_id(camera_id)
                settings_file = '{}.cfg'.format(cam.get_id())
                setFeatures(cam, settings_file)
            except VimbaCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
        else:
            for cam in cams:
                print("{}: ".format(cam.get_id()))
                settings_file = '{}.cfg'.format(cam.get_id())
                setFeatures(cam, settings_file)

if __name__ == '__main__':
    main()
