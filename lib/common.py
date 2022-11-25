'''
functions: common functions
'''
import os
import sys
import logging
import configparser

from datetime import datetime

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
LOG_INI = f"{ROOTDIR}/log/log.ini"

def insertById(l, x):
    for i in range(len(l)):
        p = l[i]
        if p.fid > x.fid:
            l.insert(i, x)
            return
    l.append(x)

# save/load project_info into project_filename
def saveConfig(project_filename, project_info):
    try:
        with open(project_filename, 'w') as configfile:
            project_info.write(configfile)
    except IOError as e:
        logging.error(e)
        sys.exit()

def loadConfig(cfg_file):
    try:
        config = configparser.ConfigParser()
        config.optionxform = str
        with open(cfg_file) as f:
            config.read_file(f)
    except IOError as e:
        logging.error(e)
        sys.exit()
    return config

def loadNodeConfig(cfg_file, node_name):
    # loading configuartion file
    config = configparser.ConfigParser()
    try:
        settings = {}
        with open(cfg_file) as f:
            config.read_file(f)

        if config.has_section('Project'):
            for name, value in config.items('Project'):
                settings[name] = value
        if config.has_section(node_name):
            for name, value in config.items(node_name):
                settings[name] = value
        # setup Logging Level
        setupLogLevel(level=settings['logging_level'], log_file=node_name)
        return settings
    except IOError:
        logging.error("config file does not exist.")

def updateLastExeDate():
    with open(f"{ROOTDIR}/log/log.ini", "w") as f:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        f.write(dt_string)
    return dt_string

def setupLogLevel(level, log_file):
    try:
        if not os.path.isfile(LOG_INI):
            with open(LOG_INI, "w") as f:
                dt_string = updateLastExeDate()
        else:
            with open(LOG_INI, "r") as f:
                dt_string = f.readline()

        logPath = f"{ROOTDIR}/log/{dt_string}"
        if not os.path.isdir(logPath):
            os.makedirs(logPath)

        logFormatter = logging.Formatter("%(asctime)s %(levelname).1s %(lineno)03s: %(message)s")
        rootLogger = logging.getLogger()

        fileHandler = logging.FileHandler(f"{logPath}/{log_file}.log")
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        if level.lower() == "debug":
            rootLogger.setLevel(logging.DEBUG)
        elif level.lower() == "info":
            rootLogger.setLevel(logging.INFO)
        else:
            rootLogger.setLevel(logging.ERROR)
    except FileExistsError:
        logging.error("Can't open log directory")

