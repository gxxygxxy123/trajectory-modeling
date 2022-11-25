import os
import argparse
from Model3D_offline import triangluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Do 3D')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    args = parser.parse_args()

    for (root, dirs, files) in os.walk(args.folder):
        if 'config' in files:
            triangluation(os.path.join(root, 'config'), atleast1hit=False)
