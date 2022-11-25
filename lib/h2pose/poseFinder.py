import cv2
import numpy as np
try:
    from Hfinder import Hfinder
    from H2Pose import H2Pose
except:
    from h2pose.Hfinder import Hfinder
    from h2pose.H2Pose import H2Pose

class poseFinder(H2Pose):
    """docstring for poseFinder"""
    def __init__(self, img, K, pad=[0,0,0,0], downScale=False):
        if type(img) == type('string'):
            self.img = cv2.imread(img)
        elif type(img) == type(np.array([])):
            self.img = img
        self.K = K
        self.pad = pad
        self.downScale = downScale

        self.FindH()

        super(poseFinder, self).__init__(self.K, self.H)
    
    def FindH(self):
        self.hf_obj = Hfinder(self.img, court2D=[], pad=self.pad, downScale=self.downScale)
        self.H = self.hf_obj.getH()

if __name__ == '__main__':
    K = np.array([[989.09, 0, 1280//2],
                  [0, 989.09,  720//2],
                  [0,      0,       1]])
    pf = poseFinder('calib_court.png', K)
    print(pf.getCenter())
    print(pf.getC2W())