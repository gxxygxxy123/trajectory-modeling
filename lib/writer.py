"""
Object: write down the point record
"""
import pandas as pd
import csv
import logging
from operator import attrgetter

from point import Point

class CSVWriter():
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.csvfile = open(filename, 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])

    def close(self):
        self.csvfile.flush()
        self.csvfile.close()

        df = pd.read_csv(self.filename)
        df = df.sort_values(by=["Frame"])
        df.to_csv(self.filename, mode='w+', index=False)

    def writePoints(self, points):
        if isinstance(points, Point): # A point
            self.writer.writerow([points.fid, points.visibility, points.x, points.y, points.z, points.event, points.timestamp])
        elif isinstance(points, list): # A point list
            for p in points:
                self.writer.writerow([p.fid, p.visibility, p.x, p.y, p.z, p.event, p.timestamp])

        self.csvfile.flush()

    def setEventByTimestamp(self, event, timestamp): # Bug TODO
        df = pd.read_csv(self.filename)
        df.loc[df['Timestamp'] == timestamp, 'Event'] = event

        # writing into the file
        df.to_csv(self.filename, mode='w+', index=False)

class CSVWriter_pd():
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.points = pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])
        self.time_events = {} # {10:1, 20:1, 30:1, 40:2} time 10,20,30 has event 1, time 40 has event 2

    def close(self):
        logging.info("[CSVWriter] Saving File {} ...".format(self.filename))

        # Sort Points by timestamp
        self.points.sort(key=lambda x: x.fid)

        # Set Events Detected By EventDetector By Nearest Timestamp
        for t,event in self.time_events.items():
            # Binary Search
            low = 0
            high = len(self.points)-1
            close_idx = low
            while low <= high:
                mid = low + (high - low) // 2
                if self.points[mid].timestamp < t:
                    low = mid + 1
                elif self.points[mid].timestamp > t:
                    high = mid - 1
                else:
                    close_idx = mid
                    break
                if abs(self.points[mid].timestamp - t) < abs(self.points[close_idx].timestamp - t):
                    close_idx = mid
            self.points[close_idx].event = event

        # Fill 2D CSV rows empty points (Tracknet detect failed)
        # if self.points:
        #     max_fid = max(p.fid for p in self.points)
        # else:
        #     max_fid = -1
        # self.fill_empty_rows_by_max_FID(max_fid)

        # Convert Point to List
        list_points = [None] * len(self.points)
        for idx, p in enumerate(self.points):
            list_points[idx] = [p.fid, p.visibility, p.x, p.y, p.z, p.event, p.timestamp]

        # Convert List to DataFrame
        df = pd.DataFrame(list_points, columns=['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])

        # SetEventByFID
        # if fid in df['Frame']:
        #     df.loc[df['Frame'] == fid, 'Event'] = event

        df = df.sort_values(by=['Frame','Timestamp'], ignore_index=True)
        df.to_csv(self.filename, index=False)
        logging.info("[CSVWriter] Finish Saving {}".format(self.filename))

    def setEventByTimestamp(self, event, timestamp):
        self.time_events[timestamp] = event

    def writePoints(self, points): # TODO if points already exists, update (Avoid duplicate points)
        if isinstance(points, Point): # A point
            self.points.append(points)
        elif isinstance(points, list): # A point list
            for p in points:
                self.points.append(p)


    def fill_empty_rows_by_max_FID(self, fid):
        if fid == -1: # Model3D.csv or empty data, skip
            return
        for p in self.points:
            if fid < p.fid:
                logging.warning("[CSVWriter][fill_empty_rows_by_max_FID] "
                                "Your fid is smaller than one current point! "
                                "Function Failed")
                return

        new_points = [None] * (fid+1) # fid start from 0
        for p in self.points:
            new_points[p.fid] = p
        for idx, new_p in enumerate(new_points):
            if new_p is None:
                new_points[idx] = Point(fid=idx, visibility=0, x=0.0, y=0.0, z=0.0, event=0, timestamp=0.0) # TODO timestamp interpolation by fps

        self.points = new_points
