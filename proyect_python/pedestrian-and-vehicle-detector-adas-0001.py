# Copyright 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or
# implied warranties, other than those that are expressly stated in the License

import os
import logging
import cv2
import numpy as np
import json
import threading
from openvino.inference_engine import IECore
from datetime import datetime
import time
import sys

supported_devices = [
    "CPU",
    "GPU",
    "HDDL",
    "MYRIAD",
    "HETERO:CPU,GPU",
]


class Udf:
    """Obstacles detection UDF"""

    def __init__(
        self, model_xml, model_bin, device, confidence_threshold, num_infer_requests
    ):
        """UDF Constructor"""
        self.log = logging.getLogger("OBSTACLES_DETECTION")
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.num_infer_requests = int(num_infer_requests)

        if not os.path.exists(self.model_xml):
            raise FileNotFoundError(f"Model xml file missing: {self.model_xml}")
        if not os.path.exists(self.model_bin):
            raise FileNotFoundError(f"Model bin file missing: {self.model_bin}")
        if self.device not in supported_devices:
            raise ValueError(
                f"Not supported device: {self.device} to run Video Analytics"
            )

        self.log.info("Config reading completed...")
        self.log.info(f"Confidence = {self.confidence_threshold}")
        self.log.info(
            f"Loading IR files. \n\txml: {self.model_xml}, \n\tbin: {self.model_bin}"
        )

        # Load OpenVINO model
        self.ie = IECore()
        self.neural_net = self.ie.read_network(
            model=self.model_xml, weights=self.model_bin
        )
        self.log.info("Scanning IR files completed...")

        if self.neural_net is not None:
            self.input_blob = next(iter(self.neural_net.input_info))
            self.output_blob = next(iter(self.neural_net.outputs))
            self.neural_net.batch_size = 1
            self.execution_net = self.ie.load_network(
                network=self.neural_net, device_name=self.device.upper()
            )

    def __check_detection_area(self, x, y, metadata):
        detection_area = metadata["detection_area"]
        if len(detection_area) != 2:
            raise ValueError("Invalid number of points in detection area")
        top_left = detection_area[0]
        bottom_right = detection_area[1]
        # Get coordinates
        xmin, ymin = top_left[0], top_left[1]
        xmax, ymax = bottom_right[0], bottom_right[1]
        # Check if the point is inside a ROI
        return xmin < x and x < xmax and ymin < y and y < ymax

    def process(self, frame, metadata):
        """[summary]

        :param frame: frame blob
        :type frame: numpy.ndarray
        :param metadata: frame's metadata
        :type metadata: str
        :return:  (should the frame be dropped, has the frame been updated,
                   new metadata for the frame if any)
        :rtype: (bool, numpy.ndarray, str)
        """
        self.log.debug(f"Entering Obstacles Detection Demo Udf::process() function...")
        curr_time = datetime.now()
        date_str = curr_time.strftime("%m/%d/%Y")
        time_str = curr_time.strftime("%T")
        self.log.debug(f"Local date: {date_str} Local time: {time_str}")

        obstacles = []
        draw_areas = []
        highest_accuracy = 0
        n, c, h, w = self.neural_net.inputs[self.input_blob].shape
        cur_request_id = 0
        initial_h = frame.shape[0]
        initial_w = frame.shape[1]

        self.log.debug(f"Resizing the image to:  width={w}, height={h}")
        in_frame = cv2.resize(frame, (w, h))

        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        self.execution_net.start_async(
            request_id=cur_request_id, inputs={self.input_blob: in_frame}
        )

        if self.execution_net.requests[cur_request_id].wait(-1) == 0:
            # Parse detection results of the current request
            results = self.execution_net.requests[cur_request_id].outputs[
                self.output_blob
            ]

            # Iterate over detections
            for detection in results[0][0]:
                label = int(detection[1])
                accuracy = float(detection[2])
                # Draw only objects when accuracy is greater than configured threshold
                if accuracy > self.confidence_threshold:
                    xmin = int(detection[3] * initial_w)
                    ymin = int(detection[4] * initial_h)
                    xmax = int(detection[5] * initial_w)
                    ymax = int(detection[6] * initial_h)
                    # Central points of detection
                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2
                    # Check if central points fall inside the detection area
                    if self.__check_detection_area(x, y, metadata):
                        obstacles.append(
                            {
                                "type": label,
                                "tl": (xmin, ymin),
                                "br": (xmax, ymax),
                                "accuracy": round(accuracy, 2),
                            }
                        )
                        if accuracy > highest_accuracy:
                            draw_areas = [[xmin, ymin, xmax, ymax]]
                            highest_accuracy = accuracy
        # Update metada
        metadata["obstacles"] = obstacles
        metadata["draw_areas"] = draw_areas
        metadata["road_segment"] = []
        metadata["date"] = date_str
        metadata["time"] = time_str

        self.log.debug(f"metadata = {metadata}")
        return False, None, metadata
