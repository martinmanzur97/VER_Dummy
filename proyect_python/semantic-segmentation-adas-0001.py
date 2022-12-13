# Copyright 2021-2022 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or
# implied warranties, other than those that are expressly stated in the License

import logging
import os

import cv2
import numpy as np
from openvino.inference_engine import IECore, StatusCode
from common.util.image_utils import BGRColors


class Udf:
    """Semantic segmentation UDF"""

    def __init__(self, model_xml, model_bin, device, confidence_threshold, paint):
        """UDF Constructor"""
        self.log = logging.getLogger("SEMANTIC_SEGMENTATION")
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.paint = bool(paint)

        if not os.path.exists(self.model_xml):
            raise FileNotFoundError("Model xml file missing: " + str(self.model_xml))
        if not os.path.exists(self.model_bin):
            raise FileNotFoundError("Model bin file missing: " + str(self.model_bin))

        self.log.info("Config reading completed...")
        self.log.info("Confidence = " + str(self.confidence_threshold))
        self.log.info(
            "Loading IR files. \n\txml: "
            + str(self.model_xml)
            + ", \n\tbin: "
            + str(self.model_bin)
        )

        # Load OpenVINO model
        self.ie = IECore()
        self.neural_net = self.ie.read_network(
            model=self.model_xml, weights=self.model_bin
        )
        self.log.info("Scanning IR files completed...")

        if self.neural_net is not None:
            self.exec_net = self.ie.load_network(
                network=self.neural_net, device_name=self.device.upper()
            )

            self.input_layer_ir = next(iter(self.exec_net.input_info))
            self.output_layer_ir = next(iter(self.exec_net.outputs))

            self.neural_net.batch_size = 1

        self.colormap = np.array(
            [
                BGRColors.BLACK.value,  # road
                BGRColors.BLACK.value,  # sidewalk
                BGRColors.BLACK.value,  # building
                BGRColors.BLACK.value,  # wall
                BGRColors.BLUE_ALICE.value,  # fence
                BGRColors.DARK_ORANGE.value,  # pole
                BGRColors.YELLOW.value,  # traffic_light
                BGRColors.GRAY_CADETE.value,  # traffic_sign
                BGRColors.BLACK.value,  # vegetation
                BGRColors.BLACK.value,  # terrain
                BGRColors.BLACK.value,  # sky
                BGRColors.BLACK.value,  # person
                BGRColors.DARK_BLUE.value,  # rider
                BGRColors.BLACK.value,  # car
                BGRColors.DARK_BLUE.value,  # truck
                BGRColors.GRAY_CADETE.value,  # bus
                BGRColors.GREEN_SPRING1.value,  # train
                BGRColors.GREEN_SPRING2.value,  # motorcycle
                BGRColors.GREEN_SPRING3.value,  # bicycle
                BGRColors.PURPLE.value,  # ego_vehicle
            ]
        )

        # search elements in the matrix
        self.road_segment_objects = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "ego_vehicle",
        ]

        # Define the transparency of the segmentation mask on the photo
        self.alpha = 0.9

    def process(self, frame, metadata):
        """[summary]

        :param frame: frame blob
        :type frame: numpy.ndarray
        :param metadata: frame's metadata
        :type metadata: str
        :return:  (should the frame be dropped, has the frame been updated,
                   new metadata for the frame if any)
        :rtype: (bool, numpy.ndarray, str)

        return:
            road_segment: [
                "road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "traffic_light",
                "traffic_sign",
                "vegetation",
                "terrain",
                "sky",
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
                "ego_vehicle"
            ]
        """
        self.log.debug(f"Entering semantic segmentation Udf::process() function...")

        detections = metadata.get("road_segment", [])
        # N,C,H,W = batch size, number of channels, height, width
        N, C, H, W = self.neural_net.input_info[self.input_layer_ir].tensor_desc.dims

        self.log.debug(f"Resizing the image to:  width={W}, height={H}")
        resized_frame = cv2.resize(frame, (W, H))
        image_h, image_w, _ = frame.shape

        # reshape to network input shape
        # Change data layout from HWC to CHW
        input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

        # Run the infernece
        result = self.exec_net.infer(inputs={self.input_layer_ir: input_image}).get(
            self.output_layer_ir
        )

        # Prepare data for visualization
        segmentation_mask = result[0]

        elem_num = 0
        for elem in self.road_segment_objects:
            elem_num += 1
            if any(elem_num in sub for sub in segmentation_mask[0]):
                detections.append(elem)

        # Update metada
        metadata["road_segment"] = detections

        # Use function from notebook_utils.py to transform mask to an RGB image
        mask = self.segmentation_map_to_image(segmentation_mask, self.colormap)
        resized_mask = cv2.resize(mask, (image_w, image_h))

        if self.paint:
            # Create image with mask put on
            image_with_mask = cv2.addWeighted(resized_mask, self.alpha, frame, 0.8, 0)
            return False, image_with_mask, metadata

        return False, frame, metadata

    def segmentation_map_to_image(
        self, result: np.ndarray, colormap: np.ndarray, remove_holes=False
    ) -> np.ndarray:
        """
        Convert network result of floating point numbers to an RGB image with
        integer values from 0-255 by applying a colormap.

        :param result: A single network result after converting to pixel values in H,W or 1,H,W shape.
        :param colormap: A numpy array of shape (num_classes, 3) with an RGB value per class.
        :param remove_holes: If True, remove holes in the segmentation result.
        :return: An RGB image where each pixel is an int8 value according to colormap.
        """
        if len(result.shape) != 2 and result.shape[0] != 1:
            raise ValueError(
                f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
            )

        if len(np.unique(result)) > colormap.shape[0]:
            raise ValueError(
                f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
                "different output values. Please make sure to convert the network output to "
                "pixel values before calling this function."
            )
        elif result.shape[0] == 1:
            result = result.squeeze(0)

        result = result.astype(np.uint8)

        contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
        mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label_index, color in enumerate(colormap):
            label_index_map = result == label_index
            label_index_map = label_index_map.astype(np.uint8) * 255
            contours, hierarchies = cv2.findContours(
                label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                mask,
                contours,
                contourIdx=-1,
                color=color.tolist(),
                thickness=cv2.FILLED,
            )

        return mask
