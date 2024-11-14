from __future__ import annotations
import copy
from typing import List

from pydantic import BaseModel, field_validator, computed_field


class PolygonBox(BaseModel):
    polygon: List[List[float]]

    @field_validator('polygon')
    @classmethod
    def check_elements(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 4:
            raise ValueError('corner must have 4 elements')

        for corner in v:
            if len(corner) != 2:
                raise ValueError('corner must have 2 elements')
        return v

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2]

    @property
    def size(self):
        return [self.width, self.height]

    @computed_field
    @property
    def bbox(self) -> List[float]:
        box = [self.polygon[0][0], self.polygon[0][1], self.polygon[1][0], self.polygon[2][1]]
        if box[0] > box[2]:
            box[0], box[2] = box[2], box[0]
        if box[1] > box[3]:
            box[1], box[3] = box[3], box[1]
        return box

    def minimum_gap(self, other: PolygonBox):
        if self.intersection_pct(other.bbox) > 0:
            return 0

        x_dist = min(abs(self.bbox[0] - other.bbox[2]), abs(self.bbox[2] - other.bbox[0]))
        y_dist = min(abs(self.bbox[1] - other.bbox[3]), abs(self.bbox[3] - other.bbox[1]))

        if x_dist == 0 or self.overlap_x(other) > 0:
            return y_dist
        if y_dist == 0 or self.overlap_y(other) > 0:
            return x_dist

        return (x_dist ** 2 + y_dist ** 2) ** 0.5

    def center_distance(self, other: PolygonBox):
        return ((self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2) ** 0.5

    def rescale(self, processor_size, image_size):
        # Point is in x, y format
        page_width, page_height = processor_size

        img_width, img_height = image_size
        width_scaler = img_width / page_width
        height_scaler = img_height / page_height

        new_corners = copy.deepcopy(self.polygon)
        for corner in new_corners:
            corner[0] = corner[0] * width_scaler
            corner[1] = corner[1] * height_scaler
        self.polygon = new_corners

    def fit_to_bounds(self, bounds):
        new_corners = copy.deepcopy(self.polygon)
        for corner in new_corners:
            corner[0] = max(min(corner[0], bounds[2]), bounds[0])
            corner[1] = max(min(corner[1], bounds[3]), bounds[1])
        self.polygon = new_corners

    def merge(self, other: PolygonBox):
        x1 = min(self.bbox[0], other.bbox[0])
        y1 = min(self.bbox[1], other.bbox[1])
        x2 = max(self.bbox[2], other.bbox[2])
        y2 = max(self.bbox[3], other.bbox[3])
        self.polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def overlap_x(self, other: PolygonBox):
        return max(0, min(self.bbox[2], other.bbox[2]) - max(self.bbox[0], other.bbox[0]))

    def overlap_y(self, other: PolygonBox):
        return max(0, min(self.bbox[3], other.bbox[3]) - max(self.bbox[1], other.bbox[1]))

    def intersection_area(self, other: PolygonBox):
        return self.overlap_x(other) * self.overlap_y(other)

    def intersection_pct(self, other: PolygonBox, x_margin=0, y_margin=0):
        assert 0 <= x_margin <= 1
        assert 0 <= y_margin <= 1
        if self.area == 0:
            return 0

        if x_margin:
            x_margin = int(min(self.width, other.width) * x_margin)
        if y_margin:
            y_margin = int(min(self.height, other.height) * y_margin)

        intersection = self.intersection_area(other)
        return intersection / self.area

    @classmethod
    def from_bbox(cls, bbox: List[float]):
        return cls(polygon=[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
