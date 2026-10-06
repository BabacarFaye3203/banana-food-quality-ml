from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal



class IoTMeasurement(BaseModel):
    temperature: float
    humidity: float
    mq2_raw: float = Field(ge=0, le=4095)
    fc28_raw: float = Field(ge=0, le=4095)
    liquid_detected: int = Field(ge=0, le=1)
    motion_detected: int = Field(ge=0, le=1)
    ldr_raw: float = Field(ge=0, le=4095)


class IoTMeasurementRequest(BaseModel):
    id: int | None = None
    device_id: str
    temperature: float
    humidity: float
    mq2_raw: float = Field(ge=0, le=4095)
    mq2_voltage: float = Field(ge=0)
    fc28_raw: float = Field(ge=0, le=4095)
    liquid_detected: int = Field(ge=0, le=1)
    motion_detected: int = Field(ge=0, le=1)
    ldr_raw: float = Field(ge=0, le=4095)
    light_label: Literal["bright", "dark"]
    detected_at: datetime