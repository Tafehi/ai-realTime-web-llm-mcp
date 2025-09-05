# server_webcam.py
import base64, cv2
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("webcam",host="localhost" , port=8000)

def _capture_jpeg(device_index=0, quality=85):
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read frame")
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

@mcp.tool()
def capture(device_index: int = 0, quality: int = 85) -> dict:
    """Capture a single frame and return {mime_type, data_base64}"""
    b64 = _capture_jpeg(device_index, quality)
    return {"mime_type": "image/jpeg", "data_base64": b64}



if __name__ == "__main__":
    # stdio transport: great for Claude Desktop / MCP Inspector
    mcp.run(transport="streamable-http")
