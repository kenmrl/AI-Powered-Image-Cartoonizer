from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

def cartoonize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

@app.route("/cartoonize", methods=["POST"])
def cartoonize():
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image)
    cartoon_image = cartoonize_image(image)

    _, buffer = cv2.imencode(".jpg", cartoon_image)
    return send_file(io.BytesIO(buffer), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
