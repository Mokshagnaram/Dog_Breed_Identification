import os
from uuid import uuid4

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from predict import predict_breed, load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit.

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file part in the request.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Invalid file type. Please upload png, jpg, jpeg, bmp, or webp.",
        )

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid4().hex}_{safe_name}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(file_path)
        # Ensure model loads to catch missing model errors early.
        load_model()
        breed, confidence = predict_breed(file_path)

        return render_template(
            "index.html",
            predicted_breed=breed,
            confidence=confidence,
            uploaded_image=file_path,
        )
    except FileNotFoundError as exc:
        return render_template("index.html", error=str(exc))
    except ValueError as exc:
        return render_template("index.html", error=str(exc))
    except Exception:
        return render_template(
            "index.html",
            error="An unexpected error occurred during prediction. Please try again.",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
