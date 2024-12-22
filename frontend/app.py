import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageEnhance, Image
import io
import json


def main():
    st.title("Image Recognition App")

    if "detection_results" not in st.session_state:
        st.session_state["detection_results"] = None
    if "result_image" not in st.session_state:
        st.session_state["result_image"] = None

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Objects"):
            # Request to the FastAPI endpoint
            files = {"file": uploaded_file}
            response = requests.post("http://backend:8000/detect_objects/", files=files)

            # Process the response
            if response.status_code == 200:
                detection_results = response.json()
                boxes = detection_results["boxes"]
                scores = detection_results["scores"]
                classes = detection_results["classes"]

                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
                fig, ax = plt.subplots()
                ax.imshow(image)

                confidence_threshold = 0.65
                for box, score, cls in zip(boxes, scores, classes):
                    if score >= confidence_threshold:
                        xmin, ymin, xmax, ymax = box
                        width = xmax - xmin
                        height = ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(xmin, ymin, f"{cls} ({score:.2f})", fontsize=8, color='r')
                
                # Save the result image
                ax.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)
                result_image = buf.read()
                buf.close()

                # Save detection results and result image in session state
                st.session_state["detection_results"] = detection_results
                st.session_state["result_image"] = result_image

            else:
                st.error("Error performing object detection")
                st.error(f"Response content: {response.text}")

        if st.session_state["result_image"] is not None:
            st.image(st.session_state["result_image"], caption="Result Image with Bounding Boxes")
        
        if st.session_state["detection_results"] is not None:
            # Save detection results as JSON
            detection_json = json.dumps(st.session_state["detection_results"], indent=4)

            st.download_button(
                label="Download Result Image",
                data=st.session_state["result_image"],
                file_name="result_image.png",
                mime="image/png"
            )

            st.download_button(
                label="Download Detection JSON",
                data=detection_json,
                file_name="detection_results.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()