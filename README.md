# Video Event Retrieval for Vietnamese News

## Overview
This repository contains a video event retrieval system tailored specifically for Vietnamese news videos. The system aims to facilitate the extraction and retrieval of relevant events within Vietnamese news videos based on text queries.

## Features
- **Text-Based Search:** Enables users to input text queries in Vietnamese to retrieve relevant video segments.
- **Keyframe Extraction:** Utilizes keyframe extraction techniques to represent video segments and facilitate efficient retrieval.
- **CLIP Model Integration:** Employs the CLIP model for feature extraction and similarity comparison between text queries and video frames.
- **HDFS Integration:** Interacts with Hadoop Distributed File System (HDFS) to access and retrieve video data.

## Workflow
1. **Text Query Input:** Users input text queries in Vietnamese via the application's interface.
2. **CLIP Processing:** The application leverages the CLIP model to extract features from text queries.
3. **Keyframe Extraction:** Keyframes are extracted from Vietnamese news videos and processed.
4. **Cosine Similarity Calculation:** The CLIP-processed text query features are compared with keyframe features using cosine similarity.
5. **Top Relevant Keyframes:** Keyframes most similar to the text query are retrieved.
6. **Video Retrieval:** The system accesses the HDFS video database and extracts 10-second video clips based on the identified keyframes.
7. **Display Results:** The retrieved video clips are displayed in the application's interface for user interaction.

## Python Libraries

-   `PIL`: For image handling.
-   `requests`: HTTP requests for data retrieval.
-   `transformers`: Hugging Face Transformers library for CLIP models.
-   `torch`: PyTorch library for machine learning.
-   `json`: Managing JSON data.
-   `pandas`: Data manipulation and analysis.
-   `matplotlib.pyplot`: Visualizations.
-   `altair`: Another library for creating visualizations.
-   `cv2 (OpenCV)`: Image and video processing.
-   `os`: Interacting with the operating system.
-   `base64, BytesIO, StringIO`: Handling binary data.
-   `tempfile`: Managing temporary files and directories.
-   `hdfs`: Interacting with Hadoop Distributed File System (HDFS).
-   `streamlit` : Deploying the application.

## Installation and Usage
1. Clone the repository.
2. Install the required dependencies.
3. Run `keyframe_features.py` to extract keyframe features.
4. Configure the HDFS connection (remember to change the connection id in file `app.py`).
5. Run the application (`streamlit run app.py`) and input Vietnamese text queries for video retrieval.
