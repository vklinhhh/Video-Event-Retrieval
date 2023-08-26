import streamlit as st
import altair as alt
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tempfile import NamedTemporaryFile
from io import BytesIO, StringIO
from hdfs import InsecureClient
from load_css import local_css
import base64

st.set_page_config(page_title="Video Event Retrieval App",layout="wide")

local_css("styles.css")
# Connect to HDFS
hdfs_client = InsecureClient("http://192.168.1.4:9870", user="mac")
hdfs_host = "http://192.168.1.4"
hdfs_port = "9870"
hdfs_user = "mac"
hdfs_base_url = f"{hdfs_host}:{hdfs_port}/webhdfs/v1"
OPS = "OPEN"
# Load the CLIP models and tokenizer
vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

hdfs_keyframe_path = "/AIChallenge/keyframe_features/keyframe_features.json"
keyframe_url = f"{hdfs_host}:{hdfs_port}/webhdfs/v1{hdfs_keyframe_path}?op={OPS}&user.name={hdfs_user}"

# Read the keyframe features JSON file from HDFS using requests
response = requests.get(keyframe_url)
keyframe_features_dict = response.json()


def main():
    # Add this line at the beginning of your script
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://wallpaperaccess.com/full/798909.png");/Videos/L0
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<link rel="stylesheet" href="styles.css">', unsafe_allow_html=True)
    st.write(
    """
    <p style="text-align: center; color: #9e2a2b; font-size: 60px;">
        <span class="border-highlight"><strong>
        <span class="title-word title-word-1">Video</span>
        <span class="title-word title-word-2">Event</span>
        <span class="title-word title-word-3">Retrieval🎞️</span></strong>
        </span>
    </p> 
    """,
    unsafe_allow_html=True
    )



    # Get user input query
    query = st.text_input("**:orange[Enter your query]** :pencil2:")
    
    if st.button("Run"):
        run_hadoop_pipeline(query)

@st.cache_data()
def run_hadoop_pipeline(query):
    # Stage 1: Calculating the best matching key frames
    st.subheader("Progress")
    progress_stage = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Running Stage 1: Calculating best matching key frames")
    # Tokenize and process the query text
    query_inputs = tokenizer(query, padding=True, return_tensors="pt")
    text_features = vision_model.get_text_features(**query_inputs)
    
    # Iterate through each video's keyframes
    highest_similarity = -1
    best_matching_keyframe = None

    similarity_dict = {}
    for keyframe_path, keyframe_features in keyframe_features_dict.items():
        keyframe_features = torch.tensor(keyframe_features)
        # Calculate cosine similarity
        cos = torch.nn.CosineSimilarity(dim=1)
        similarity = cos(text_features, keyframe_features)
        # Store similarity and keyframe_path in the dictionary
        similarity_dict[keyframe_path] = similarity.item()
        # Update best matching keyframe
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_matching_keyframe = keyframe_path

    # Sort the similarity_dict in descending order based on similarity scores
    sorted_similarity = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

    # Get the top 10 keyframes with the highest similarity scores
    top_keyframes = sorted_similarity[:10]

    # Extract keyframe names and similarity scores for plotting
    keyframe_names = [keyframe_path.split('/')[-2] + '/' + keyframe_path.split('/')[-1] for keyframe_path, _ in top_keyframes]
    similarity_scores = [similarity for _, similarity in top_keyframes]
    st.write(f"The best matching keyframe is: {best_matching_keyframe}")

    progress_stage.progress(33)
    progress_text.text("Running Stage 2: Plotting top matching keyframes")
    similarity_df = pd.DataFrame({"Keyframe": keyframe_names, "Similarity Score": similarity_scores}).set_index("Keyframe")
    # Display similarity scores plot in the sidebar
    st.sidebar.title("Similarity Scores")
    st.sidebar.bar_chart(data=similarity_df)

    # Display keyframe images
    plt.style.use("Solarize_Light2")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Query: {query}')
    for idx, (keyframe_path, similarity) in enumerate(top_keyframes):
        # Your keyframe image retrieval code here
        keyframe_folder = keyframe_path.split('/')[0][:3]
        keyframe_id = keyframe_path.split('/')[1]
        hdfs_keyframe_path = f"/AIChallenge/Keyframes/{keyframe_folder}/{keyframe_path}"
        keyframe_image_url = f"{hdfs_base_url}{hdfs_keyframe_path}?op={OPS}&user.name={hdfs_user}"
        row = idx // 5
        col = idx % 5

        # Fetch image content from HDFS using requests
        response = requests.get(keyframe_image_url)
        keyframe_image_data = response.content
        keyframe_image = Image.open(BytesIO(keyframe_image_data))
        axes[row, col].imshow(keyframe_image)
        axes[row, col].set_title(f'Similarity: {similarity:.2f}')
        axes[row, col].axis('off')
    st.pyplot(fig)


    # Metadata path
    metadata_video = best_matching_keyframe.split('/')[0]

    hdfs_metadata_path = f"/AIChallenge/metadata/{metadata_video}.json"
    metadata_url = f"{hdfs_host}:{hdfs_port}/webhdfs/v1{hdfs_metadata_path}?op={OPS}&user.name={hdfs_user}"
    response = requests.get(metadata_url)
    metadata_dict = response.json()
    # Display metadata description in the sidebar
    st.sidebar.subheader("Video Description")
    st.sidebar.write("This is the description of the video that include the best matching image.")
    for key, value in metadata_dict.items():
        st.sidebar.markdown(f"**{key}:** {value}")

    # Plotting
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Frame 1","Frame 2","Frame 3","Frame 4","Frame 5",
                                                                           "Frame 6","Frame 7","Frame 8","Frame 9","Frame 10"])
    with tab1:
        keyframe_path_1 = top_keyframes[0][0]
        keyframe_folder_1 = keyframe_path_1[:3]
        keyframe_id_1 = keyframe_path_1.split('/')[0]
        similarity_id_1 = top_keyframes[0][1]
        hdfs_keyframe_path_1 = f"/AIChallenge/Keyframes/{keyframe_folder_1}/{keyframe_path_1}"
        keyframe_image_url_1 = f"{hdfs_base_url}{hdfs_keyframe_path_1}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_1}")
        st.subheader(f'Similarity: {similarity_id_1:.2f}')
        st.image(keyframe_image_url_1, width=500)

    with tab2:
        keyframe_path_2 = top_keyframes[1][0]
        keyframe_folder_2 = keyframe_path_2[:3]
        keyframe_id_2 = keyframe_path_2.split('/')[0]
        similarity_id_2 = top_keyframes[1][1]
        hdfs_keyframe_path_2 = f"/AIChallenge/Keyframes/{keyframe_folder_2}/{keyframe_path_2}"
        keyframe_image_url_2 = f"{hdfs_base_url}{hdfs_keyframe_path_2}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_2}")
        st.subheader(f'Similarity: {similarity_id_2:.2f}')
        st.image(keyframe_image_url_2, width=500)

    with tab3:
        keyframe_path_3 = top_keyframes[2][0]
        similarity_id_3 = top_keyframes[2][1]

        keyframe_folder_3 = keyframe_path_3[:3]
        keyframe_id_3 = keyframe_path_3.split('/')[0]
        
        hdfs_keyframe_path_3 = f"/AIChallenge/Keyframes/{keyframe_folder_3}/{keyframe_path_3}"
        keyframe_image_url_3 = f"{hdfs_base_url}{hdfs_keyframe_path_3}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_3}")
        st.subheader(f'Similarity: {similarity_id_3:.2f}')
        st.image(keyframe_image_url_3, width=500)

    with tab4:
        keyframe_path_4 = top_keyframes[3][0]
        similarity_id_4 = top_keyframes[3][1]

        keyframe_folder_4 = keyframe_path_4[:3]
        keyframe_id_4 = keyframe_path_4.split('/')[0]
        
        hdfs_keyframe_path_4 = f"/AIChallenge/Keyframes/{keyframe_folder_4}/{keyframe_path_4}"
        keyframe_image_url_4 = f"{hdfs_base_url}{hdfs_keyframe_path_4}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_4}")
        st.subheader(f'Similarity: {similarity_id_4:.2f}')
        st.image(keyframe_image_url_4, width=500)

    with tab5:
        keyframe_path_5 = top_keyframes[4][0]
        similarity_id_5 = top_keyframes[4][1]

        keyframe_folder_5 = keyframe_path_4[:3]
        keyframe_id_5 = keyframe_path_4.split('/')[0]
        
        hdfs_keyframe_path_5 = f"/AIChallenge/Keyframes/{keyframe_folder_5}/{keyframe_path_5}"
        keyframe_image_url_5 = f"{hdfs_base_url}{hdfs_keyframe_path_5}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_5}")
        st.subheader(f'Similarity: {similarity_id_5:.2f}')
        st.image(keyframe_image_url_5, width=500)

    with tab6:
        keyframe_path_6 = top_keyframes[5][0]
        similarity_id_6 = top_keyframes[5][1]

        keyframe_folder_6 = keyframe_path_6[:3]
        keyframe_id_6 = keyframe_path_6.split('/')[0]
        
        hdfs_keyframe_path_6 = f"/AIChallenge/Keyframes/{keyframe_folder_6}/{keyframe_path_6}"
        keyframe_image_url_6 = f"{hdfs_base_url}{hdfs_keyframe_path_6}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_6}")
        st.subheader(f'Similarity: {similarity_id_6:.2f}')
        st.image(keyframe_image_url_6, width=500)
    
    with tab7:
        keyframe_path_7 = top_keyframes[6][0]
        similarity_id_7 = top_keyframes[6][1]

        keyframe_folder_7 = keyframe_path_7[:3]
        keyframe_id_7 = keyframe_path_7.split('/')[0]
        
        hdfs_keyframe_path_7 = f"/AIChallenge/Keyframes/{keyframe_folder_7}/{keyframe_path_7}"
        keyframe_image_url_7 = f"{hdfs_base_url}{hdfs_keyframe_path_7}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_7}")
        st.subheader(f'Similarity: {similarity_id_7:.2f}')
        st.image(keyframe_image_url_7, width=500)

    with tab8:
        keyframe_path_8 = top_keyframes[7][0]
        similarity_id_8 = top_keyframes[7][1]

        keyframe_folder_8 = keyframe_path_8[:3]
        keyframe_id_8 = keyframe_path_8.split('/')[0]
        
        hdfs_keyframe_path_8 = f"/AIChallenge/Keyframes/{keyframe_folder_8}/{keyframe_path_8}"
        keyframe_image_url_8 = f"{hdfs_base_url}{hdfs_keyframe_path_8}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_8}")
        st.subheader(f'Similarity: {similarity_id_8:.2f}')
        st.image(keyframe_image_url_8, width=500)

    with tab9:
        keyframe_path_9 = top_keyframes[8][0]
        similarity_id_9 = top_keyframes[8][1]

        keyframe_folder_9 = keyframe_path_9[:3]
        keyframe_id_9 = keyframe_path_9.split('/')[0]
        
        hdfs_keyframe_path_9 = f"/AIChallenge/Keyframes/{keyframe_folder_9}/{keyframe_path_9}"
        keyframe_image_url_9 = f"{hdfs_base_url}{hdfs_keyframe_path_9}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_9}")
        st.subheader(f'Similarity: {similarity_id_9:.2f}')
        st.image(keyframe_image_url_9, width=500)

    with tab10:
        keyframe_path_10 = top_keyframes[9][0]
        similarity_id_10 = top_keyframes[9][1]

        keyframe_folder_10 = keyframe_path_10[:3]
        keyframe_id_10 = keyframe_path_10.split('/')[0]
        
        hdfs_keyframe_path_10 = f"/AIChallenge/Keyframes/{keyframe_folder_10}/{keyframe_path_10}"
        keyframe_image_url_10 = f"{hdfs_base_url}{hdfs_keyframe_path_10}?op={OPS}&user.name={hdfs_user}"
        st.header(f"Video: {keyframe_id_10}")
        st.subheader(f'Similarity: {similarity_id_10:.2f}')
        st.image(keyframe_image_url_10, width=500)

    # Stage 3: Video extraction
    progress_stage.progress(66)
    progress_text.text("Running Stage 3: Video extraction")
    # Display video player
    # Create a mapping of keyframes to their pts_time values
    keyframe_pts_mapping = {}

    for keyframe_path, _ in top_keyframes:
        csv_filename = keyframe_path.split('/')[-2] + '.csv'
        hdfs_csv_path = f"/AIChallenge/map-keyframes/{csv_filename}"
        csv_url = f"{hdfs_base_url}{hdfs_csv_path}?op={OPS}&user.name={hdfs_user}"
        
        # Fetch CSV content from HDFS using requests
        response = requests.get(csv_url)
        csv_content = response.text
        
        # Create a DataFrame from the CSV content
        csv_data = pd.read_csv(StringIO(csv_content))
        frame_idx = int(keyframe_path.split('/')[-1].split('.')[0])
        
        matching_row = csv_data[csv_data['n'] == frame_idx]
        if not matching_row.empty:
            pts_time = matching_row.iloc[0]['pts_time']
            keyframe_pts_mapping[keyframe_path] = pts_time

    best_matching_video_folder = best_matching_keyframe[:3]
    best_matching_video = best_matching_keyframe.split('/')[0]
    # Input video path on HDFS
    video_path_hdfs = f"/AIChallenge/Videos/{best_matching_video_folder}/{best_matching_video}.mp4"

    # Desired clip duration (in seconds)
    clip_duration = 10

    # Fetch video content from HDFS using requests
    video_url = f"{hdfs_base_url}{video_path_hdfs}?op={OPS}&user.name={hdfs_user}"
    video_response = requests.get(video_url)
    video_content = video_response.content

    # Create a temporary file and write video content to it
    with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(video_content)
        temp_file_path = temp_file.name

    # Open the downloaded local video file
    cap = cv2.VideoCapture(temp_file_path)

    # Frame pts_time to center the clip around
    target_pts_time = keyframe_pts_mapping.get(best_matching_keyframe)

    # Get the frames per second (fps) of the video
    fps = 25

    # Calculate the frame index corresponding to the target pts_time
    target_frame = int(target_pts_time * fps)

    # Calculate the frame indices for the start and end of the clip
    start_frame = int(target_frame - (clip_duration / 2) * fps)
    end_frame = int(target_frame + (clip_duration / 2) * fps)

    # Create a VideoWriter to save the extracted clip
    fourcc = cv2.VideoWriter_fourcc(*'h264')  # Use 'h264' codec for compatibility
    clip_writer = cv2.VideoWriter('best_matching_clip.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Extract and write frames within the specified time range
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if start_frame <= frame_number <= end_frame:
                clip_writer.write(frame)
            elif frame_number > end_frame:
                break
            frame_number += 1
        else:
            break

    cap.release()
    clip_writer.release()
    st.video('best_matching_clip.mp4')
    # Update progress bar for stage 3
    progress_stage.progress(100)
    progress_text.text("Done!")
    #st.text("Stage 3: Video extraction - Done")
if __name__ == "__main__":
    main()
