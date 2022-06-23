# from google.cloud import storage

# def download_blob(bucket_name, 
#                   source_blob_name, 
#                   destination_file_name):
#     """Downloads a blob from the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#     print(
#         "Blob {} downloaded to {}.".format(
#             source_blob_name, destination_file_name
#         )
#     )
# download_blob('objectron', 'v1/index/cup_annotation',  './cup_annotation')


import requests
public_url = "https://storage.googleapis.com/objectron"
blob_path = public_url + "/v1/index/cup_annotations_test"
video_ids = requests.get(blob_path).text
video_ids = video_ids.split('\n')
# Download the first ten videos in cup test dataset
for i in range(1):
    video_filename = public_url + "/videos/" + video_ids[i] + "/video.MOV"
    metadata_filename = public_url + "/videos/" + video_ids[i] + "/geometry.pbdata"
    annotation_filename = public_url + "/annotations/" + video_ids[i] + ".pbdata"
    # video.content contains the video file.
    video = requests.get(video_filename)
    metadata = requests.get(metadata_filename)
    annotation = requests.get(annotation_filename)
    file = open("example.MOV", "wb")
    file.write(video.content)
    file.close()