import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# dir = "20200727_195335.bag"

def main():
    # if not os.path.exists(args.directory):
    #     os.mkdir(args.directory)
    index = 0
    try:
        config = rs.config()
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        pipeline = rs.pipeline()
        rs.config.enable_device_from_file(config, "bag/20200917_172950.bag")
        # rs.config.enable_device_from_file(config, "bag/20200811_141534.bag")
        # rs.config.enable_device_from_file(config, "bag/20200727_195335.bag")
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # print("scale : " + str(depth_scale))
        # compare_img = cv2.imread("/home/user/real_scene_color2.png")

        profile = pipeline.get_active_profile()

        color_stream = profile.get_stream(rs.stream.color)
        color_profile = rs.video_stream_profile(color_stream)
        color_intrinsics = color_profile.get_intrinsics()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()

        print("*** color intrinsics ***")
        print(color_intrinsics)

        print("*** depth intrinsics ***")
        print(depth_intrinsics)

        img = cv2.imread("/home/user/sample_image/image0_depth.png", cv2.IMREAD_GRAYSCALE)

        while True:
            # print("frame:", index)
            # print("scale : " + str(depth_scale))
            frames = pipeline.wait_for_frames()
            # print("frames type : " + str(type(frames)))
            # align the deph to color frame
            aligned_frames = align.process(frames)
            # get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            # scaled_depth_image = depth_image * depth_scale
            aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
            # convert color image to BGR for OpenCV
            r, g, b = cv2.split(aligned_color_image)
            aligned_color_image = cv2.merge((b, g, r))

            # depth_intrinsics = rs.video_stream_profile(
            #     depth_image.profile).get_intrinsics()
            #
            # print(depth_intrinsics)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            r, g, b = cv2.split(color_image)
            color_image = cv2.merge((b, g, r))

            # print("frame_image type : " + str(type(color_image)))
            # print("frame_image shape : " + str(color_image.shape))
            # print("depth_image type : " + str(type(depth_image)))
            # print("depth_image shape : " + str(depth_image.shape))
            # depth_image *= 10

            # print(depth_image)
            # print(scaled_depth_image)
            # print(compare_img)
            # print(color_image)

            # row, col = depth_image.shape
            # flag = True
            #
            # for i in range(row):
            #     if flag is False:
            #         break
            #     for j in range(col):
            #         if flag is False:
            #             break
            #         if img[i][j] != depth_image[i][j]:
            #             flag = False
            #             break
            # if flag is True:
            #     print("index : " + str(index))
            #     exit()


            # if compare_img is color_image:
            #     print("index : " + str(i))
            #     break
                # print("true")
            # else:
                # print("no")
            print(aligned_depth_image)
            # print(aligned_depth_image.shape)
            # print(min(aligned_depth_image[:,:,0]))

            cv2.imshow("color", color_image)
            # cv2.imshow("depth", depth_image)
            cv2.imshow("aligned_depth_image", aligned_depth_image)
            # cv2.imshow("aligned_color_image", aligned_color_image)
            cv2.waitKey(0)

            # print(scaled_depth_image)
            # scaled_depth_image *= 1000
            # scaled_depth_image = scaled_depth_image.astype(np.uint64)
            # print(scaled_depth_image)

            # cv2.imwrite("/home/user/sample_image/image0_depth_raw.png", depth_image)
            # cv2.imwrite("/home/user/bag_images2/depth_image" + str(index) + ".png", depth_image)
            # cv2.imwrite("/home/user/bag_images2/color_image" + str(index) + ".png", color_image)
            # break
            # if index % 10 == 0:
            # cv2.imwrite("/home/user/NOCS_test_imgs/depth_image" + str(index) + ".png", aligned_depth_image)
            # cv2.imwrite("/home/user/NOCS_test_imgs/color_image" + str(index) + ".png", color_image)
            print("index : " + str(index))

            if index == 100:
                break
            index += 1
    finally:
        pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    # parser.add_argument("-i", "--input", type=str, help="Bag file to read")
    # args = parser.parse_args()

    main()