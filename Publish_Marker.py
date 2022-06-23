import rospy
from visualization_msgs.msg import Marker

def talker():
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.get_time()
    marker.ns = "my_namespace"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = 1
    marker.pose.position.y = 1
    marker.pose.position.z = 1
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0


    pub = rospy.Publisher('pub_marker', Marker, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    # rate = rospy.Rate(10)  # 10hz
    # while not rospy.is_shutdown():
    # hello_str = "hello world %s" % rospy.get_time()
    # rospy.loginfo(hello_str)
    pub.publish(marker)
    # rate.sleep()

if __name__ == '__main__':
    talker()