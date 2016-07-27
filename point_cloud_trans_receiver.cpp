#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
typedef pcl::PointCloud<pcl::PointXYZ> myPointCloud;
using namespace std;
using namespace pcl;
string path;
//ros::Publisher pub;
void callback(const myPointCloud::ConstPtr& temp_cloud)
{

	//myPointCloud::Ptr temp_cloud (new myPointCloud);
	myPointCloud::Ptr cloud (new myPointCloud);
	int j = 0,count = 0;
	//pcl::PCLPointCloud2 pcl_pc;
	//pcl_conversions::toPCL(*recent_cloud, pcl_pc);

        //pcl::fromPCLPointCloud2(pcl_pc, *temp_cloud);

	for(j=0; j<temp_cloud->points.size();j++)
	{
		if (temp_cloud->points[j].x > 0.4 && temp_cloud->points[j].x < 1.6
			&& temp_cloud->points[j].y > -0.7 && temp_cloud->points[j].y < 0.7 
			&& temp_cloud->points[j].z > 0.6 && temp_cloud->points[j].z < 1.5)
		{
			cloud->points.push_back(temp_cloud->points[j]);
			count++;
		}
	}
	printf("%d\n",cloud->points.size());
	printf("%d\n",count);
	cloud->width = count;
	cloud->height = 1;
	cloud->header.frame_id = "point_cloud_after_trans";
	cloud->header.stamp = ros::Time::now().toNSec();
        //pub.publish(*cloud);
        pcl::io::savePCDFileASCII(path.c_str(),*cloud);

}
int main(int argc, char** argv)
{

	path=argv[1];
        ros::init(argc, argv, "pointcloud_translator");
        ros::NodeHandle n;
	string topic = "/head_camera/depth_registered/points";
	//pub = n.advertise<myPointCloud> ("points_AT", 0);
        //sensor_msgs::PointCloud2::ConstPtr recent_cloud = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(topic, n, ros::Duration(0.0));
	ros::Subscriber sub = n.subscribe<myPointCloud>(topic, 10, callback);
	/*if(!recent_cloud)
        {
                ROS_ERROR("No point_cloud2 has been received");
                return -1;
        }
	*/

        //pcl::io::savePCDFileASCII(path.c_str(), *cloud);
	ros::spin();
        return 0;
}
