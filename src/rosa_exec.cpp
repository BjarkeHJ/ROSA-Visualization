#include <rosa_main.hpp>
#include <string>

pcl::PointCloud<pcl::PointXYZ>::Ptr load_pcd_pts(const std::string &pcd_path);
void save_pcd_pts(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pts, const std::string &save_path);
void save_pcd_pts_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pts_nrms, const std::string &save_path);

/* Input and Output Paths... */
// std::string pcd_path = "../data/cloud.pcd";
std::string pcd_path = "../data/05_horizontal_wing_side.pcd";
std::string save_path = "../vis_tools/data/output.pcd";

int main() {
    /* Load .pcd file */
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    input_cloud = load_pcd_pts(pcd_path);

    /* ROSA Algorithm */
    std::shared_ptr<RosaPoints> skel_op;
    skel_op.reset(new RosaPoints);
    skel_op->init();
    skel_op->rosa_main(input_cloud);

    /* Save Output */
    save_pcd_pts(skel_op->skeleton_ver_cloud, save_path);
    return 0;
}



/* Loading and Saving PointClouds*/
pcl::PointCloud<pcl::PointXYZ>::Ptr load_pcd_pts(const std::string &pcd_path) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_path, *cloud) == -1) {
        PCL_ERROR ("Could not read PointCloud %s\n", pcd_path.c_str());
        return nullptr;
    }
    PCL_INFO("Loaded PointCloud... Size: %lu\n", cloud->points.size());
    return cloud;
}

void save_pcd_pts(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pts, const std::string &save_path) {
    if (cloud_pts->points.empty()) {
        std::cout << "Error: PointCloud Empty" << std::endl;
        return;
    }
    cloud_pts->height = 1;
    cloud_pts->width = cloud_pts->points.size();
    pcl::PCDWriter writer;
    if (writer.writeASCII(save_path, *cloud_pts, 8) == -1) {
        std::cout << "Error: Could not save PointCloud to: " << save_path << std::endl;
        return;
    }
    else {
        std::cout << "Saved PointCloud... Size: " << cloud_pts->points.size() << std::endl;
    }
}

void save_pcd_pts_normals(pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pts_nrms, const std::string &save_path) {
    if (cloud_pts_nrms->points.empty()) {
        std::cout << "Error: PointCloud Empty" << std::endl;
        return;
    }
    cloud_pts_nrms->height = 1;
    cloud_pts_nrms->width = cloud_pts_nrms->points.size();
    pcl::PCDWriter writer;
    if (writer.writeASCII(save_path, *cloud_pts_nrms, 8) == -1) {
        std::cout << "Error: Could not save PointCloud to: " << save_path << std::endl;
        return;
    }
    else {
        std::cout << "Saved PointCloud... Size: " << cloud_pts_nrms->points.size() << std::endl;
    }
}