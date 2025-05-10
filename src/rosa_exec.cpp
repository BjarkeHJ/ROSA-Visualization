#include <rosa_main.hpp>
#include <string>

using namespace predrecon;

pcl::PointCloud<pcl::PointXYZ>::Ptr load_pcd_pts(const std::string &pcd_path);
void save_pcd_pts(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pts, const std::string &save_path);
void save_pcd_pts_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pts_nrms, const std::string &save_path);

/* Input and Output Paths... */
// std::string pcd_path = "../data/cloud.pcd";
std::string pcd_path = "../data/windmill.pcd";
// std::string pcd_path = "../data/05_horizontal_wing_side.pcd";
// std::string pcd_path = "../data/08_nacelle_side.pcd";
// std::string pcd_path = "../data/09_wings_only_front.pcd";

std::string save_path_01 = "../vis_tools/data/output_01.pcd";
std::string save_path_02 = "../vis_tools/data/output_02.pcd";
std::string save_path_03 = "../vis_tools/data/output_03.pcd";
std::string save_path_04 = "../vis_tools/data/output_04.pcd";
std::string save_ds_cloud_path = "../vis_tools/data/input_ds.pcd";

int main() {
    /* Load .pcd file */
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    input_cloud = load_pcd_pts(pcd_path);

    /* ROSA Algorithm */
    std::shared_ptr<ROSA_main> skel_op;
    skel_op.reset(new ROSA_main);
    skel_op->init(input_cloud);
    skel_op->main();

    /* Save Output */
    save_pcd_pts(skel_op->P.pts_, save_ds_cloud_path);
    save_pcd_pts(skel_op->output_cloud_01, save_path_01);
    save_pcd_pts(skel_op->output_cloud_02, save_path_02);
    save_pcd_pts(skel_op->output_cloud_03, save_path_03);
    save_pcd_pts(skel_op->output_cloud_04, save_path_04);
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
