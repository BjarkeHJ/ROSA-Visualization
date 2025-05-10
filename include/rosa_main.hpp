#ifndef _ROSA_MAIN_H_
#define _ROSA_MAIN_H_

#include <datawrapper.hpp>
#include <Extra_Del.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/copy_point.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/features/normal_3d.h>

#include <pcl/filters/passthrough.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <string>
#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <math.h>
#include <time.h>
#include <deque>
#include <stack>

using namespace std;
using std::unique_ptr;
using std::shared_ptr;

namespace predrecon
{

    struct Vector3dCompare //OBS ADDED _1 
    {
        bool operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const {
            if (v1(0) != v2(0)) return v1(0) < v2(0);
            if (v1(1) != v2(1)) return v1(1) < v2(1);
            return v1(2) < v2(2);
        }
    };


    struct Vector3dHash_1 //OBS ADDED _1
    {
        size_t operator()(const Eigen::Vector3d& v) const {
            std::hash<double> hasher;
            size_t seed = 0;
            seed ^= hasher(v.x()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
            seed ^= hasher(v.y()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
            seed ^= hasher(v.z()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
            return seed;
        }
    };

    struct Pcloud
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pts_; // Downsampled (will be)
        pcl::PointCloud<pcl::PointXYZ>::Ptr ori_pts_; // Original points
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pts_; // Only if ground=true
        pcl::PointCloud<pcl::Normal>::Ptr normals_; // Downsampled normals
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals; // Original points with normals
        Eigen::MatrixXd pts_mat;
        Eigen::MatrixXd ar_pts_mat;
        Eigen::MatrixXd nrs_mat;
        vector<vector<int>> neighs; // Neighbours (idxs) to each point
        vector<vector<int>> surf_neighs; // updated neighbours after surf search
        vector<vector<int>> neighs_new; // Final updated neighbours

        double* datas; // Pointer member of struct Pcloud
        double* fastDatas;
        Eigen::MatrixXd skelver; // Skeleton vertices
        Eigen::MatrixXd corresp;
        Eigen::MatrixXi skeladj;
        Eigen::MatrixXd vertices; // Final vertices of skeleton
        Eigen::MatrixXi edges; // Final edges of skeleton
        Eigen::MatrixXi degrees;
        deque<int> joint;
        vector<list<int>> graph;
        vector<vector<int>> branches; // branches of skeleton
        vector<bool> visited;
        vector<bool> pts_distributed;
        vector<int> pts_segment_slave; // each input point belong to which segment.
        vector<double> pts_segment_sim; // the similarity of each input point and its segemnt belonging.
        vector<vector<int>> branch_seg_pairs; // the segments index set of each branch.
        map<int, vector<int>> segments; // e.g. map[0] = {id_0, id_1}, id_0 and id_1 are both indexer in vertices.
        pcl::PointCloud<pcl::PointXYZ>::Ptr cut_plane;
        Eigen::Vector3d cut_position;
        Eigen::Vector3d cut_vector;
        
        // SUB SPACE POINTS
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sub_space; // vector of points contained in each sub-space
        map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> seg_clouds; // points contained in each segment, [seg_id, points]
        
        // SUB SPACE POINTS SCALE RESTORED
        double scale;
        Eigen::Vector3d center;
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sub_space_scale; // Points in each branch sub-space (5)
        map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> seg_clouds_scale; // Points in each segment, [seg_id, points] (27)
        
        Eigen::MatrixXd vertices_scale;
        map<Eigen::Vector3d, int, Vector3dCompare> pt_seg_pair;
        map<int, double> inner_dist_set; // inner distance (avg) of each sub-segment-space (for viewpoints generation)
        
        // Real scene scale skeleton graph
        Eigen::MatrixXd realVertices;
        Eigen::MatrixXd outputVertices;
        Eigen::MatrixXi outputEdges;
        vector<vector<int>> startendBranches;
        vector<Eigen::Vector3d> mainDirBranches;
        vector<Eigen::Vector3d> centroidBranches; 
    };

    class ROSA_main {
    public:
        ROSA_main(){
        }
        ~ROSA_main(){
        }

        void init(pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud);

        void main();

        Pcloud P;

        bool visFlag;
        double groundHeight;

        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud_01;
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud_02;
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud_03;
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud_04;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds_restored;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        /* Func */
        void pcloud_read_off();
        double mahalanobis_leth(pcl::PointXYZ& p1, pcl::Normal& v1, pcl::PointXYZ& p2, pcl::Normal& v2, double& r);
        void pcloud_adj_matrix_mahalanobis(double& r_range);
        Eigen::Matrix3d create_orthonormal_frame(Eigen::Vector3d& v);
        void rosa_initialize(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::Normal>::Ptr& normals);
        void distance_query(DataWrapper& data, const vector<double>& Pp, const vector<double>& Np, double delta, vector<int>& isoncut);
        void pcloud_isoncut(Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut, vector<int>& isoncut, double*& datas, int& size);
        Eigen::MatrixXd rosa_compute_active_samples(int& idx, Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut);
        Eigen::Vector3d compute_symmetrynormal(Eigen::MatrixXd& local_normals);
        double symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals);
        Eigen::Vector3d symmnormal_smooth(Eigen::MatrixXd& V, Eigen::MatrixXd& w);
        Eigen::Vector3d closest_projection_point(Eigen::MatrixXd& P, Eigen::MatrixXd& V);
        void rosa_drosa();
        void rosa_dcrosa();
        void rosa_lineextract();
        void rosa_recenter();
        void graph_decomposition();
        void inner_decomposition();
        void branch_merge();
        void prune_branches();
        void restore_scale();
        void cal_inner_dist();
        void distribute_ori_cloud();
        void storeRealGraph();
        pcl::KdTreeFLANN<pcl::PointXYZ> rosa_tree;

        /* PARAMS */
        double prob_upper;
        double pt_downsample_voxel_size;
        double Radius, th_mah, delta, sample_radius, alpha_recenter, angle_upper, length_upper, length_lower, prune_lower;
        int numiter_drosa, numiter_dcrosa, k_KNN, ne_KNN;
        bool prune_flag;
        int ori_num;
        bool ground;
        int estNum;

        /* Data */
        string input_pcd, input_mesh;
        int pcd_size_;
        int seg_count;
        double norm_scale;
        Eigen::MatrixXd MAHADJ;
        Eigen::MatrixXd pset, dpset;
        Eigen::MatrixXd vset;
        Eigen::MatrixXd vvar;
        Eigen::Vector4d centroid;
        pcl::PointCloud<pcl::PointXYZ>::Ptr skeleton_ver_cloud;
        Eigen::MatrixXi adj_before_collapse;

         /* Utils */
        // shared_ptr<PlanningVisualization> vis_utils_;
        int argmax_eigen(Eigen::MatrixXd &x);
        void normalize();
        void normal_estimation();
        // void VisCallback(const ros::TimerEvent& e);
        void dfs(int& v);
        bool ocr_node(int& n, list<int>& candidates);
        vector<vector<int>> divide_branch(vector<int>& input_branch);
        vector<int> merge_branch(vector<int>& input_branch);
        bool prune(vector<int>& input_branch);
        double distance_point_line(Eigen::Vector3d& point, Eigen::Vector3d& line_pt, Eigen::Vector3d& line_dir);
        Eigen::Vector3d PCA(Eigen::MatrixXd& A);
        
        /* Timer */
        // ros::Timer vis_timer_;
    };

}

#endif