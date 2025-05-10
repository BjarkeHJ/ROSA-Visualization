#include <rosa_main.hpp>

void RosaPoints::init() {
    RC.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    RC.normals_.reset(new pcl::PointCloud<pcl::Normal>);
    RC.cloud_w_normals.reset(new pcl::PointCloud<pcl::PointNormal>);

    vis_curr_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    vis_rosa_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    skeleton_ver_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    pcd_size_ = 0;
}

void RosaPoints::rosa_main(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    if (cloud->points.empty()) {
        std::cout << "Recieved Cloud is Empty..." << std::endl;
        return;
    }

    set_cloud(cloud);
    normalize();

    std::cout << "Downsampled PointCloud size: " << pcd_size_ << std::endl;

    pset.resize(pcd_size_, 3);
    vset.resize(pcd_size_, 3);
    vvar.resize(pcd_size_, 1);

    // Insert normalized points in ordered structure
    RC.datas = new double[pcd_size_ * 3]();
    for (int idx=0; idx<pcd_size_; idx++){
        RC.datas[idx] = RC.pts_->points[idx].x; 
        RC.datas[idx+pcd_size_] = RC.pts_->points[idx].y;
        RC.datas[idx+2*pcd_size_] = RC.pts_->points[idx].z;
    }

    adj_matrix(r_range);
    rosa_drosa();
    rosa_dcrosa();
    rosa_lineextract();
}


/* Main Components */
void RosaPoints::set_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    RC.pts_->clear();
    RC.normals_->clear();
    RC.pts_ = cloud;
    pcd_size_ = RC.pts_->points.size();
    // RC.pts_mat.resize(pcd_size_, 3);
    // RC.nrs_mat.resize(pcd_size_, 3);
    // for (int i=0; i<pcd_size_; i++) {
    //     RC.pts_mat(i,0) = RC.pts_->points[i].x;
    //     RC.pts_mat(i,1) = RC.pts_->points[i].y;
    //     RC.pts_mat(i,2) = RC.pts_->points[i].z;
    // }
}

void RosaPoints::normal_estimation() {
    if (!RC.pts_->empty()) {
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(RC.pts_);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);

        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setKSearch(ne_KNN);
        ne.compute(*cloud_normals);

        RC.normals_ = cloud_normals;
    }
    else std::cout << "ROSA Warning: Normal Estimation - Point Cloud empty..." << std::endl;
}

void RosaPoints::normalize() {
    pcl::PointXYZ min, max;
    pcl::getMinMax3D(*RC.pts_, min, max);
    
    float x_scale, y_scale, z_scale, max_scale;
    x_scale = max.x - min.x;
    y_scale = max.y - min.y;
    z_scale = max.z - min.z;

    if (x_scale >= y_scale) {
        max_scale = x_scale;
    }
    else max_scale = y_scale;
    if (max_scale < z_scale) {
        max_scale = z_scale;
    }
    norm_scale = max_scale;

    // Normalize point cloud
    pcl::compute3DCentroid(*RC.pts_, centroid);
    for (int i=0; i<pcd_size_; i++) {
        RC.pts_->points[i].x = (RC.pts_->points[i].x - centroid(0)) / norm_scale;
        RC.pts_->points[i].y = (RC.pts_->points[i].y - centroid(1)) / norm_scale;
        RC.pts_->points[i].z = (RC.pts_->points[i].z - centroid(2)) / norm_scale;
    }

    // Estimate surface normals
    normal_estimation();

    RC.cloud_w_normals.reset(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*RC.pts_, *RC.normals_, *RC.cloud_w_normals);

    // Downsampling 
    pcl::VoxelGrid<pcl::PointNormal> vgf;
    int iter = 0;
    ds_leaf_size = 0.001;
    while (pcd_size_ > max_points) {
        vgf.setInputCloud(RC.cloud_w_normals);
        vgf.setLeafSize(ds_leaf_size, ds_leaf_size, ds_leaf_size);
        vgf.filter(*RC.cloud_w_normals);
        pcd_size_ = RC.cloud_w_normals->points.size(); // update cloud size
        ds_leaf_size += 0.005;
        iter++;

    }

    // Reset variable for to accomodate normalized points...
    RC.pts_->clear();
    RC.normals_->clear();
    RC.pts_mat.resize(pcd_size_, 3);
    RC.nrs_mat.resize(pcd_size_, 3);

    pcl::Normal normal;
    pcl::PointXYZ pt;
    for (int i=0; i<pcd_size_; ++i) {
        pt.x = RC.cloud_w_normals->points[i].x;
        pt.y = RC.cloud_w_normals->points[i].y;
        pt.z = RC.cloud_w_normals->points[i].z;
        normal.normal_x = -RC.cloud_w_normals->points[i].normal_x; 
        normal.normal_y = -RC.cloud_w_normals->points[i].normal_y; 
        normal.normal_z = -RC.cloud_w_normals->points[i].normal_z;
        RC.pts_->push_back(pt);
        RC.normals_->points.push_back(normal);
        RC.pts_mat(i,0) = RC.pts_->points[i].x;
        RC.pts_mat(i,1) = RC.pts_->points[i].y;
        RC.pts_mat(i,2) = RC.pts_->points[i].z;
        RC.nrs_mat(i,0) = RC.normals_->points[i].normal_x;
        RC.nrs_mat(i,1) = RC.normals_->points[i].normal_y;
        RC.nrs_mat(i,2) = RC.normals_->points[i].normal_z;
    }
}

float RosaPoints::pt_similarity_metric(pcl::PointXYZ &p1, pcl::Normal &v1, pcl::PointXYZ &p2, pcl::Normal &v2, float &range_r) {
    float Fs = 2.0;
    float k = 0.0;
    float dist, vec_dot, w;
    Eigen::Vector3d p1_, p2_, v1_, v2_;
    p1_ << p1.x, p1.y, p1.z;
    p2_ << p2.x, p2.y, p2.z;
    v1_ << v1.normal_x, v1.normal_y, v1.normal_z;
    v2_ << v2.normal_x, v2.normal_y, v2.normal_z;

    // distance similarity metric
    dist = (p1_ - p2_ + Fs*((p1_ - p2_).dot(v1_))*v1_).norm();
    dist = dist/range_r;

    if (dist <= 1) {
        k = 2*pow(dist,3) - 3*pow(dist,2) + 1;
    }

    vec_dot = v1_.dot(v2_);
    w = k*pow(std::max(0.0f, vec_dot), 2);

    return w;
}

void RosaPoints::adj_matrix(float &range_r) {
    if (RC.pts_->empty()) {
        return;
    }

    RC.neighs.clear();
    RC.neighs.resize(pcd_size_);
    
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(RC.pts_); // Normalized points
    
    pcl::PointXYZ search_pt, p1, p2;
    pcl::Normal v1, v2;
    std::vector<int> indxs;
    std::vector<float> radius_squared_distance;
    float w1, w2, w;
    std::vector<std::vector<int>> pt_neighs_idx;

    for (int i=0; i<pcd_size_; i++) {
        // Do radius search for each point in current cloud ... 
        std::vector<int>().swap(indxs);
        std::vector<float>().swap(radius_squared_distance);
        p1 = RC.pts_->points[i];
        v1 = RC.normals_->points[i];
        search_pt = RC.pts_->points[i];
        tree.radiusSearch(search_pt, range_r, indxs, radius_squared_distance);
        std::vector<int> temp_neighs;

        for (int j=0; j<(int)indxs.size(); j++) {
            // For each neighbour compute the distance metric ...
            p2 = RC.pts_->points[indxs[j]];
            v2 = RC.normals_->points[indxs[j]];
            w1 = pt_similarity_metric(p1, v1, p2, v2, range_r);
            w2 = pt_similarity_metric(p2, v2, p1, v1, range_r);
            w = std::min(w1, w2);
            
            // Check if valid neighbour
            if (w > th_mah) {
                temp_neighs.push_back(indxs[j]);
            }
        }
        RC.neighs[i] = temp_neighs;
    }
}

void RosaPoints::rosa_drosa() {

    Extra_Del ed_; // Matrix helper functions

    rosa_initialize(RC.pts_, RC.normals_); // Initialized with normalized data (sets pset and vset)

    std::vector<std::vector<int>>().swap(RC.surf_neighs);
    pcl::KdTreeFLANN<pcl::PointXYZ> surf_kdtree;
    surf_kdtree.setInputCloud(RC.pts_);

    std::vector<int> temp_surf(k_KNN);
    std::vector<float> nn_squared_distance(k_KNN);
    pcl::PointXYZ search_point_surf;

    for (int i=0; i<pcd_size_; i++) {
        std::vector<int>().swap(temp_surf);
        std::vector<float>().swap(nn_squared_distance);
        search_point_surf = RC.pts_->points[i];
        surf_kdtree.nearestKSearch(search_point_surf, k_KNN, temp_surf, nn_squared_distance);
        RC.surf_neighs.push_back(temp_surf); // Store surface neighbours of each point... RC.surf_neighs[i] = k_KNN nearest points
    }

    Eigen::Vector3d var_p, var_v, new_v;
    Eigen::MatrixXd indxs, extract_normals;

    for (int n=0; n<num_drosa_iter; n++) {
        Eigen::MatrixXd vnew = Eigen::MatrixXd::Zero(pcd_size_, 3);
        for (int pidx=0; pidx<pcd_size_; pidx++) {
            var_p = pset.row(pidx); // current point
            var_v = vset.row(pidx); // vector orthogonal to surface normal
            indxs = compute_active_samples(pidx, var_p, var_v); // indices in plane slice 
            extract_normals = ed_.rows_ext_M(indxs, RC.nrs_mat);
            vnew.row(pidx) = compute_symmetrynormal(extract_normals).transpose(); // Plane slice normal (local normal variance minimization)
            new_v = vnew.row(pidx);

            if (extract_normals.rows() > 0) {
                vvar(pidx, 0) = symmnormal_variance(new_v, extract_normals); // Variance of projections of local normals on found symmetry normal
            }
            else {
                vvar(pidx,0) = 0.0;
            }
        }

        Eigen::MatrixXd offset(vvar.rows(), vvar.cols());
        offset.setOnes();
        offset = 0.00001 * offset;
        vvar = (vvar.cwiseAbs2() + offset).cwiseInverse(); // Ensure no exact zero-value and do coefficient wise inversion (reciprocal)
        vset = vnew; // Set found symmetry normals for next iteration (first iter: in stead of orthonormal frame vector)

        // At this point: For each point in the cloud a symmetry normal vector has been estimated.
        // Now smoothing of each symmetry normal vector is done by comparing with neighbouring symmetry normals
        // This is done by a weighted least squares of the neighbouring symnorms (weighted by the variance (inverse/reciprocal) computed earlier)

        /* Smoothing */
        std::vector<int> surf_;
        Eigen::MatrixXi snidxs;
        Eigen::MatrixXd snidxs_d, vset_ex, vvar_ex;

        for (int i=0; i<1; i++) {
            for (int p=0; p<pcd_size_; p++) {
                std::vector<int>().swap(surf_);
                surf_ = RC.surf_neighs[p]; // The initial neighbours to point p
                snidxs.resize(surf_.size(), 1);
                snidxs = Eigen::Map<Eigen::MatrixXi>(surf_.data(), surf_.size(), 1);
                snidxs_d = snidxs.cast<double>();

                vset_ex = ed_.rows_ext_M(snidxs_d, vset); // Extract symmetry normals for indices of the initial neighbours
                vvar_ex = ed_.rows_ext_M(snidxs_d, vvar); // Extract variances (reciprocal) ...
                vset.row(p) = symmnormal_smooth(vset_ex, vvar_ex);
            }
            vnew = vset;
        }
    } // Loop next rosa point orientation iteration

    /* --- compute positions of ROSA --- */
    std::vector<int> poorIdx;
    pcl::PointCloud<pcl::PointXYZ>::Ptr goodPts (new pcl::PointCloud<pcl::PointXYZ>);
    std::map<Eigen::Vector3d, Eigen::Vector3d, Vector3dCompare> goodPtsPset; // This could be used to accumulate points in a real-time application?
    Eigen::Vector3d var_p_p, var_v_p, centroid;
    Eigen::MatrixXd indxs_p, extract_pts, extract_nrs;

    for (int pIdx=0; pIdx<pcd_size_; pIdx++) {
        var_p_p = pset.row(pIdx); 
        var_v_p = vset.row(pIdx); // Symmetry normal of point at pIdx
        indxs_p = compute_active_samples(pIdx, var_p_p, var_v_p); // Points in plane-slice after iteration-optimization of symmetry normals. 

        //* Update Neighbors
        std::vector<int> temp_neigh;
        for (int p=0; p<(int)indxs_p.rows(); p++) {
            temp_neigh.push_back(indxs_p(p,0));
        }

        RC.neighs_new.push_back(temp_neigh); // RC.neighs_new now contains the the "active samples" corresponding to each point

        extract_pts = ed_.rows_ext_M(indxs_p, RC.pts_mat); //Points in plane-slice
        extract_nrs = ed_.rows_ext_M(indxs_p, RC.nrs_mat); //Surface normals of these points
        centroid = closest_projection_point(extract_pts, extract_nrs); // Computes the best fit projection point of local points/normals
        
        // The centroid of the active points should not be outside the max scale (dimensions) of the normalized cloud
        if (abs(centroid(0)) < 1 && abs(centroid(1)) < 1 && abs(centroid(2)) < 1) {
            pset.row(pIdx) = centroid; // sets the points to the centroid (Rosa Point)

            pcl::PointXYZ goodPoint;
            Eigen::Vector3d goodPointP;
            goodPoint = RC.pts_->points[pIdx]; // Points that do not yield a rosa point (centroid) outside the allowed dimensions
            goodPointP(0) = RC.pts_->points[pIdx].x;
            goodPointP(1) = RC.pts_->points[pIdx].y;
            goodPointP(2) = RC.pts_->points[pIdx].z;
            goodPts->points.push_back(goodPoint);
            goodPtsPset[goodPointP] = centroid;
        }
        else {
            poorIdx.push_back(pIdx);
        }
    }

    rosa_tree.setInputCloud(goodPts);
    
    for (int pp=0; pp<(int)poorIdx.size(); pp++) {
        int pair = 1;
        pcl::PointXYZ search_point;
        search_point.x = RC.pts_->points[poorIdx[pp]].x; 
        search_point.y = RC.pts_->points[poorIdx[pp]].y; 
        search_point.z = RC.pts_->points[poorIdx[pp]].z;
        
        std::vector<int> pair_id(pair);
        std::vector<float> nn_squared_distance(pair);
        rosa_tree.nearestKSearch(search_point, pair, pair_id, nn_squared_distance);
        
        Eigen::Vector3d pairpos;
        pairpos(0) = goodPts->points[pair_id[0]].x;
        pairpos(1) = goodPts->points[pair_id[0]].y;
        pairpos(2) = goodPts->points[pair_id[0]].z;
        Eigen::Vector3d goodrp = goodPtsPset.find(pairpos)->second;
        pset.row(poorIdx[pp]) = goodrp;
    }
    dpset = pset;
}

void RosaPoints::rosa_dcrosa() {
    Extra_Del ed_dc;
    
    for (int n=0; n<num_dcrosa_iter; n++) {
        Eigen::MatrixXi int_nidxs;
        Eigen::MatrixXd newpset, indxs, extract_neighs;
        newpset.resize(pcd_size_, 3);

        for (int i=0; i<pcd_size_; i++) {
            if (RC.neighs[i].size() > 0) { // If curret point has any neighbours
                int_nidxs = Eigen::Map<Eigen::MatrixXi>(RC.neighs[i].data(), RC.neighs[i].size(), 1); // Creates an Eigen column vector of the neighbours
                indxs = int_nidxs.cast<double>();
                extract_neighs = ed_dc.rows_ext_M(indxs, pset); //extract ROSA points of neighbouring points 
                newpset.row(i) = extract_neighs.colwise().mean(); // Sets the mean of the neighbouring points as a single ROSA Point
            }
            else {
                newpset.row(i) = pset.row(i); // If no neighbours
            }
        }
        pset = newpset; //overwrite pset with the averaged neighbourhood ROSA points

        /* Shrinking */
        pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pset_cloud->width = pset.rows();
        pset_cloud->height = 1;
        pset_cloud->points.resize(pset_cloud->width * pset_cloud->height);
        
        for (size_t i=0; i<pset_cloud->points.size(); i++) {
            pset_cloud->points[i].x = pset(i,0);
            pset_cloud->points[i].y = pset(i,1);
            pset_cloud->points[i].z = pset(i,2);
        }

        pcl::KdTreeFLANN<pcl::PointXYZ> pset_tree;
        pset_tree.setInputCloud(pset_cloud);

        // Calculate confidence
        Eigen::VectorXd conf = Eigen::VectorXd::Zero(pset.rows());
        Eigen::MatrixXd newpset2;
        newpset2.resize(pcd_size_, 3);
        newpset2 = pset;
        double CONFIDENCE_TH = 0.5;

        for (int i=0; i<(int)pset.rows(); i++) {
            std::vector<int> pointIdxNKNSearch(k_KNN);
            std::vector<float> pointNKNSquaredDistance(k_KNN);
            // K nearest neigbours of each ROSA point in pset / pset_cloud.
            pset_tree.nearestKSearch(pset_cloud->points[i], k_KNN, pointIdxNKNSearch, pointNKNSquaredDistance);

            Eigen::MatrixXd neighbours(k_KNN, 3);
            for (int j=0; j<k_KNN; j++) {
                neighbours.row(j) = pset.row(pointIdxNKNSearch[j]); //pointIdxNKNSearch contains the k_KNN nearest points.
            }

            Eigen::Vector3d local_mean = neighbours.colwise().mean(); // average of the neighbouring ROSA points
            neighbours.rowwise() -= local_mean.transpose(); 

            // The confidence metric is based on the fact that the singular-values represent the variance in the respective 
            // principal direction. If the largest singular value (0) is large relative to the sum, it indicates that
            // the neighbours are highly linear in nature: i.e. skeletonized.
            Eigen::BDCSVD<Eigen::MatrixXd> svd(neighbours, Eigen::ComputeThinU | Eigen::ComputeThinV);
            conf(i) = svd.singularValues()(0) / svd.singularValues().sum();

            // Compute linear projection
            // if the neighbouring ROSA points are not linear enough, a linear projection is performed:
            // The direction with least variance (dominant singular vector) is the one with the largest singular value (0)
            // The points are then projected onto the dominant direction in the neighbourhood
            if (conf(i) < CONFIDENCE_TH) continue;
            newpset2.row(i) = svd.matrixU().col(0).transpose() * (svd.matrixU().col(0) * (pset.row(i) - local_mean.transpose())) + local_mean.transpose();
        }
        pset = newpset2; // Overwrite
    }
}

void RosaPoints::rosa_lineextract() {

    Extra_Del ed_le;
    int outlier = 2;

    pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ pset_pt;

    Eigen::MatrixXi bad_sample = Eigen::MatrixXi::Zero(pcd_size_, 1); // Bad samples zero initialized
    std::vector<int> indxs;
    std::vector<float> radius_squared_distance;
    pcl::PointXYZ search_point;
    Eigen::MatrixXi int_nidxs;
    Eigen::MatrixXd nIdxs, extract_corresp;

    for (int i=0; i<pcd_size_; i++) {
        if ((int)RC.neighs[i].size() <= outlier) {
            // if the point has less than or equal to two neighbours it is classified as a bad_sample
            bad_sample(i,0) = 1; // flagged with 1
        }
    }

    for (int j=0; j<pset.rows(); j++) {
        pset_pt.x = pset(j,0);
        pset_pt.y = pset(j,1);
        pset_pt.z = pset(j,2);
        pset_cloud->points.push_back(pset_pt);
    }

    if (pset_cloud->empty()) return;

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(pset_cloud);
    RC.skelver.resize(0,3);
    Eigen::MatrixXd mindst = Eigen::MatrixXd::Constant(pcd_size_, 1, std::numeric_limits<double>::quiet_NaN()); // "min distance" initialized with quiet_NaN
    RC.corresp = Eigen::MatrixXd::Constant(pcd_size_, 1, -1); // initialized with value -1

    // Farthest Point Sampling (FPS) / Skeletonization
    for (int k=0; k<pcd_size_; k++) {
        if (RC.corresp(k,0) != -1) continue; // if index is not equal to -1 -> continue

        mindst(k,0) = 1e8; //set mindst(k) to a very large distance

        // run while ANY element in corresp is still -1
        while (!((RC.corresp.array() != -1).all())) {

            int maxIdx = argmax_eigen(mindst); // maxIdx represents the most distant unassigned point

            if (mindst(maxIdx,0) == 0) break; // If that value is 0 break
            if (!std::isnan(mindst(maxIdx, 0)) && mindst(maxIdx,0) == 0) break; // if not a NaN and value is 0 break

            std::vector<int>().swap(indxs);
            std::vector<float>().swap(radius_squared_distance);
            search_point.x = pset(maxIdx,0);
            search_point.y = pset(maxIdx,1);
            search_point.z = pset(maxIdx,2);

            // Search for points within the sample_radius of the current search point. 
            // The indices of the nearest points are set in indxs
            tree.radiusSearch(search_point, sample_radius, indxs, radius_squared_distance);

            int_nidxs = Eigen::Map<Eigen::MatrixXi>(indxs.data(), indxs.size(), 1); // Maps the indices of the nearest points. 
            nIdxs = int_nidxs.cast<double>();
            extract_corresp = ed_le.rows_ext_M(nIdxs, RC.corresp); // Extract the section RC.corresp according to the indices of the nearest points

            // if these are all different from -1 set the mindst value at maxIdx to 0...
            // If all neighbours wihtin sample_radius already has been assigned
            if ((extract_corresp.array() != -1).all()) {
                mindst(maxIdx,0) = 0;
                continue; // go to beginning of while loop
            }


            RC.skelver.conservativeResize(RC.skelver.rows()+1, RC.skelver.cols()); // adds one row - add one vertice point to the skeleton
            RC.skelver.row(RC.skelver.rows()-1) = pset.row(maxIdx); // set new vertice to the point at the farthest distance within the sample_radius

            // for every point index withing the sample_radius
            for (int z=0; z<(int)indxs.size(); z++) {
                // if the distance value at this index is NaN (unassigned) OR larger than the squared distance to the point
                if (std::isnan(mindst(indxs[z],0)) || mindst(indxs[z],0) > radius_squared_distance[z]) {
                    mindst(indxs[z],0) = radius_squared_distance[z]; // set distance value to the squared distance
                    RC.corresp(indxs[z], 0) = RC.skelver.rows() - 1; // sets the corresp value to the number of rows of skelver minus one
                    // the above line will act as a sorting algorithm for connected points. It will assign 0, 1, 2,..., pcd_size_
                }
            }
        }
    }

    skeleton_ver_cloud->clear();
    pcl::PointXYZ pt_vertex;

    for (int r=0; r<(int)RC.skelver.rows(); r++) {
        pt_vertex.x = RC.skelver(r,0);
        pt_vertex.y = RC.skelver(r,1);
        pt_vertex.z = RC.skelver(r,2);
        skeleton_ver_cloud->points.push_back(pt_vertex);
    }


    // int dim = RC.skelver.rows(); // number of vertices - should be current number of vertices perhaps?
    // Eigen::MatrixXi Adj;
    // Adj = Eigen::MatrixXi::Zero(dim, dim);
    // std::vector<int> temp_surf(k_KNN);
    // std::vector<int> good_neighs;

    // for (int pIdx=0; pIdx<pcd_size_; pIdx++) {
    //     temp_surf.clear();
    //     good_neighs.clear();
    //     temp_surf = RC.surf_neighs[pIdx];

    //     for (int ne=0; ne<(int)temp_surf.size(); ne++) {
    //         if (bad_sample(temp_surf[ne], 0) == 0) {
    //             good_neighs.push_back(temp_surf[ne]);
    //         }
    //     }

    //     if (RC.corresp(pIdx,0) == -1) continue;

    //     for (int nidx=0; nidx<(int)good_neighs.size(); nidx++) {
    //         if (RC.corresp(good_neighs[nidx],0) == -1) continue;
    //         Adj((int)RC.corresp(pIdx,0), (int)RC.corresp(good_neighs[nidx],0)) = 1;
    //         Adj((int)RC.corresp(good_neighs[nidx],0), (int)RC.corresp(pIdx,0)) = 1;
    //     }
    // }

    // adj_before_collapse.resize(Adj.rows(), Adj.cols());
    // adj_before_collapse = Adj;

    // /* Edge collapse */
    // std::vector<int> ec_neighs;
    // Eigen::MatrixXd edge_rows;
    // edge_rows.resize(2,3);
    // while (1) {
    //     int tricount = 0;
    //     Eigen::MatrixXi skeds;
    //     skeds.resize(0,2);
    //     Eigen::MatrixXd skcst;
    //     skcst.resize(0,1);

    //     for (int i=0; i<RC.skelver.rows(); i++) {
    //         ec_neighs.clear();
    //         for (int col=0; col<Adj.cols(); ++col) {
    //             if (Adj(i,col) == 1 && col>i) {
    //                 ec_neighs.push_back(col);
    //             }
    //         }
    //         std::sort(ec_neighs.begin(), ec_neighs.end());

    //         for (int j=0; j<(int)ec_neighs.size(); j++) {
    //             for (int k=j+1; k<(int)ec_neighs.size(); k++) {
    //                 if (Adj(ec_neighs[j], ec_neighs[k]) == 1) {
    //                     tricount++;
    //                     skeds.conservativeResize(skeds.rows()+1, skeds.cols()); // add one row
    //                     skeds(skeds.rows()-1, 0) = i;
    //                     skeds(skeds.rows()-1, 1) = ec_neighs[j];

    //                     skcst.conservativeResize(skcst.rows()+1, skcst.cols()); 
    //                     skcst(skcst.rows()-1, 0) = (RC.skelver.row(i) - RC.skelver.row(ec_neighs[j])).norm();

    //                     skeds.conservativeResize(skeds.rows()+1, skeds.cols());
    //                     skeds(skeds.rows()-1, 0) = ec_neighs[j];
    //                     skeds(skeds.rows()-1, 1) = ec_neighs[k];

    //                     skcst.conservativeResize(skcst.rows()+1, skcst.cols());
    //                     skcst(skcst.rows()-1, 0) = (RC.skelver.row(ec_neighs[i]) - RC.skelver.row(ec_neighs[k])).norm();

    //                     skeds.conservativeResize(skeds.rows()+1, skeds.cols());
    //                     skeds(skeds.rows()-1, 0) = ec_neighs[k];
    //                     skeds(skeds.rows()-1, 1) = i;

    //                     skcst.conservativeResize(skcst.rows()+1, skcst.cols());
    //                     skcst(skcst.rows()-1, 0) = (RC.skelver.row(ec_neighs[k]) - RC.skelver.row(i)).norm();
    //                 }
    //             }
    //         }
    //     }
    //     if (tricount == 0) break;

    //     Eigen::MatrixXd::Index minRow, minCol;
    //     skcst.minCoeff(&minRow, &minCol);
    //     int idx = minRow;
    //     Eigen::Vector2i edge = skeds.row(idx);

    //     edge_rows.row(0) = RC.skelver.row(edge(0));
    //     edge_rows.row(1) = RC.skelver.row(edge(1));
    //     RC.skelver.row(edge(0)) = edge_rows.colwise().mean();
    //     RC.skelver.row(edge(1)).setConstant(std::numeric_limits<double>::quiet_NaN());

    //     for (int k=0; k<Adj.rows(); k++) {
    //         if (Adj(edge(1), k) == 1) {
    //             Adj(edge(0), k) = 1;
    //             Adj(k, edge(0)) = 1;
    //         }
    //     }

    //     Adj.row(edge(1)) = Eigen::MatrixXi::Zero(1, Adj.cols());
    //     Adj.row(edge(1)) = Eigen::MatrixXi::Zero(Adj.rows(), 1);

    //     for (int r=0; r<RC.corresp.rows(); r++) {
    //         if (RC.corresp(r,0) == (double)edge(1)) {
    //             RC.corresp(r,0) = (double)edge(0);
    //         }
    //     }
    // }
    // RC.skeladj = Adj;
}




/* Helper Functions */

void RosaPoints::rosa_initialize(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals) {
    Eigen::Matrix3d M;
    Eigen::Vector3d normal_v;

    for (int i=0; i<pcd_size_; i++) {
        pset(i,0) = cloud->points[i].x;
        pset(i,1) = cloud->points[i].y;
        pset(i,2) = cloud->points[i].z;

        normal_v(0) = normals->points[i].normal_x;
        normal_v(1) = normals->points[i].normal_y;
        normal_v(2) = normals->points[i].normal_z;

        M = create_orthonormal_frame(normal_v);
        vset.row(i) = M.row(1); // Extracts a vector orthogonal to normal_v... This vector lies in the tanget plane of the structure...
    }
}

Eigen::Matrix3d RosaPoints::create_orthonormal_frame(Eigen::Vector3d &v) {
    // ! /* random process for generating orthonormal basis */
    v = v/v.norm();
    double TH_ZERO = 1e-10;
    // srand((unsigned)time(NULL));
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    M(0,0) = v(0); 
    M(0,1) = v(1); 
    M(0,2) = v(2);
    Eigen::Vector3d new_vec, temp_vec;

    // Seems inefficient to just iterate until satisfaction? - Rewrite using deterministic linear algebra (cross product method)?
    // The outer for loops finds an orthonormal basis
    for (int i=1; i<3; ++i) {
      new_vec.setRandom();
      new_vec = new_vec/new_vec.norm();

      while (abs(1.0 - v.dot(new_vec)) < TH_ZERO) {
        // Run until vector (not too parallel) is found... Avoid colinear vectors
        new_vec.setRandom();
        new_vec = new_vec / new_vec.norm();
      }

      // Gramm-Schmidt process to find orthogonal vectors
      for (int j=0; j<i; ++j) {
        temp_vec = (new_vec - new_vec.dot(M.row(j)) * (M.row(j).transpose()));
        new_vec = temp_vec/temp_vec.norm();
      }

      M(i,0) = new_vec(0);
      M(i,1) = new_vec(1);
      M(i,2) = new_vec(2);
    }

    return M;
}

Eigen::MatrixXd RosaPoints::compute_active_samples(int &idx, Eigen::Vector3d &p_cut, Eigen::Vector3d &v_cut) {
    // Returns indices of normals considered in the same plane (plane-slice)

    // idx: index of the current point in the point cloud
    // p_cut: point corresponding to idx
    // v_cut: a vector orthogonal to the surface normal at idx (from create_orthonormal_fram)
    
    Eigen::MatrixXd out_indxs(pcd_size_, 1);
    int out_size = 0; // Is incremented as points in the plane-slice is determined...
    std::vector<int> isoncut(pcd_size_, 0); // vector initialized with zeros

    // Sets the indices of isoncut that are in place-slice to 1 and resizes the vector.
    pcloud_isoncut(p_cut, v_cut, isoncut, RC.datas, pcd_size_);
    
    std::vector<int> queue;
    queue.reserve(pcd_size_);
    queue.emplace_back(idx);

    int curr;
    while (!queue.empty()) {
        curr = queue.back();
        queue.pop_back();
        isoncut[curr] = 2;
        out_indxs(out_size++, 0) = curr;

        for (size_t i = 0; i < RC.neighs[curr].size(); ++i) {
            if (isoncut[RC.neighs[curr][i]] == 1) {
                isoncut[RC.neighs[curr][i]] = 3;
                queue.emplace_back(RC.neighs[curr][i]);
            }
        }
    }

    out_indxs.conservativeResize(out_size, 1); // Reduces the size down to an array of indices corresponding to the active samples
    return out_indxs;
}

void RosaPoints::pcloud_isoncut(Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut, std::vector<int>& isoncut, double*& datas, int& size) {

    DataWrapper data;
    data.factory(datas, size); // datas size is 3 x size

    std::vector<double> p(3); 
    p[0] = p_cut(0); 
    p[1] = p_cut(1); 
    p[2] = p_cut(2);
    std::vector<double> n(3); 
    n[0] = v_cut(0); 
    n[1] = v_cut(1); 
    n[2] = v_cut(2);
    distance_query(data, p, n, delta, isoncut);
}

void RosaPoints::distance_query(DataWrapper& data, const std::vector<double>& Pp, const std::vector<double>& Np, double delta, std::vector<int>& isoncut) {
    std::vector<double> P(3);
    
    // data.lenght() is just pcd_size_
    for (int pIdx=0; pIdx < data.length(); pIdx++) {
        // retrieve current point
        data(pIdx, P);
        
        // check distance (fabs is floating point absolute value...)
        // Np is normal vector to plane of point Pp. Distance is calculated as d = Np (dot) (Pp - P)
        // using the plane equation: https://tutorial.math.lamar.edu/classes/calciii/eqnsofplanes.aspx
        if (fabs( Np[0]*(Pp[0]-P[0]) + Np[1]*(Pp[1]-P[1]) + Np[2]*(Pp[2]-P[2]) ) < delta) {
            isoncut[pIdx] = 1;
        }
    }
}

Eigen::Vector3d RosaPoints::compute_symmetrynormal(Eigen::MatrixXd& local_normals) {
    // This function determines the vector least variance amongst the local_normals. 
    // This can be interpreted as the "direction" of the skeleton inside the structure...

    Eigen::Matrix3d M; Eigen::Vector3d vec;
    double alpha = 0.0;
    int size = local_normals.rows();
    double Vxx, Vyy, Vzz, Vxy, Vyx, Vxz, Vzx, Vyz, Vzy;

    // Variances: Computing the mean squared value and substracting the mean squared value -> Variance = E[X²] - E[X]²
    Vxx = (1.0+alpha)*local_normals.col(0).cwiseAbs2().sum()/size - pow(local_normals.col(0).sum(), 2)/pow(size, 2);
    Vyy = (1.0+alpha)*local_normals.col(1).cwiseAbs2().sum()/size - pow(local_normals.col(1).sum(), 2)/pow(size, 2);
    Vzz = (1.0+alpha)*local_normals.col(2).cwiseAbs2().sum()/size - pow(local_normals.col(2).sum(), 2)/pow(size, 2);

    // Covariances: Computing the mean of the product of 2 components and subtracting the product of the means of each components -> Covariance = E[XY] - E[X]E[Y]
    Vxy = 2*(1.0+alpha)*(local_normals.col(0).cwiseProduct(local_normals.col(1))).sum()/size - 2*local_normals.col(0).sum()*local_normals.col(1).sum()/pow(size, 2);
    Vyx = Vxy;
    Vxz = 2*(1.0+alpha)*(local_normals.col(0).cwiseProduct(local_normals.col(2))).sum()/size - 2*local_normals.col(0).sum()*local_normals.col(2).sum()/pow(size, 2);
    Vzx = Vxz;
    Vyz = 2*(1.0+alpha)*(local_normals.col(1).cwiseProduct(local_normals.col(2))).sum()/size - 2*local_normals.col(1).sum()*local_normals.col(2).sum()/pow(size, 2);
    Vzy = Vyz;
    M << Vxx, Vxy, Vxz, Vyx, Vyy, Vyz, Vzx, Vzy, Vzz;

    // Perform singular-value-decomposition on the Covariance matrix M = U(Sigma)V^T
    Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU();
    // The last column of the matrix U corresponds to the smallest singular value (in Sigma)
    // This in turn represents the direction of smallest variance
    // I.e. for the plance slice -> plane normal. 
    vec = U.col(M.cols()-1);
    return vec;
}

double RosaPoints::symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals) {
    // Computes the variance of the local normal vectors projected onto a symmetric normal vector
    Eigen::VectorXd alpha;
    int num = local_normals.rows();
    
    // Eigen::MatrixXd repmat; 
    // repmat.resize(num,3); //repeated matrix... 
    // for (int i=0; i<num; ++i) {
    //     repmat.row(i) = symm_nor;
    // }
    
    // // sums the coef wise product of the normal vector and the neighbouring normals
    // alpha = local_normals.cwiseProduct(repmat).rowwise().sum(); // coefficient wise product m(i,j)*n(i,j) -> summation of each row
    
    Eigen::MatrixXd repmat = symm_nor.transpose().replicate(num, 1); // replicate with a row-factor of num and col-factor of 1-
    alpha = local_normals * symm_nor; // calculate the projection of each local normal on the symmetry normal...

    int n = alpha.size(); // same size as local_normals
    double var;
    
    // Calculate sample variance of the projections
    if (n>1) {
        var = (n+1)*(alpha.squaredNorm()/(n+1) - alpha.mean()*alpha.mean())/n;
    }
    else {
        var = alpha.squaredNorm()/(n+1) - alpha.mean()*alpha.mean();
    }

    return var;
}

Eigen::Vector3d RosaPoints::symmnormal_smooth(Eigen::MatrixXd& V, Eigen::MatrixXd& w) {
    // V: vset_ex = symmetry normals computed for the neighbours of a point
    // w: vvar_ex = reciprocal variances of local normal projections on symmetry normal

    Eigen::Matrix3d M; Eigen::Vector3d vec;
    double Vxx, Vyy, Vzz, Vxy, Vyx, Vxz, Vzx, Vyz, Vzy;

    // Variances: Summation of the elemet wise product (inner product) between variance and the squared abs value of the
    // sum(w(i)*V(i)²) --- Where V is either x,y, or z component of symmetry normal vector
    Vxx = (w.cwiseProduct(V.col(0).cwiseAbs2())).sum();
    Vyy = (w.cwiseProduct(V.col(1).cwiseAbs2())).sum();
    Vzz = (w.cwiseProduct(V.col(2).cwiseAbs2())).sum();

    // Covariances: Similarly
    // sum(w(i)*Vx(i)*Vy(i)) etc..
    Vxy = (w.cwiseProduct(V.col(0)).cwiseProduct(V.col(1))).sum();
    Vyx = Vxy;
    Vxz = (w.cwiseProduct(V.col(0)).cwiseProduct(V.col(2))).sum();
    Vzx = Vxz;
    Vyz = (w.cwiseProduct(V.col(1)).cwiseProduct(V.col(2))).sum();
    Vzy = Vyz;
    M << Vxx, Vxy, Vxz, Vyx, Vyy, Vyz, Vzx, Vzy, Vzz;

    // The variances are reciprocal meaning large variances contribute with smaller values in the summation...

    Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU();

    // The vector corresponding to the largest singular value (first column of U)
    // It represents the the vector of smallest variance amongst the symmetry normals in the neighbourhood of the current point. 
    vec = U.col(0);

    return vec;
}

Eigen::Vector3d RosaPoints::closest_projection_point(Eigen::MatrixXd& P, Eigen::MatrixXd& V) {
    // Takes points (P) and corresponding surface normal vectors (V)
    Eigen::Vector3d vec;
    Eigen::VectorXd Lix2, Liy2, Liz2;

    // Squared components of V
    Lix2 = V.col(0).cwiseAbs2();
    Liy2 = V.col(1).cwiseAbs2();
    Liz2 = V.col(2).cwiseAbs2();

    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    Eigen::Vector3d B = Eigen::Vector3d::Zero();

    M(0,0) = (Liy2+Liz2).sum(); 
    M(0,1) = -(V.col(0).cwiseProduct(V.col(1))).sum();
    M(0,2) = -(V.col(0).cwiseProduct(V.col(2))).sum();

    M(1,0) = -(V.col(1).cwiseProduct(V.col(0))).sum();
    M(1,1) = (Lix2 + Liz2).sum();
    M(1,2) = -(V.col(1).cwiseProduct(V.col(2))).sum();

    M(2,0) = -(V.col(2).cwiseProduct(V.col(0))).sum();
    M(2,1) = -(V.col(2).cwiseProduct(V.col(1))).sum();
    M(2,2) = (Lix2 + Liy2).sum();

    B(0) = (P.col(0).cwiseProduct(Liy2 + Liz2)).sum() - (V.col(0).cwiseProduct(V.col(1)).cwiseProduct(P.col(1))).sum() - (V.col(0).cwiseProduct(V.col(2)).cwiseProduct(P.col(2))).sum();
    B(1) = (P.col(1).cwiseProduct(Lix2 + Liz2)).sum() - (V.col(1).cwiseProduct(V.col(0)).cwiseProduct(P.col(0))).sum() - (V.col(1).cwiseProduct(V.col(2)).cwiseProduct(P.col(2))).sum();
    B(2) = (P.col(2).cwiseProduct(Lix2 + Liy2)).sum() - (V.col(2).cwiseProduct(V.col(0)).cwiseProduct(P.col(0))).sum() - (V.col(2).cwiseProduct(V.col(1)).cwiseProduct(P.col(1))).sum();

    if (std::abs(M.determinant()) < 1e-3) {
        vec << 1e8, 1e8, 1e8;
    }
    else {
        // Solving a least squares minimization problem to find the best fit projection point
        // vec = M^(-1) * B
        vec = M.inverse()*B;
    }

    return vec;
}

int RosaPoints::argmax_eigen(Eigen::MatrixXd &x) {
    Eigen::MatrixXd::Index maxRow, maxCol;
    x.maxCoeff(&maxRow,&maxCol);
    int idx = maxRow;
    return idx;
}