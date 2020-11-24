MAP_BUILDER = {
    pose_graph = {
        constraint_builder = {
            min_local_search_score = 0.40,
            min_global_search_score = 0.30,

            -- used when adding INTER submap constraints
            constraint_translation_weight = 2,
            constraint_rotation_weight = 2,
            ceres_scan_matcher = {
                occupied_space_weight = 1,
                translation_weight = 1,
                rotation_weight = 1,
                ceres_solver_options = {
                    use_nonmonotonic_steps = true,
                    max_num_iterations = 100,
                    num_threads = 1,
                },
            },
            global_icp_scan_matcher_options_2d = {
                num_global_samples_per_sq_m = 3,
                num_global_rotations = 128,

                proposal_point_inlier_threshold = 1.0,
                proposal_feature_inlier_threshold = 1.0,

                proposal_min_points_inlier_fraction = 0.4,
                proposal_min_features_inlier_fraction = 0.5,

                proposal_features_weight = 2.0,
                proposal_points_weight = 1.0,

                proposal_raytracing_max_error = 0.4,

                proposal_max_points_error = 0.5,
                proposal_max_features_error = 1.2,
                proposal_max_error = 0.5,

                min_cluster_size = 1,
                max_cluster_size = 4,
                min_cluster_distance = 0.4,

                num_local_samples = 8,

                local_sample_linear_distance = 0.2,
                local_sample_angular_distance = 0.1,

                icp_options = {
                    nearest_neighbour_point_huber_loss = 0.01,
                    nearest_neighbour_feature_huber_loss = 0.01,

                    point_pair_point_huber_loss = 0.01,
                    point_pair_feature_huber_loss = 0.01,

                    point_weight = 1.0,
                    feature_weight = 0.5,

                    point_inlier_threshold = 1.0,
                    feature_inlier_threshold = 1.0,

                    -- Used for evaluating match
                    raytrace_threshold = 0.3;
                    hit_threshold = 0.3;
                    feature_match_threshold = 0.2,
                }
            },
            min_icp_score = 0.97,
            min_icp_points_inlier_fraction = 0.5,
            min_icp_features_inlier_fraction = 0.5,
            min_hit_fraction = 0.5,
            min_ray_trace_fraction = 0.85,
            min_icp_features_match_fraction = 0.6,
        },

        -- used when adding INTRA submap constraints
        -- these are from node to current submap and previous submap
        matcher_translation_weight = 1,
        matcher_rotation_weight = 1,
        optimization_problem = {
            huber_scale = 1e1, -- only for inter-submap constraints

            -- these are between nodes based on front-end mapping
            local_slam_pose_translation_weight = 0,
            local_slam_pose_rotation_weight = 0,

            fixed_frame_pose_translation_weight = 1e1, -- only in 3d
            fixed_frame_pose_rotation_weight = 1e2, -- only in 3d

            log_solver_summary = false,
            ceres_solver_options = {
                use_nonmonotonic_steps = false,
                max_num_iterations = 50,
                num_threads = 7,
            },
        },
        max_num_final_iterations = 200,

        --  overlapping_submaps_trimmer_2d = {
        --    fresh_submaps_count = 1,
        --    min_covered_area = 2,
        --    min_added_submaps_count = 5,
        --  },

        -- keep searching globally until this many found in total
        min_globally_searched_constraints_for_trajectory = 1,

        -- keep searching locally until this many inside submap
        local_constraint_every_n_nodes = 8,

        -- keep searching globally until this many inside submap
        global_constraint_every_n_nodes = 8,

        max_constraint_match_distance = 9.0,
    },
    collate_by_trajectory = false,
}