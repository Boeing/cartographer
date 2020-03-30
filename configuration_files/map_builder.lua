MAP_BUILDER = {
    num_background_threads = 1,
    pose_graph = {
        optimize_every_n_nodes = 1,
        constraint_builder = {
            min_local_search_score = 0.40,
            min_global_search_score = 0.45,

            -- used when adding INTER submap constraints
            constraint_translation_weight = 2,
            constraint_rotation_weight = 2,
            log_matches = true,
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
                num_global_samples = 200,
                num_global_rotations = 32,

                proposal_point_inlier_threshold = 0.4,
                proposal_feature_inlier_threshold = 0.4,

                proposal_min_points_inlier_fraction = 0.3,
                proposal_min_features_inlier_fraction = 0.5,

                proposal_features_weight = 10.0,
                proposal_points_weight = 1.0,

                proposal_raytracing_max_error = 1.0,

                proposal_max_points_error = 0.6,
                proposal_max_features_error = 0.6,
                proposal_max_error = 0.4,

                min_cluster_size = 3,
                min_cluster_distance = 0.5,

                num_local_samples = 40,

                local_sample_linear_distance = 0.2,
                local_sample_angular_distance = 0.2,

                icp_options = {
                    nearest_neighbour_point_huber_loss = 0.01,
                    nearest_neighbour_feature_huber_loss = 0.01,

                    point_pair_point_huber_loss = 0.01,
                    point_pair_feature_huber_loss = 0.01,

                    point_weight = 1.0,
                    feature_weight = 10.0,

                    point_inlier_threshold = 1.0,
                    feature_inlier_threshold = 1.0,
                }
            },
            min_icp_score = 0.98,
            min_icp_points_inlier_fraction = 0.3,
            min_icp_features_inlier_fraction = 0.5,
            min_hit_fraction = 0.50,
        },

        -- used when adding INTRA submap constraints
        -- these are from node to current submap and previous submap
        matcher_translation_weight = 1,
        matcher_rotation_weight = 1,
        optimization_problem = {
            huber_scale = 1e1, -- only for inter-submap constraints

            local_slam_pose_translation_weight = 1,
            local_slam_pose_rotation_weight = 1,

            odometry_translation_weight = 1,
            odometry_rotation_weight = 1,

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
        log_residual_histograms = true,
        global_constraint_search_after_n_seconds = 10000.,

        --  overlapping_submaps_trimmer_2d = {
        --    fresh_submaps_count = 1,
        --    min_covered_area = 2,
        --    min_added_submaps_count = 5,
        --  },

        -- global search is EXPENSIVE (~1-2seconds)

        -- keep searching globally until this many found in total
        min_globally_searched_constraints_for_trajectory = 4,

        -- keep searching locally until this many inside submap
        min_local_constraints_for_submap = 3,

        -- keep searching globally until this many inside submap
        min_global_constraints_for_submap = 1,

        max_constraint_match_distance = 9.0,
        max_work_queue_size = 10,
    },
    collate_by_trajectory = false,
}
