MAP_BUILDER = {
    num_background_threads = 4,
    pose_graph = {
        optimize_every_n_nodes = 1,
        constraint_builder = {
            min_local_search_score = 0.40,
            min_global_search_score = 0.40,
            constraint_translation_weight = 1.1e4,
            constraint_rotation_weight = 1e5,
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
                num_global_samples = 400,
                num_global_rotations = 16,
                proposal_max_score = 1.0,
                min_cluster_size = 2,
                min_cluster_distance = 3.0,
                num_local_samples = 100,
                local_sample_linear_distance = 0.2,
                local_sample_angular_distance = 0.2,
                icp_options = {
                    nearest_neighbour_point_huber_loss = 0.01,
                    nearest_neighbour_feature_huber_loss = 0.01,
                    point_pair_point_huber_loss = 0.01,
                    point_pair_feature_huber_loss = 0.01,
                    unmatched_feature_cost = 1.0,
                    point_weight = 1.0,
                    feature_weight = 2.0,
                }
            },
            min_icp_score = 0.95,
            min_scan_agreement_fraction = 0.40,
        },

        -- used when adding intra submap constraints
        matcher_translation_weight = 5e2,
        matcher_rotation_weight = 1.6e3,
        optimization_problem = {
            huber_scale = 1e1, -- only for inter-submap constraints

            local_slam_pose_translation_weight = 1e5,
            local_slam_pose_rotation_weight = 1e5,
            odometry_translation_weight = 1e5,
            odometry_rotation_weight = 1e5,
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

        min_globally_searched_constraints_for_trajectory = 2,
        min_local_constraints_for_submap = 1,
        min_global_constraints_for_submap = 1,
        max_constraint_match_distance = 8.0,
        max_work_queue_size = 10,
    },
    collate_by_trajectory = false,
}
