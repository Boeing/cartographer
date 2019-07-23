TRAJECTORY_BUILDER = {
    trajectory_builder_2d = {
        use_imu_data = false,
        min_range = 0.,
        max_range = 28.,
        min_z = -0.8,
        max_z = 2.,
        missing_data_ray_length = 12.,

        circle_feature_options = {
            detect_radii = {0.07}
        },

        num_accumulated_range_data = 1,

        -- used before scan matching
        voxel_filter_size = 0.01,

        -- used before scan matching
        adaptive_voxel_filter = {
            max_length = 0.5,
            min_num_points = 200,
            max_range = 50.,
        },
        use_online_correlative_scan_matching = false,
        real_time_correlative_scan_matcher = {
            linear_search_window = 0.1,
            angular_search_window = math.rad(20.),
            translation_delta_cost_weight = 1e-1,
            rotation_delta_cost_weight = 1e-1,
        },
        ceres_scan_matcher = {
            occupied_space_weight = 1.,
            translation_weight = 1.,
            rotation_weight = 1.,
            ceres_solver_options = {
                use_nonmonotonic_steps = false,
                max_num_iterations = 20,
                num_threads = 1,
            },
        },
        motion_filter = {
            max_time_seconds = 5.,
            max_distance_meters = 0.1,
            max_angle_radians = math.rad(10.0),
        },
        imu_gravity_time_constant = 10.,
        submaps = {
            num_range_data = 20,
            grid_options_2d = {
                grid_type = "PROBABILITY_GRID",
                resolution = 0.02,
            },
            range_data_inserter = {
                range_data_inserter_type = "PROBABILITY_GRID_INSERTER_2D",
                probability_grid_range_data_inserter = {
                    insert_free_space = true,
                    hit_probability = 0.55,
                    miss_probability = 0.49,
                },
                tsdf_range_data_inserter = {
                    truncation_distance = 0.3,
                    maximum_weight = 10.,
                    update_free_space = false,
                    normal_estimation_options = {
                        num_normal_samples = 4,
                        sample_radius = 0.5,
                    },
                    project_sdf_distance_to_scan_normal = true,
                    update_weight_range_exponent = 0,
                    update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0.5,
                    update_weight_distance_cell_to_hit_kernel_bandwidth = 0.5,
                },
            },
        },
    },
    --  pure_localization_trimmer = {
    --    max_submaps_to_keep = 3,
    --  },
    collate_fixed_frame = true,
    collate_landmarks = false,
}
